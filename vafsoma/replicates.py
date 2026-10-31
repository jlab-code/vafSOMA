"""Replicate-group generalization of the vafSOMA mixture model.

Organs can be grouped into replicate groups (e.g. neighbouring leaves of one
branch, all derived from the same shoot apical meristem). A fixed mutation of
that meristem is expected in every replicate of the group at the same VAF,
whereas leaf-private variants and sampling noise are not. This module fits the
mixture to (variant, group) units instead of single (variant, organ)
observations:

* one latent state per unit, Z in {E_shared, E_private, L1, L2, L3};
* fixed layer k: replicates are independent binomial (or beta-binomial) draws
  at mu_k, ZERO counts included -- absence in a sibling leaf is evidence;
* E_shared: one background VAF q per unit, drawn from the truncated
  exponential and common to all replicates (recurrent artefacts);
* E_private: an independent q per replicate (leaf-private variants);
* every component is conditioned on >= 1 positive replicate in the unit, the
  group-level analogue of the positive-only conditioning of the base model.

With one organ per group (R_g = 1), E_shared and E_private coincide and the
unit likelihood equals the base model's positive-only per-observation
likelihood, so the model reduces exactly to the base model. The pipeline only
enters this code path when replicate groups are supplied; without them the
base code path runs unchanged.

Downstream, group units take the place of organs: counts (16) and posteriors (17)
are written per group, with L_g the mean root-to-tip length of the group's members,
and the depth-window metric averages N_g / (2 G L_g) over groups.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

from .model import (
    ExponentialComponentCache,
    FitResult,
    MuTriple,
    _default_pi_starts,
    active_mask_from_components,
    log_betabinom_pmf,
    read_params,
)
from .utils import COMPONENTS, LAYER_COMPONENTS

# Internal 5-component order; collapsed to the base COMPONENTS order
# (E = E_shared + E_private, L1, L2, L3) for all downstream outputs.
# 0.4.0: E_error (index 5; no replicate carries the variant, q = 0, all alt reads are sequencing errors) is active
# whenever read_error > 0; with read_error = 0 its weight is 0 and every result equals 0.3.0.
REP_COMPONENTS = ("E_shared", "E_private", "L1", "L2", "L3", "E_error")
KEY_COLS = ["variant_id", "chrom", "pos", "ref", "alt"]


# --------------------------------------------------------------------------- input
def read_replicate_groups(path: str | Path) -> dict[str, dict[str, str]]:
    """Read a tab- or comma-separated table with columns tree, sample, group.

    Returns {tree: {sample: group}}. Samples not listed for a tree are treated
    as singleton groups (one organ per group), i.e. the base model.
    """
    df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    cols = {c.lower().strip(): c for c in df.columns}
    for need in ("tree", "sample", "group"):
        if need not in cols:
            raise ValueError(f"replicate-groups file must have columns tree, sample, group; got {list(df.columns)}")
    out: dict[str, dict[str, str]] = {}
    for r in df.itertuples(index=False):
        tree = str(getattr(r, cols["tree"])).strip()
        sample = str(getattr(r, cols["sample"])).strip()
        group = str(getattr(r, cols["group"])).strip()
        out.setdefault(tree, {})
        if sample in out[tree] and out[tree][sample] != group:
            raise ValueError(f"sample {sample!r} of tree {tree!r} assigned to two groups")
        out[tree][sample] = group
    return out


def sample_to_group(
    replicate_groups: dict[str, dict[str, str]] | None,
    tree: str,
    samples: Sequence[str],
    control_set: set[str],
) -> dict[str, str]:
    """Map every sample of a tree to a group id. Unlisted samples and control
    organs are singleton groups; control organs are never pooled with others."""
    spec = (replicate_groups or {}).get(str(tree), {})
    out = {}
    for s in samples:
        if s in control_set:
            out[s] = f"control:{s}"
        else:
            out[s] = str(spec.get(s, s))
    used_control = [g for s, g in out.items() if s not in control_set and g.startswith("control:")]
    if used_control:
        raise ValueError("group ids may not start with 'control:'")
    return out


def tree_has_replicates(replicate_groups: dict[str, dict[str, str]] | None, tree: str) -> bool:
    return bool(replicate_groups) and str(tree) in replicate_groups


# --------------------------------------------------------------------------- units
def build_units(df_depth: pd.DataFrame, smap: dict[str, str]) -> dict:
    """(variant, group) units with >= 1 positive replicate.

    Returns padded arrays Y, D (n_units x R_max) with a validity mask M, plus
    unit metadata. Replicates absent from the input for a variant (no row) are
    masked out rather than treated as zeros.
    """
    df = df_depth[["variant_id", "chrom", "pos", "ref", "alt", "sample", "alt_depth", "DP"]].copy()
    df["sample"] = df["sample"].astype(str)
    df["group"] = df["sample"].map(smap)
    if df["group"].isna().any():
        missing = sorted(df.loc[df["group"].isna(), "sample"].unique())
        raise ValueError(f"samples without group assignment: {missing}")
    pos_units = df.loc[df["alt_depth"] > 0, ["variant_id", "group"]].drop_duplicates()
    df = df.merge(pos_units, on=["variant_id", "group"], how="inner")
    if df.empty:
        return dict(n=0)
    df = df.sort_values(["group", "variant_id", "sample"], kind="stable")
    df["rep_idx"] = df.groupby(["variant_id", "group"], sort=False).cumcount()
    keys = df[["variant_id", "group"]].drop_duplicates().reset_index(drop=True)
    keys["unit"] = np.arange(len(keys))
    df = df.merge(keys, on=["variant_id", "group"], how="left")
    n = len(keys)
    rmax = int(df["rep_idx"].max()) + 1
    Y = np.zeros((n, rmax), dtype=np.int64)
    D = np.zeros((n, rmax), dtype=np.int64)
    M = np.zeros((n, rmax), dtype=bool)
    S = np.full((n, rmax), "", dtype=object)     # 0.4.1: organ of each replicate (organ-specific composition)
    u = df["unit"].to_numpy()
    r = df["rep_idx"].to_numpy()
    Y[u, r] = df["alt_depth"].to_numpy(dtype=np.int64)
    D[u, r] = df["DP"].to_numpy(dtype=np.int64)
    M[u, r] = True
    S[u, r] = df["sample"].to_numpy()
    meta = df.drop_duplicates("unit").sort_values("unit")[["unit", "variant_id", "chrom", "pos", "ref", "alt", "group"]]
    n_rep = M.sum(1)
    n_pos = ((Y > 0) & M).sum(1)
    return dict(n=n, Y=Y, D=D, M=M, S=S, meta=meta.reset_index(drop=True), n_rep=n_rep, n_pos=n_pos)


# --------------------------------------------------------------------------- likelihood
def _log_binom(y: np.ndarray, d: np.ndarray, p: np.ndarray | float) -> np.ndarray:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return gammaln(d + 1.0) - gammaln(y + 1.0) - gammaln(d - y + 1.0) + y * np.log(p) + (d - y) * np.log1p(-p)


def _log1mexp(x: np.ndarray) -> np.ndarray:
    """log(1 - exp(x)) for x <= 0, numerically stable."""
    x = np.minimum(x, -1e-300)
    return np.where(x > -0.6931471805599453, np.log(-np.expm1(x)), np.log1p(-np.exp(x)))


class ReplicateLikelihood:
    """Precomputes lambda-independent quadrature terms for a set of units."""

    def __init__(self, units: dict, exp_cache: ExponentialComponentCache, by_sample: bool = False):
        self.cache = exp_cache
        # Collapse identical replicate vectors (Y, D, mask) into weighted rows, the
        # analogue of the base model's grouping of identical (y, d) observations.
        # 0.4.1: by_sample (organ-specific composition) keeps the organs of the replicates in the key.
        parts = [units["Y"], units["D"], units["M"].astype(np.int64)]
        names = None
        if by_sample and units.get("S") is not None:
            names, codes = np.unique(units["S"].astype(str), return_inverse=True)
            parts.append(np.asarray(codes).reshape(units["S"].shape).astype(np.int64))
        key = np.concatenate(parts, axis=1)
        uniq, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        self.inverse = np.asarray(inverse).ravel()
        self.counts = counts.astype(float)
        r = units["Y"].shape[1]
        self.u = dict(n=len(uniq), Y=uniq[:, :r], D=uniq[:, r:2 * r], M=uniq[:, 2 * r:3 * r].astype(bool),
                      S=(names[uniq[:, 3 * r:4 * r]] if names is not None else None))
        Y, D, M = self.u["Y"].astype(float), self.u["D"].astype(float), self.u["M"]
        q = np.clip(exp_cache.p, 1e-15, 1 - 1e-15)          # read probability at each node (0.4.0: incl. error)
        # per replicate x quadrature node: log Bin(y|d,q) and log Bin(0|d,q)
        lb = _log_binom(Y[..., None], D[..., None], q[None, None, :])
        lb0 = D[..., None] * np.log1p(-q)[None, None, :]
        lb = np.where(M[..., None], lb, 0.0)
        lb0 = np.where(M[..., None], lb0, 0.0)
        self._lb_rep = lb                      # (n, R, nq)
        self._lb0_rep = lb0
        self._lb_sum = lb.sum(1)               # (n, nq), shared-q product
        self._lb0_sum = lb0.sum(1)
        self.error_component = bool(getattr(exp_cache, "error_component", False))
        if self.error_component:
            e = exp_cache.read_error
            le = np.where(M, _log_binom(Y, D, e), 0.0).sum(1)
            le0 = np.where(M, D * np.log1p(-e), 0.0).sum(1)
            self._l_err = le - _log1mexp(le0)

    def comp_loglik(self, lam: float, mu: MuTriple, active_components: Sequence[str], rho: float = 0.0) -> np.ndarray:
        u, c = self.u, self.cache
        Y, D, M = u["Y"], u["D"], u["M"]
        base = c.logw + c._log_trunc_exp_density(lam)                      # (nq,)
        out = np.full((u["n"], len(REP_COMPONENTS)), -np.inf)
        # E_shared
        l_sh = logsumexp(self._lb_sum + base[None, :], axis=1)
        l_sh0 = logsumexp(self._lb0_sum + base[None, :], axis=1)
        out[:, 0] = l_sh - _log1mexp(l_sh0)
        # E_private
        m_rep = logsumexp(self._lb_rep + base[None, None, :], axis=2)        # (n, R)
        m0_rep = logsumexp(self._lb0_rep + base[None, None, :], axis=2)
        m_rep = np.where(M, m_rep, 0.0)
        m0_rep = np.where(M, m0_rep, 0.0)
        out[:, 1] = m_rep.sum(1) - _log1mexp(m0_rep.sum(1))
        # fixed layers
        active = set(active_components)
        for j, comp in enumerate(LAYER_COMPONENTS, start=2):
            if comp not in active:
                continue
            organ = mu.organ_scale is not None and u.get("S") is not None
            m, r = read_params(mu.values(comp, u["S"]) if organ else mu.value(comp), rho, c.read_error)
            if np.any(np.asarray(r) > 0):
                mm, rr = np.broadcast_to(m, Y.shape).ravel(), np.broadcast_to(r, Y.shape).ravel()
                lf = log_betabinom_pmf(Y.ravel(), D.ravel(), mm, rr).reshape(Y.shape)
                lf0 = log_betabinom_pmf(np.zeros(Y.size, dtype=int), D.ravel(), mm, rr).reshape(Y.shape)
            else:
                lf = _log_binom(Y.astype(float), D.astype(float), m)
                lf0 = D * np.log1p(-m)
            lf = np.where(M, lf, 0.0).sum(1)
            lf0 = np.where(M, lf0, 0.0).sum(1)
            out[:, j] = lf - _log1mexp(lf0)
        if self.error_component:
            out[:, 5] = self._l_err
        return out

    def comp_loglik_units(self, lam, mu, active_components, rho=0.0) -> np.ndarray:
        """Component log-likelihoods expanded back to one row per original unit."""
        return self.comp_loglik(lam, mu, active_components, rho)[self.inverse]


def _active_mask5(active_components: Sequence[str], error_component: bool = False) -> np.ndarray:
    a4 = active_mask_from_components(active_components)
    return np.array([a4[0], a4[0], a4[1], a4[2], a4[3], bool(error_component)], dtype=bool)


def collapse_pi(pi5: np.ndarray) -> np.ndarray:
    """Internal weights (E_shared, E_private, L1, L2, L3[, E_error]) -> (E, L1, L2, L3)."""
    e_err = pi5[5] if len(pi5) > 5 else 0.0
    return np.array([pi5[0] + pi5[1] + e_err, pi5[2], pi5[3], pi5[4]], dtype=float)


# --------------------------------------------------------------------------- EM
def em5(log_comp: np.ndarray, active5: np.ndarray, pi_init: np.ndarray, max_iter: int = 1000, tol: float = 1e-8,
        counts: np.ndarray | None = None):
    k = log_comp.shape[1]
    w = np.ones(log_comp.shape[0]) if counts is None else np.asarray(counts, dtype=float)
    wsum = w.sum()
    pi = np.where(active5, np.clip(pi_init, 1e-12, None), 0.0)
    pi /= pi.sum()
    lc = log_comp.copy()
    lc[:, ~active5] = -np.inf
    prev = -np.inf
    for it in range(max_iter):
        lp = np.full(k, -np.inf)
        lp[active5] = np.log(np.clip(pi[active5], 1e-300, None))
        lj = lc + lp
        den = logsumexp(lj, axis=1)
        resp = np.exp(lj - den[:, None])
        new = (w[:, None] * resp).sum(0) / wsum
        new = np.where(active5, np.clip(new, 1e-12, None), 0.0)
        pi = new / new.sum()
        lp[active5] = np.log(np.clip(pi[active5], 1e-300, None))
        ll = float((w * logsumexp(lc + lp, axis=1)).sum())
        if it > 0 and abs(ll - prev) <= tol * (1.0 + abs(prev)):
            return pi, ll, it + 1, True
        prev = ll
    return pi, float(prev), max_iter, False


def _starts5(active_components: Sequence[str], n_starts: int, seed: int, extra: list[np.ndarray] | None,
             error_component: bool = False) -> list[np.ndarray]:
    a4 = active_mask_from_components(active_components)
    out = []
    for p in _default_pi_starts(a4, n_random=max(0, n_starts), seed=seed):
        for w in (0.5, 0.1, 0.9):
            if error_component:          # a third of the background weight starts in E_error
                b = p[0] * 2 / 3
                out.append(np.array([b * (1 - w), b * w, p[1], p[2], p[3], p[0] / 3]))
            else:
                out.append(np.array([p[0] * (1 - w), p[0] * w, p[1], p[2], p[3], 0.0]))
    if extra:
        out = list(extra) + out
    return out


def _fit_at(lik: ReplicateLikelihood, lam, mu, active, rho, n_starts, seed, extra, max_iter, tol):
    lc = lik.comp_loglik(lam, mu, active, rho)
    a5 = _active_mask5(active, lik.error_component)
    best = None
    for s in _starts5(active, n_starts, seed, extra, lik.error_component):
        pi, ll, nit, conv = em5(lc, a5, s, max_iter=max_iter, tol=tol, counts=lik.counts)
        if best is None or ll > best[1]:
            best = (pi, ll, nit, conv)
    return best


def profile_fit_lambda_replicates(
    lik: ReplicateLikelihood,
    mu: MuTriple,
    lambda_grid,
    active_components: Sequence[str],
    rho: float = 0.0,
    em_n_starts: int = 16,
    em_random_seed: int = 1,
    em_max_iter: int = 1000,
    em_tol: float = 1e-8,
    refine: bool = True,
    n_zoom_levels: int = 2,
    n_zoom_points: int = 9,
):
    """Profile likelihood over lambda: coarse grid (reduced multistart, warm-started),
    then a zoom grid around the two best coarse points (as in the base model's search),
    and a final full-multistart refit at the best lambda."""
    grid = sorted({float(x) for x in lambda_grid})
    n_red = max(2, em_n_starts // 3)
    ev: dict[float, tuple] = {}
    warm = None
    for lam in grid:
        r = _fit_at(lik, lam, mu, active_components, rho, n_red, em_random_seed, warm, em_max_iter, em_tol)
        ev[lam] = r
        warm = [r[0]]
    if refine and len(grid) >= 3:
        top = sorted(grid, key=lambda l: ev[l][1], reverse=True)[:2]
        for lam0 in top:
            i = grid.index(lam0)
            lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
            for _ in range(n_zoom_levels):
                pts = np.exp(np.linspace(np.log(lo), np.log(hi), n_zoom_points))
                for lam in pts:
                    lam = float(lam)
                    if lam not in ev:
                        ev[lam] = _fit_at(lik, lam, mu, active_components, rho, n_red, em_random_seed,
                                          [ev[lam0][0]], em_max_iter, em_tol)
                best_local = max(pts, key=lambda l: ev[float(l)][1])
                j = int(np.argmin(np.abs(pts - best_local)))
                lo, hi = float(pts[max(j - 1, 0)]), float(pts[min(j + 1, len(pts) - 1)])
    lam_hat = max(ev, key=lambda l: ev[l][1])
    if refine and len(ev) >= 3:
        # final bounded polish inside the bracket of evaluated neighbours (log-lambda scale)
        from scipy.optimize import minimize_scalar
        ks = sorted(ev)
        i = ks.index(lam_hat)
        lo, hi = ks[max(i - 1, 0)], ks[min(i + 1, len(ks) - 1)]
        if hi > lo:
            warm = [ev[lam_hat][0]]
            res = minimize_scalar(
                lambda t: -_fit_at(lik, float(np.exp(t)), mu, active_components, rho, n_red, em_random_seed,
                                   warm, em_max_iter, em_tol)[1],
                bounds=(np.log(lo), np.log(hi)), method="bounded", options={"xatol": 1e-4},
            )
            lam_b = float(np.exp(res.x))
            ev[lam_b] = _fit_at(lik, lam_b, mu, active_components, rho, n_red, em_random_seed, warm, em_max_iter, em_tol)
            lam_hat = max(ev, key=lambda l: ev[l][1])
    pi, ll, nit, conv = _fit_at(lik, lam_hat, mu, active_components, rho, em_n_starts, em_random_seed,
                                [ev[lam_hat][0]], em_max_iter, em_tol)
    return lam_hat, pi, ll, nit, conv


def fit_replicate_depth(
    df_depth: pd.DataFrame,
    smap: dict[str, str],
    mu: MuTriple,
    lambda_grid,
    exp_cache: ExponentialComponentCache,
    active_components: Sequence[str],
    rho: float = 0.0,
    em_n_starts: int = 16,
    em_random_seed: int = 1,
    em_max_iter: int = 1000,
    em_tol: float = 1e-8,
    lambda_refine: bool = True,
):
    units = build_units(df_depth, smap)
    if units["n"] == 0:
        raise ValueError("No positive replicate units for this tree/depth; cannot fit model.")
    lik = ReplicateLikelihood(units, exp_cache, by_sample=mu.organ_scale is not None)
    lam, pi5, ll, nit, conv = profile_fit_lambda_replicates(
        lik, mu, lambda_grid, active_components, rho=rho, em_n_starts=em_n_starts,
        em_random_seed=em_random_seed, em_max_iter=em_max_iter, em_tol=em_tol, refine=lambda_refine,
    )
    pos = df_depth[df_depth["alt_depth"] > 0]
    fit = FitResult(
        mu=mu, lambda_hat=float(lam), pi=collapse_pi(pi5), pi_E_error=float(pi5[5]), loglik=float(ll),
        n_positive_rows=int(len(pos)), n_unique_positive_variants=int(pos["variant_id"].nunique()),
        grouped=pd.DataFrame(), active_components=tuple(active_components), rho=float(rho),
        n_iter=int(nit), converged=bool(conv),
    )
    any_multi = bool((units["n_rep"] > 1).any())
    extras = dict(
        pi5=pi5,
        pi_E_shared=float(pi5[0]) if any_multi else float("nan"),
        pi_E_private=float(pi5[1]) if any_multi else float("nan"),
        pi_E_error=float(pi5[5]),
        n_units=int(units["n"]),
        n_multi_replicate_units=int((units["n_rep"] > 1).sum()),
    )
    return fit, extras, units, lik


# --------------------------------------------------------------------------- classification
def _posteriors(lc: np.ndarray, pi5: np.ndarray, active5: np.ndarray) -> np.ndarray:
    lp = np.full(len(pi5), -np.inf)
    lp[active5] = np.log(np.clip(pi5[active5], 1e-300, None))
    a = lc + lp
    return np.exp(a - logsumexp(a, axis=1)[:, None])


def classify_group_units(fit: FitResult, extras: dict, units: dict, lik: ReplicateLikelihood) -> pd.DataFrame:
    """Per-(variant, group) posteriors, in the organ-calls schema with `sample` =
    group id, so the per-unit counts are formed per group."""
    lc = lik.comp_loglik_units(fit.lambda_hat, fit.mu, fit.active_components, fit.rho)
    post = _posteriors(lc, extras["pi5"], _active_mask5(fit.active_components, lik.error_component))
    meta = units["meta"]
    out = meta[KEY_COLS].copy()
    out["sample"] = meta["group"].astype(str).to_numpy()
    out["replicate_group"] = out["sample"]
    out["n_replicates"] = units["n_rep"]
    out["n_positive_replicates"] = units["n_pos"]
    out["P_E"] = post[:, 0] + post[:, 1] + post[:, 5]
    out["P_E_shared"] = post[:, 0]
    out["P_E_private"] = post[:, 1]
    if lik.error_component:
        out["P_E_error"] = post[:, 5]
    out["P_L1"], out["P_L2"], out["P_L3"] = post[:, 2], post[:, 3], post[:, 4]
    out["P_fixed"] = out["P_L1"] + out["P_L2"] + out["P_L3"]
    p4 = out[["P_E", "P_L1", "P_L2", "P_L3"]].to_numpy()
    labels = np.asarray(COMPONENTS, dtype=object)[np.argmax(p4, axis=1)]
    out["obs_inferred_component_MAP"] = labels
    out["obs_inferred_fixed_MAP"] = np.isin(labels, list(LAYER_COMPONENTS))
    out["obs_inferred_fixed_highconf_95"] = out["P_fixed"] >= 0.95
    return out


def classify_variants_pooled(fit: FitResult, extras: dict, units: dict, lik: ReplicateLikelihood) -> pd.DataFrame:
    """Tree-wide variant posterior: product of group-unit likelihoods under one
    variant-level state (the analogue of the base model's pooling over organs;
    identical to it when every group is a single organ)."""
    lc = lik.comp_loglik_units(fit.lambda_hat, fit.mu, fit.active_components, fit.rho)
    meta = units["meta"]
    vids, inv = np.unique(meta["variant_id"].to_numpy(), return_inverse=True)
    summed = np.zeros((len(vids), lc.shape[1]))
    np.add.at(summed, inv, np.where(np.isfinite(lc), lc, -1e300))
    a5 = _active_mask5(fit.active_components, lik.error_component)
    summed[:, ~a5] = -np.inf
    post = _posteriors(summed, extras["pi5"], a5)
    first = meta.drop_duplicates("variant_id").set_index("variant_id").loc[vids]
    out = pd.DataFrame({"variant_id": vids, "chrom": first["chrom"].to_numpy(), "pos": first["pos"].to_numpy(),
                        "ref": first["ref"].to_numpy(), "alt": first["alt"].to_numpy()})
    out["n_positive_observations"] = np.bincount(inv, weights=units["n_pos"]).astype(int)
    out["n_positive_groups"] = np.bincount(inv).astype(int)
    out["P_E"] = post[:, 0] + post[:, 1] + post[:, 5]
    out["P_L1"], out["P_L2"], out["P_L3"] = post[:, 2], post[:, 3], post[:, 4]
    out["P_fixed"] = out["P_L1"] + out["P_L2"] + out["P_L3"]
    p4 = out[["P_E", "P_L1", "P_L2", "P_L3"]].to_numpy()
    labels = np.asarray(COMPONENTS, dtype=object)[np.argmax(p4, axis=1)]
    out["variant_inferred_component_MAP"] = labels
    out["variant_inferred_fixed_MAP"] = np.isin(labels, list(LAYER_COMPONENTS))
    out["variant_inferred_fixed_highconf_95"] = out["P_fixed"] >= 0.95
    out["variant_inferred_layer_MAP"] = np.where(out["variant_inferred_fixed_MAP"], labels, "")
    return out


def group_lengths(length_by_sample: dict[str, float], smap: dict[str, str]) -> dict[str, float]:
    """Root-to-tip length per group: mean over its members."""
    acc: dict[str, list[float]] = {}
    for s, g in smap.items():
        if s in length_by_sample:
            acc.setdefault(g, []).append(float(length_by_sample[s]))
    return {g: float(np.mean(v)) for g, v in acc.items()}


def summarize_group_calls_window(organ_calls_all: pd.DataFrame, selected: pd.DataFrame, stable_fraction: float = 1.0) -> pd.DataFrame:
    """Window summary of group-level calls: one row per (tree, variant, group)."""
    if organ_calls_all.empty or "replicate_group" not in organ_calls_all.columns:
        return pd.DataFrame()
    rows = []
    for s in selected.itertuples(index=False):
        tree, a, b = str(s.tree), int(s.selected_window_start), int(s.selected_window_end)
        oc = organ_calls_all[(organ_calls_all["tree"].astype(str) == tree)
                             & organ_calls_all["depth_filter"].astype(int).between(a, b)]
        if oc.empty:
            continue
        n_cut = oc["depth_filter"].nunique()
        for (vid, grp), g in oc.groupby(["variant_id", "replicate_group"], sort=False):
            first = g.iloc[0]
            lab = g["obs_inferred_component_MAP"].astype(str)
            # 0.1.7: fractions over the depth cutoffs at which the unit is present (as in the per-organ and
            # per-variant summaries). 0.1.6 divided by all cutoffs of the window, so a unit that dropped out of
            # higher cutoffs (site depth filter) could never be called stably fixed.
            n_here = int(g["depth_filter"].nunique())
            frac_fixed = float(g["obs_inferred_fixed_MAP"].astype(bool).sum()) / n_here
            fr = {L: float((lab == L).sum()) / n_here for L in LAYER_COMPONENTS}
            stable_layer = next((L for L in LAYER_COMPONENTS if fr[L] >= stable_fraction), "")
            if stable_layer:
                status = f"called_fixed_{stable_layer}"
            elif frac_fixed >= stable_fraction:
                status = "called_fixed_ambiguous_layer"
            elif frac_fixed <= 1 - stable_fraction:
                status = "called_noise"
            else:
                status = "evaluated_unstable"
            rows.append(dict(
                dataset_label=first.get("dataset_label", ""), tree=tree, chrom=first["chrom"], pos=first["pos"],
                ref=first["ref"], alt=first["alt"], replicate_group=grp, selected_window_start=a,
                selected_window_end=b, n_dp_cutoffs_present=len(g), n_replicates=int(g["n_replicates"].max()),
                max_positive_replicates=int(g["n_positive_replicates"].max()),
                fraction_MAP_fixed=frac_fixed, **{f"fraction_MAP_{L}": fr[L] for L in LAYER_COMPONENTS},
                mean_P_E=float(g["P_E"].mean()), mean_P_E_shared=float(g["P_E_shared"].mean()),
                mean_P_E_private=float(g["P_E_private"].mean()),
                **{f"mean_P_{L}": float(g[f"P_{L}"].mean()) for L in LAYER_COMPONENTS},
                mean_P_fixed=float(g["P_fixed"].mean()), calling_status=status,
            ))
    return pd.DataFrame(rows)
