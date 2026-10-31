from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, logsumexp

from .utils import COMPONENTS, LAYER_COMPONENTS


@dataclass(frozen=True)
class MuTriple:
    L1: float | None = 0.10
    L2: float | None = 0.36
    L3: float | None = 0.04
    # 0.4.1: organ-specific layer composition. Tuple of (sample, (s_L1, s_L2, s_L3)); the expected VAF of layer k in
    # that sample is value(k) * s_k, with s_k = a_k(organ) / a_k(common). None (default) = one composition for all organs,
    # identical to 0.4.0. Samples not listed use s = 1.
    organ_scale: tuple | None = None

    def as_tuple(self) -> tuple[float | None, float | None, float | None]:
        return (self.L1, self.L2, self.L3)

    def as_dict(self) -> dict[str, float | None]:
        return {"L1": self.L1, "L2": self.L2, "L3": self.L3}

    def value(self, component: str) -> float:
        val = self.as_dict().get(component)
        if val is None or not np.isfinite(float(val)) or float(val) <= 0 or float(val) >= 1:
            raise ValueError(f"Invalid active-component mu for {component}: {val}")
        return float(val)

    def scale_map(self) -> dict[str, tuple[float, float, float]]:
        return {str(k): tuple(float(x) for x in v) for k, v in (self.organ_scale or ())}

    def for_sample(self, sample: str) -> "MuTriple":
        """Expected VAFs of one organ (organ_scale applied; returns a MuTriple without organ_scale)."""
        sc = self.scale_map().get(str(sample))
        if sc is None:
            return MuTriple(self.L1, self.L2, self.L3)
        vals = [None if v is None else float(np.clip(v * s, MU_ORGAN_MIN, MU_ORGAN_MAX))
                for v, s in zip(self.as_tuple(), sc)]
        return MuTriple(*vals)

    def values(self, component: str, samples=None):
        """Expected VAF of a layer: a float, or one value per row when organ_scale is set and samples are given."""
        v = self.value(component)
        if self.organ_scale is None or samples is None:
            return v
        j = LAYER_INDEX[component]
        sm = self.scale_map()
        s = np.array([sm.get(str(x), (1.0, 1.0, 1.0))[j] for x in np.asarray(samples, dtype=object).ravel()], dtype=float)
        return np.clip(v * s, MU_ORGAN_MIN, MU_ORGAN_MAX).reshape(np.shape(samples))


MU_ORGAN_MIN, MU_ORGAN_MAX = 1e-6, 0.999
LAYER_INDEX = {"L1": 0, "L2": 1, "L3": 2}


@dataclass
class FitResult:
    mu: MuTriple
    lambda_hat: float
    pi: np.ndarray
    loglik: float
    n_positive_rows: int
    n_unique_positive_variants: int
    grouped: pd.DataFrame
    active_components: tuple[str, ...] = COMPONENTS
    rho: float = 0.0
    n_iter: int = 0
    converged: bool = True
    # 0.4.0: weight of the error-only background (q = 0); part of pi[0] (E). Nonzero only with read_error > 0.
    pi_E_error: float = 0.0

    @property
    def error_frac(self) -> float:
        """Share of the background weight E that is error-only."""
        return float(self.pi_E_error / self.pi[0]) if self.pi[0] > 0 else 0.0


@dataclass
class DepthResult:
    fit_rows: list[dict]
    variant_calls: pd.DataFrame
    tier_row: dict


def active_mask_from_components(active_components: Sequence[str]) -> np.ndarray:
    active = set(active_components)
    if "E" not in active:
        raise ValueError("The exponential background component E must always be active.")
    unknown = active.difference(COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown active components: {sorted(unknown)}")
    return np.asarray([c in active for c in COMPONENTS], dtype=bool)


def log_binom_pmf(y: np.ndarray, d: np.ndarray, p: float | np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    d = np.asarray(d, dtype=float)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return gammaln(d + 1) - gammaln(y + 1) - gammaln(d - y + 1) + y * np.log(p) + (d - y) * np.log1p(-p)


def log_positive_mass_binom(d: np.ndarray, p) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)   # 0.4.1: p may vary by row
    log_p0 = d * np.log1p(-p)
    return np.log(-np.expm1(log_p0))


def log_betabinom_pmf(y: np.ndarray, d: np.ndarray, mu: float, rho: float) -> np.ndarray:
    """Beta-binomial pmf in mean/dispersion parametrization.

    rho in (0,1) is the intra-observation correlation (overdispersion) parameter;
    rho -> 0 recovers the binomial. alpha = mu*(1-rho)/rho, beta = (1-mu)*(1-rho)/rho.
    """
    y = np.asarray(y, dtype=float)
    d = np.asarray(d, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-12, 1 - 1e-12)   # 0.4.1: mu and rho may vary by row
    rho = np.clip(np.asarray(rho, dtype=float), 1e-9, 1 - 1e-9)
    s = (1.0 - rho) / rho
    alpha = mu * s
    beta = (1.0 - mu) * s
    log_coeff = gammaln(d + 1) - gammaln(y + 1) - gammaln(d - y + 1)
    return (
        log_coeff
        + gammaln(y + alpha)
        + gammaln(d - y + beta)
        - gammaln(d + alpha + beta)
        - (gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta))
    )


def log_positive_mass_betabinom(d: np.ndarray, mu: float, rho: float) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-12, 1 - 1e-12)
    rho = np.clip(np.asarray(rho, dtype=float), 1e-9, 1 - 1e-9)
    s = (1.0 - rho) / rho
    alpha = mu * s
    beta = (1.0 - mu) * s
    # BetaBinom(0 | d, alpha, beta) = B(alpha, d+beta) / B(alpha, beta)
    log_p0 = gammaln(d + beta) - gammaln(d + alpha + beta) + gammaln(alpha + beta) - gammaln(beta)
    return np.log(-np.expm1(log_p0))


def read_params(mu: float, rho: float = 0.0, read_error: float = 0.0) -> tuple[float, float]:
    """Mean and overdispersion of the read probability for a site with allele fraction mu (0.4.0).

    Observation model of the reads: the allele fraction is
    overdispersed, v ~ Beta with mean mu and intra-class correlation rho (rho = 1 / (concentration + 1)), and each
    read reports the other allele with the symmetric sequencing error e: p = v (1 - e) + (1 - v) e. The read count is
    represented by a beta-binomial with the matching mean m = e + (1 - 2e) mu and dispersion
    rho' = (1 - 2e)^2 mu (1 - mu) rho / (m (1 - m)) (exact for e = 0 or rho = 0; moment-matched otherwise).
    """
    e = float(read_error)
    scalar = np.ndim(mu) == 0                  # 0.4.1: mu may be an array (organ-specific composition)
    mu = float(mu) if scalar else np.asarray(mu, dtype=float)
    if e <= 0:
        return mu, float(rho or 0.0)
    m = e + (1.0 - 2.0 * e) * mu
    if not rho or rho <= 0:
        return m, 0.0
    r = (1.0 - 2.0 * e) ** 2 * mu * (1.0 - mu) * float(rho) / (m * (1.0 - m))
    return m, r


def layer_loglik(y: np.ndarray, d: np.ndarray, mu: float, rho: float = 0.0, read_error: float = 0.0) -> np.ndarray:
    """Conditional-on-positive log-likelihood for a fixed-layer component.

    rho == 0 (the default) uses the plain binomial.
    rho > 0 switches to a beta-binomial with that overdispersion (optional
    extension).
    read_error > 0 (0.4.0) adds the symmetric per-read sequencing error (see read_params).
    """
    mu, rho = read_params(mu, rho, read_error)
    if np.any(np.asarray(rho) > 0):          # rho is an array when mu varies by row (0.4.1)
        return log_betabinom_pmf(y, d, mu, rho) - log_positive_mass_betabinom(d, mu, rho)
    return log_binom_pmf(y, d, mu) - log_positive_mass_binom(d, mu)


class ExponentialComponentCache:
    """Gauss-Legendre quadrature cache for the conditional-positive exponential component."""

    def __init__(self, q_max: float = 0.5, n_quad: int = 96, read_error: float = 0.0):
        if q_max <= 0 or q_max >= 1:
            raise ValueError("q_max must be in (0, 1).")
        if n_quad < 16:
            raise ValueError("n_quad should be at least 16 for stable integration.")
        self.q_max = float(q_max)
        self.n_quad = int(n_quad)
        nodes, weights = np.polynomial.legendre.leggauss(self.n_quad)
        self.q = 0.5 * self.q_max * (nodes + 1.0)
        self.w = 0.5 * self.q_max * weights
        self.logw = np.log(self.w)
        # 0.4.0: symmetric per-read sequencing error; reads at background VAF q report the allele with
        # probability e + (1 - 2e) q (e = 0 reproduces 0.3.0 exactly)
        if not (0.0 <= float(read_error) < 0.5):
            raise ValueError("read_error must be in [0, 0.5).")
        self.read_error = float(read_error)
        # 0.4.0: with e > 0 the background has an error-only part (q = 0: the organ or group does not carry the variant
        # and every alt read is a sequencing error), with its own mixture weight. It is part of the model whenever
        # e > 0; with e = 0 it does not exist and the model is the 0.3.0 model.
        self.error_component = self.read_error > 0
        self.p = self.read_error + (1.0 - 2.0 * self.read_error) * self.q

    def _log_trunc_exp_density(self, lam: float) -> np.ndarray:
        lam = float(lam)
        if lam <= 0:
            raise ValueError("lambda must be > 0.")
        z = -np.expm1(-lam * self.q_max)
        return np.log(lam) - lam * self.q - np.log(z)

    def loglik(self, y: np.ndarray, d: np.ndarray, lam: float) -> np.ndarray:
        """Vectorized conditional-positive exponential-component log-likelihood."""
        y = np.asarray(y, dtype=int)
        d = np.asarray(d, dtype=int)
        lam = float(lam)
        if len(y) == 0:
            return np.asarray([], dtype=float)

        logg = self._log_trunc_exp_density(lam)
        base = self.logw + logg

        yy = y[:, None].astype(float)
        dd = d[:, None].astype(float)
        q = self.p[None, :]
        q_clip = np.clip(q, 1e-15, 1 - 1e-15)

        log_coeff_y = gammaln(dd + 1) - gammaln(yy + 1) - gammaln(dd - yy + 1)
        log_pmf_y = log_coeff_y + yy * np.log(q_clip) + (dd - yy) * np.log1p(-q_clip)
        log_i_y = logsumexp(base[None, :] + log_pmf_y, axis=1)

        log_pmf_0 = dd * np.log1p(-q_clip)
        log_i0 = logsumexp(base[None, :] + log_pmf_0, axis=1)
        prob_i0 = np.clip(np.exp(log_i0), 0.0, 1.0 - 1e-15)
        log_pos = np.log1p(-prob_i0)
        return log_i_y - log_pos


def error_only_loglik(y: np.ndarray, d: np.ndarray, read_error: float) -> np.ndarray:
    """Conditional-on-positive log-likelihood of the error-only background: Binomial(d, e) given y >= 1."""
    return log_binom_pmf(y, d, read_error) - log_positive_mass_binom(d, read_error)


def make_component_loglik(
    y: np.ndarray,
    d: np.ndarray,
    mu: MuTriple,
    lam: float,
    exp_cache: ExponentialComponentCache,
    active_components: Sequence[str] = COMPONENTS,
    rho: float = 0.0,
    error_frac: float = 0.0,
    split_error: bool = False,
    samples=None,
) -> np.ndarray:
    """Return 4-column log-likelihood array for E/L1/L2/L3.

    Inactive biological layer components are set to -inf, not given mu=0.
    This is important because conditional-on-positive binomial likelihoods are
    undefined when mu=0.

    rho is the shared beta-binomial overdispersion for the L1/L2/L3 layer
    components (0.0 = plain binomial).

    samples (0.4.1): the organ of each row; used only if mu has an organ_scale (organ-specific composition).
    """
    y = np.asarray(y, dtype=int)
    d = np.asarray(d, dtype=int)
    active = set(active_components)
    logs = np.full((len(y), len(COMPONENTS)), -np.inf, dtype=float)
    logs[:, 0] = exp_cache.loglik(y, d, lam)
    for j, comp in enumerate(COMPONENTS[1:], start=1):
        if comp in active:
            logs[:, j] = layer_loglik(y, d, mu.values(comp, samples), rho=rho, read_error=exp_cache.read_error)
    if exp_cache.error_component:
        l_err = error_only_loglik(y, d, exp_cache.read_error)
        if split_error:
            # fitting: the error-only background as a fifth column with its own weight (internal order E, L1, L2, L3,
            # E_error); collapsed into E for every output
            return np.column_stack([logs, l_err])
        if error_frac > 0:
            w = min(max(float(error_frac), 0.0), 1.0)
            with np.errstate(divide="ignore"):
                logs[:, 0] = np.logaddexp(np.log1p(-w) + logs[:, 0] if w < 1 else -np.inf, np.log(w) + l_err)
    return logs


def _row_samples(pos: pd.DataFrame, fit: "FitResult"):
    """Organ of each row if the fit has an organ-specific composition (0.4.1), else None."""
    if fit.mu.organ_scale is None or "sample" not in pos.columns:
        return None
    return pos["sample"].astype(str).to_numpy()


def fit_component_loglik(y: np.ndarray, d: np.ndarray, fit: "FitResult", exp_cache: ExponentialComponentCache,
                         samples=None) -> np.ndarray:
    """4-column (E, L1, L2, L3) log-likelihoods under a fitted model; E includes the error-only part.
    samples: organ of each row (0.4.1; needed only with an organ-specific composition)."""
    return make_component_loglik(y, d, fit.mu, fit.lambda_hat, exp_cache, active_components=fit.active_components,
                                 rho=fit.rho, error_frac=fit.error_frac, samples=samples)


def _normalize_pi(pi: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    pi = np.asarray(pi, dtype=float).copy()
    k = len(active_mask)
    if pi.shape[0] != k:
        raise ValueError(f"pi_init must have length {k}")
    pi[~active_mask] = 0.0
    pi[active_mask] = np.clip(pi[active_mask], 1e-12, None)
    total = pi.sum()
    if not np.isfinite(total) or total <= 0:
        pi = np.zeros(k, dtype=float)
        pi[active_mask] = 1.0 / active_mask.sum()
        return pi
    return pi / total


def _default_pi_starts(
    active_mask: np.ndarray,
    n_random: int = 16,
    seed: int = 1,
    extra_starts: Sequence[np.ndarray] | None = None,
) -> list[np.ndarray]:
    starts: list[np.ndarray] = []
    active_idx = np.where(active_mask)[0]

    uniform = np.zeros(len(active_mask), dtype=float)
    uniform[active_idx] = 1.0 / len(active_idx)
    starts.append(uniform)

    # Background-heavy start; useful when low-VAF observations can otherwise be
    # captured spuriously by L3.
    bg = np.zeros(len(active_mask), dtype=float)
    bg[0] = 0.995 if active_mask[0] else 0.0
    layer_idx = [i for i in active_idx if i != 0]
    if layer_idx:
        rem = 1.0 - bg[0]
        for i in layer_idx:
            bg[i] = rem / len(layer_idx)
    starts.append(bg)

    # 0.4.0: with an error-only column (index 4), a start that splits the background between the two parts.
    if len(active_mask) > len(COMPONENTS) and active_mask[len(COMPONENTS)]:
        layer_idx = [i for i in layer_idx if i < len(COMPONENTS)]
        sp = np.zeros(len(active_mask), dtype=float)
        sp[0], sp[len(COMPONENTS)] = 0.6, 0.38
        for i in layer_idx:
            sp[i] = 0.02 / max(len(layer_idx), 1)
        starts.append(sp)

    # One layer-heavy start per active layer.
    for i in layer_idx:
        p = np.zeros(len(active_mask), dtype=float)
        p[0] = 0.05
        p[i] = 0.90
        others = [j for j in active_idx if j not in (0, i)]
        if others:
            for j in others:
                p[j] = 0.05 / len(others)
        else:
            p[0] = 0.10
            p[i] = 0.90
        starts.append(p)

    if extra_starts:
        for p in extra_starts:
            starts.append(np.asarray(p, dtype=float))

    if n_random > 0:
        rng = np.random.default_rng(seed)
        alpha = np.ones(len(active_idx), dtype=float)
        # Mildly favor the background in some random starts but not all.
        if 0 in active_idx:
            alpha[np.where(active_idx == 0)[0][0]] = 3.0
        for _ in range(int(n_random)):
            p = np.zeros(len(active_mask), dtype=float)
            p[active_idx] = rng.dirichlet(alpha)
            starts.append(p)

    # Deduplicate after normalization.
    out: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for p in starts:
        q = _normalize_pi(p, active_mask)
        key = tuple(np.round(q, 12))
        if key not in seen:
            seen.add(key)
            out.append(q)
    return out


def em_fit_mixture(
    log_comp: np.ndarray,
    counts: np.ndarray,
    pi_init: np.ndarray | None = None,
    max_iter: int = 1000,
    tol: float = 1e-8,
    active_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float, int]:
    counts = np.asarray(counts, dtype=float)
    k = log_comp.shape[1]
    if k not in (len(COMPONENTS), len(COMPONENTS) + 1):
        raise ValueError(f"log_comp must have {len(COMPONENTS)} (or, with the error-only column, {len(COMPONENTS) + 1}) columns")
    if active_mask is None:
        active_mask = np.ones(k, dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool)

    if active_mask.sum() < 1:
        raise ValueError("At least one active component is required.")
    if pi_init is None:
        pi = np.zeros(k, dtype=float)
        pi[active_mask] = 1.0 / active_mask.sum()
    else:
        pi = _normalize_pi(np.asarray(pi_init, dtype=float), active_mask)

    log_comp_fit = log_comp.copy()
    log_comp_fit[:, ~active_mask] = -np.inf

    prev = -np.inf
    for it in range(max_iter):
        log_pi = np.full(k, -np.inf, dtype=float)
        log_pi[active_mask] = np.log(np.clip(pi[active_mask], 1e-300, None))
        log_joint = log_comp_fit + log_pi[None, :]
        log_den = logsumexp(log_joint, axis=1)
        resp = np.exp(log_joint - log_den[:, None])
        resp[:, ~active_mask] = 0.0
        weighted = counts[:, None] * resp
        new_pi = np.zeros(k, dtype=float)
        new_pi[active_mask] = weighted[:, active_mask].sum(axis=0) / counts.sum()
        new_pi = _normalize_pi(new_pi, active_mask)
        pi = new_pi
        log_pi = np.full(k, -np.inf, dtype=float)
        log_pi[active_mask] = np.log(np.clip(pi[active_mask], 1e-300, None))
        new_log_den = logsumexp(log_comp_fit + log_pi[None, :], axis=1)
        loglik = float(np.sum(counts * new_log_den))
        # NOTE: `prev` starts at -inf. Without the `it > 0` guard, the very
        # first iteration satisfies abs(loglik - prev) <= tol * (1 + abs(prev))
        # trivially, because both sides evaluate to +inf under IEEE-754
        # arithmetic (inf <= inf is True), so the loop would return after a
        # single M-step regardless of tol. See the walkthrough in the PR/issue
        # this fix accompanies for an empirical before/after comparison.
        if it > 0 and abs(loglik - prev) <= tol * (1.0 + abs(prev)):
            return pi, loglik, it + 1
        prev = loglik
    return pi, float(prev), max_iter


def em_fit_mixture_multistart(
    log_comp: np.ndarray,
    counts: np.ndarray,
    active_mask: np.ndarray,
    n_starts: int = 16,
    seed: int = 1,
    extra_starts: Sequence[np.ndarray] | None = None,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> tuple[np.ndarray, float, int]:
    starts = _default_pi_starts(active_mask, n_random=max(0, n_starts), seed=seed, extra_starts=extra_starts)
    best_pi: np.ndarray | None = None
    best_ll = -np.inf
    best_iter = 0
    for start in starts:
        pi, ll, n_iter = em_fit_mixture(
            log_comp,
            counts,
            pi_init=start,
            max_iter=max_iter,
            tol=tol,
            active_mask=active_mask,
        )
        if ll > best_ll:
            best_pi = pi.copy()
            best_ll = float(ll)
            best_iter = int(n_iter)
    assert best_pi is not None
    return best_pi, best_ll, best_iter


def _profile_lambda_search(
    grouped: pd.DataFrame,
    mu: MuTriple,
    lambda_grid: Iterable[float],
    exp_cache: ExponentialComponentCache,
    em_max_iter: int = 1000,
    em_tol: float = 1e-8,
    active_components: Sequence[str] = COMPONENTS,
    em_n_starts: int = 16,
    em_random_seed: int = 1,
    rho: float = 0.0,
    refine: bool = True,
    refine_top_k: int = 2,
    refine_n_starts: int | None = None,
    n_zoom_levels: int = 2,
    n_zoom_points: int = 9,
    zoom_shrink: float = 2.0,
) -> tuple[float, np.ndarray, float, int, bool, dict]:
    """Core lambda-profiling search, with optional diagnostics returned.

    Stage 1 (coarse): evaluate the profile log-likelihood ll(lambda) = max_pi
    loglik(pi, lambda) at every point of `lambda_grid` (warm-starting each EM
    fit's multistart set with the previous grid point's best pi, as before).

    Stage 2 (refine, default on): a fixed grid -- however fine -- necessarily
    quantizes lambda_hat to a grid point, and a too-coarse grid can miss the
    true optimum entirely (this is exactly what motivated this rewrite: a
    25-point grid landed on a visibly worse optimum than a 200-point grid at
    some tree/depth combinations, both in lambda_hat and in the downstream
    pi_L3). For each of the top `refine_top_k` coarse points, this localizes
    the optimum inside the bracket formed by its immediate coarse-grid
    neighbors using a "zoom grid": a small log-spaced grid of `n_zoom_points`
    is evaluated across the bracket, the best point's own immediate neighbors
    become the next (narrower) bracket, and this repeats for `n_zoom_levels`
    passes before a final short bounded Brent polish inside the
    now-very-narrow last bracket.

    An earlier version of this stage used a single bounded 1-D optimizer
    (scipy's Brent-type "bounded" method) directly on the wide coarse-point
    bracket. That is WRONG in general: the profile log-likelihood ll(lambda)
    is not guaranteed to be unimodal (two synthetic-data regression tests
    below demonstrate a real case with two competing local optima ~1.8x apart
    in lambda), and a 1-D optimizer that assumes unimodality can converge to
    the wrong (lower-likelihood) mode when both live inside one coarse
    bracket. An exhaustive-at-each-scale zoom grid does not make that
    assumption -- it resolves the bracket at successively finer scales, so a
    second mode close to the best coarse point is still found as long as the
    zoom grid's resolution is finer than the coarse grid's original spacing
    (true by construction, since each zoom level's bracket is already
    narrower than the previous one). Only the FINAL Brent polish -- run on
    the already-narrow bracket after zooming, where a second competing mode
    at that scale is implausible -- assumes local unimodality.

    Zoom-grid trials use a reduced EM multistart budget by default
    (`refine_n_starts`, default max(2, em_n_starts // 3)) -- unlike an earlier
    version of this function, which reduced starts under a single Brent
    search and got the wrong answer, reducing starts here is safe because the
    zoom grid (not the start count) is what finds the right region: the
    empirical test below found the identical, correct optimum whether
    refine_n_starts was 2 or the full em_n_starts, just faster. The FINAL
    chosen point is always re-fit once with the full em_n_starts budget
    before being returned, so the reported pi still gets the same multistart
    guarantee as a full grid point regardless of how the search located it.
    See test_lambda_refine.py for the empirical checks that motivated both of
    these design choices.
    """
    y = grouped["alt_depth"].to_numpy(dtype=int)
    d = grouped["DP"].to_numpy(dtype=int)
    counts = grouped["n_rows"].to_numpy(dtype=float)
    samples = grouped["sample"].to_numpy() if (mu.organ_scale is not None and "sample" in grouped.columns) else None
    active_mask = active_mask_from_components(active_components)
    if exp_cache.error_component:
        active_mask = np.append(active_mask, True)

    lam_list_sorted = sorted({float(l) for l in lambda_grid})
    if not lam_list_sorted:
        raise ValueError("lambda_grid must contain at least one value.")

    def _fit_at(lam: float, n_starts: int, seed: int, extra_starts: list[np.ndarray] | None):
        log_comp = make_component_loglik(
            y, d, mu, float(lam), exp_cache, active_components=active_components, rho=rho, split_error=True,
            samples=samples,
        )
        return em_fit_mixture_multistart(
            log_comp,
            counts,
            active_mask=active_mask,
            n_starts=n_starts,
            seed=seed,
            extra_starts=extra_starts,
            max_iter=em_max_iter,
            tol=em_tol,
        )

    evaluations: dict[float, tuple[np.ndarray, float, int]] = {}

    def _eval_and_store(lam: float, n_starts: int, seed: int, extra_starts: list[np.ndarray] | None) -> float:
        lam = float(lam)
        if lam not in evaluations:
            pi, ll, n_iter = _fit_at(lam, n_starts, seed, extra_starts)
            evaluations[lam] = (pi, ll, n_iter)
        return evaluations[lam][1]

    previous_best_pi: np.ndarray | None = None
    for idx, lam in enumerate(lam_list_sorted):
        extra = [previous_best_pi] if previous_best_pi is not None else None
        _eval_and_store(lam, em_n_starts, em_random_seed + idx, extra)
        previous_best_pi = evaluations[lam][0].copy()

    n_coarse_evals = len(evaluations)
    n_refine_evals = 0

    if refine and len(lam_list_sorted) >= 2:
        ranked = sorted(evaluations.items(), key=lambda kv: kv[1][1], reverse=True)
        candidates = [lam for lam, _ in ranked[: max(1, refine_top_k)]]
        r_starts = refine_n_starts if refine_n_starts is not None else max(2, em_n_starts // 3)

        for ci, cand in enumerate(candidates):
            pos = lam_list_sorted.index(cand)
            bracket_lo = lam_list_sorted[pos - 1] if pos > 0 else cand / zoom_shrink
            bracket_hi = lam_list_sorted[pos + 1] if pos < len(lam_list_sorted) - 1 else cand * zoom_shrink
            bracket_lo = max(bracket_lo, 1e-6)
            if bracket_hi <= bracket_lo:
                continue
            warm_pi = evaluations[cand][0]

            for level in range(max(1, n_zoom_levels)):
                if bracket_hi <= bracket_lo:
                    break
                local_grid = np.geomspace(bracket_lo, bracket_hi, n_zoom_points)
                prev_local_pi = warm_pi
                for j, lam_val in enumerate(local_grid):
                    seed = em_random_seed + 60_000 + ci * 5_000 + level * 500 + j
                    already = lam_val in evaluations
                    _eval_and_store(lam_val, r_starts, seed, [prev_local_pi])
                    if not already:
                        n_refine_evals += 1
                    prev_local_pi = evaluations[lam_val][0]
                level_best_idx = int(np.argmax([evaluations[float(l)][1] for l in local_grid]))
                level_best_lam = float(local_grid[level_best_idx])
                warm_pi = evaluations[level_best_lam][0]
                bracket_lo = float(local_grid[level_best_idx - 1]) if level_best_idx > 0 else level_best_lam / zoom_shrink
                bracket_hi = (
                    float(local_grid[level_best_idx + 1])
                    if level_best_idx < len(local_grid) - 1
                    else level_best_lam * zoom_shrink
                )
                bracket_lo = max(bracket_lo, 1e-6)

            # Final short Brent polish inside the now-narrow last bracket, where
            # a second competing mode at this scale is implausible.
            if bracket_hi > bracket_lo:
                xatol = max(1e-3, (bracket_hi - bracket_lo) * 1e-2)

                def objective(lam_val: float, _warm: np.ndarray = warm_pi, _ci: int = ci) -> float:
                    nonlocal n_refine_evals
                    lam_val = float(lam_val)
                    already = lam_val in evaluations
                    ll = _eval_and_store(lam_val, r_starts, em_random_seed + 70_000 + _ci * 1000, [_warm])
                    if not already:
                        n_refine_evals += 1
                    return -ll

                minimize_scalar(objective, bounds=(bracket_lo, bracket_hi), method="bounded", options={"xatol": xatol})

        # Re-fit the best point found so far (coarse or refined) with the full
        # multistart budget as a final safety net.
        best_lam_so_far = max(evaluations, key=lambda l: evaluations[l][1])
        warm_pi = evaluations[best_lam_so_far][0]
        pi_f, ll_f, n_iter_f = _fit_at(best_lam_so_far, em_n_starts, em_random_seed + 99_000, [warm_pi])
        if ll_f >= evaluations[best_lam_so_far][1]:
            evaluations[best_lam_so_far] = (pi_f, ll_f, n_iter_f)

    best_lam = max(evaluations, key=lambda l: evaluations[l][1])
    best_pi, best_ll, best_n_iter = evaluations[best_lam]
    converged = best_n_iter < em_max_iter
    diagnostics = {
        "n_coarse_evals": n_coarse_evals,
        "n_refine_evals": n_refine_evals,
        "n_total_evals": len(evaluations),
        "evaluations": {lam: ll for lam, (_, ll, _) in evaluations.items()},
    }
    return float(best_lam), best_pi, float(best_ll), int(best_n_iter), bool(converged), diagnostics


def profile_fit_lambda(
    grouped: pd.DataFrame,
    mu: MuTriple,
    lambda_grid: Iterable[float],
    exp_cache: ExponentialComponentCache,
    em_max_iter: int = 1000,
    em_tol: float = 1e-8,
    active_components: Sequence[str] = COMPONENTS,
    em_n_starts: int = 16,
    em_random_seed: int = 1,
    rho: float = 0.0,
    refine: bool = True,
    refine_top_k: int = 2,
) -> tuple[float, np.ndarray, float, int, bool]:
    """Profile-likelihood lambda estimate: coarse grid, then local continuous refinement.

    `refine=False` reproduces the original pure-grid-search behavior exactly
    (useful for regression tests and for reproducing older results bit-for-bit).
    See `_profile_lambda_search` for the full rationale and
    `profile_fit_lambda_with_diagnostics` for a version that also returns the
    evaluated (lambda -> loglik) surface, used by the validation tests.
    """
    best_lam, best_pi, best_ll, best_n_iter, converged, _diag = _profile_lambda_search(
        grouped,
        mu,
        lambda_grid,
        exp_cache,
        em_max_iter=em_max_iter,
        em_tol=em_tol,
        active_components=active_components,
        em_n_starts=em_n_starts,
        em_random_seed=em_random_seed,
        rho=rho,
        refine=refine,
        refine_top_k=refine_top_k,
    )
    return best_lam, best_pi, best_ll, best_n_iter, converged


def profile_fit_lambda_with_diagnostics(
    grouped: pd.DataFrame,
    mu: MuTriple,
    lambda_grid: Iterable[float],
    exp_cache: ExponentialComponentCache,
    em_max_iter: int = 1000,
    em_tol: float = 1e-8,
    active_components: Sequence[str] = COMPONENTS,
    em_n_starts: int = 16,
    em_random_seed: int = 1,
    rho: float = 0.0,
    refine: bool = True,
    refine_top_k: int = 2,
) -> tuple[float, np.ndarray, float, int, bool, dict]:
    """Same as `profile_fit_lambda` but also returns a diagnostics dict with the
    evaluated lambda -> loglik surface and evaluation counts. Used for validating
    the search against fine-grid reference points without re-plumbing the
    5-tuple return type used throughout the rest of the pipeline."""
    return _profile_lambda_search(
        grouped,
        mu,
        lambda_grid,
        exp_cache,
        em_max_iter=em_max_iter,
        em_tol=em_tol,
        active_components=active_components,
        em_n_starts=em_n_starts,
        em_random_seed=em_random_seed,
        rho=rho,
        refine=refine,
        refine_top_k=refine_top_k,
    )


def group_positive_rows(df: pd.DataFrame, by_sample: bool = False) -> pd.DataFrame:
    """Identical positive observations collapsed into weighted rows; by_sample (0.4.1, organ-specific composition)
    keeps the organ in the key."""
    pos = df[df["alt_depth"] > 0].copy()
    keys = ["alt_depth", "DP"] + (["sample"] if by_sample and "sample" in pos.columns else [])
    if pos.empty:
        return pd.DataFrame(columns=keys + ["n_rows"])
    return (
        pos.groupby(keys, as_index=False)
        .size()
        .rename(columns={"size": "n_rows"})
        .sort_values(["DP", "alt_depth"])
        .reset_index(drop=True)
    )


def fit_one_depth(
    df_depth: pd.DataFrame,
    mu: MuTriple,
    lambda_grid: Iterable[float],
    exp_cache: ExponentialComponentCache,
    em_max_iter: int = 1000,
    em_tol: float = 1e-8,
    active_components: Sequence[str] = COMPONENTS,
    em_n_starts: int = 16,
    em_random_seed: int = 1,
    rho: float = 0.0,
    lambda_refine: bool = True,
) -> FitResult:
    pos = df_depth[df_depth["alt_depth"] > 0].copy()
    grouped = group_positive_rows(df_depth, by_sample=mu.organ_scale is not None)
    if grouped.empty:
        raise ValueError("No positive observations for this tree/depth; cannot fit model.")
    lam, pi, ll, n_iter, converged = profile_fit_lambda(
        grouped,
        mu,
        lambda_grid,
        exp_cache,
        em_max_iter=em_max_iter,
        em_tol=em_tol,
        active_components=active_components,
        em_n_starts=em_n_starts,
        em_random_seed=em_random_seed,
        rho=rho,
        refine=lambda_refine,
    )
    pi = np.asarray(pi, dtype=float)
    pi_err = float(pi[len(COMPONENTS)]) if len(pi) > len(COMPONENTS) else 0.0
    pi4 = pi[: len(COMPONENTS)].copy()
    pi4[0] += pi_err
    return FitResult(
        mu=mu,
        lambda_hat=lam,
        pi=pi4,
        pi_E_error=pi_err,
        loglik=ll,
        n_positive_rows=int(len(pos)),
        n_unique_positive_variants=int(pos["variant_id"].nunique()),
        grouped=grouped,
        active_components=tuple(active_components),
        rho=float(rho),
        n_iter=n_iter,
        converged=converged,
    )


def classify_variants_for_fit(df_depth: pd.DataFrame, fit: FitResult, exp_cache: ExponentialComponentCache) -> pd.DataFrame:
    pos = df_depth[df_depth["alt_depth"] > 0].copy()
    if pos.empty:
        cols = ["variant_id", "chrom", "pos", "ref", "alt", "P_E", "P_L1", "P_L2", "P_L3", "P_fixed"]
        return pd.DataFrame(columns=cols)

    y = pos["alt_depth"].to_numpy(dtype=int)
    d = pos["DP"].to_numpy(dtype=int)
    log_comp = fit_component_loglik(y, d, fit, exp_cache, samples=_row_samples(pos, fit))
    comp_cols = [f"loglik_{c}" for c in COMPONENTS]
    tmp = pos[["variant_id", "chrom", "pos", "ref", "alt"]].copy()
    for j, c in enumerate(comp_cols):
        tmp[c] = log_comp[:, j]

    rows = []
    active_mask = active_mask_from_components(fit.active_components)
    log_pi = np.full(len(COMPONENTS), -np.inf, dtype=float)
    log_pi[active_mask] = np.log(np.clip(fit.pi[active_mask], 1e-300, None))
    for vid, g in tmp.groupby("variant_id", sort=False):
        summed = np.array([g[f"loglik_{c}"].sum() for c in COMPONENTS], dtype=float)
        log_post = log_pi + summed
        log_post -= logsumexp(log_post)
        post = np.exp(log_post)
        post[~active_mask] = 0.0
        post = post / post.sum()
        first = g.iloc[0]
        row = {
            "variant_id": vid,
            "chrom": first["chrom"],
            "pos": first["pos"],
            "ref": first["ref"],
            "alt": first["alt"],
            "n_positive_observations": int(len(g)),
            "P_E": post[0],
            "P_L1": post[1],
            "P_L2": post[2],
            "P_L3": post[3],
        }
        row["P_fixed"] = row["P_L1"] + row["P_L2"] + row["P_L3"]
        comp = COMPONENTS[int(np.argmax(post))]
        row["variant_inferred_component_MAP"] = comp
        row["variant_inferred_fixed_MAP"] = comp in LAYER_COMPONENTS
        row["variant_inferred_fixed_highconf_95"] = row["P_fixed"] >= 0.95
        row["variant_inferred_layer_MAP"] = comp if comp in LAYER_COMPONENTS else ""
        rows.append(row)
    return pd.DataFrame(rows)


def softmax_grid_weights(logliks: Sequence[float]) -> np.ndarray:
    """Normalize a set of grid-point (mu, rho) total log-likelihoods into model-average weights.

    Equivalent to unnormalized-AIC weighting when every grid point has the same
    number of free parameters (true here: only pi/lambda are fit, mu and rho are
    fixed per grid point) -- softmax(loglik) then sums to 1 and reduces to a
    plain average when all grid points fit equally well.
    """
    ll = np.asarray(list(logliks), dtype=float)
    if len(ll) == 0:
        return ll
    ll = ll - np.nanmax(ll)
    w = np.exp(ll)
    w = np.where(np.isfinite(w), w, 0.0)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(len(ll), 1.0 / len(ll))
    return w / total


def _weighted_average_calls(
    calls: list[pd.DataFrame],
    weights: np.ndarray | None,
    id_cols: list[str],
    meta_cols: list[str],
    post_cols: list[str],
) -> pd.DataFrame:
    if not calls:
        return pd.DataFrame()
    non_empty = [(c, w) for c, w in zip(calls, weights if weights is not None else [None] * len(calls)) if not c.empty]
    if not non_empty:
        return pd.DataFrame()
    tagged = []
    for i, (c, _) in enumerate(non_empty):
        cc = c.copy()
        cc["_grid_idx"] = i
        tagged.append(cc)
    all_calls = pd.concat(tagged, ignore_index=True)

    if weights is None:
        all_calls["_w"] = 1.0
    else:
        w_arr = np.asarray([w for _, w in non_empty], dtype=float)
        w_map = {i: w_arr[i] for i in range(len(w_arr))}
        all_calls["_w"] = all_calls["_grid_idx"].map(w_map)

    for c in post_cols:
        all_calls[f"_wx_{c}"] = all_calls[c] * all_calls["_w"]

    grp = all_calls.groupby(id_cols, sort=False)
    wsum = grp["_w"].transform("sum")
    agg_rows = []
    for keys, g in grp:
        if not isinstance(keys, tuple):
            keys = (keys,)
        wtot = float(g["_w"].sum())
        row = dict(zip(id_cols, keys))
        if wtot <= 0:
            for c in post_cols:
                row[c] = float(g[c].mean())
        else:
            for c in post_cols:
                row[c] = float((g[c] * g["_w"]).sum() / wtot)
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)

    meta = all_calls[id_cols + meta_cols].drop_duplicates(id_cols)
    out = meta.merge(agg, on=id_cols, how="inner")
    if "n_positive_observations" in all_calls.columns:
        obs = all_calls.groupby(id_cols, as_index=False)["n_positive_observations"].max()
        out = out.merge(obs, on=id_cols, how="left")
    return out


def average_grid_variant_calls(calls: list[pd.DataFrame], weights: np.ndarray | None = None) -> pd.DataFrame:
    """Combine variant-level classifications across a (mu, rho) grid.

    weights=None reduces to a plain (unweighted) mean, matching earlier behavior.
    Passing per-grid-point total log-likelihoods via `softmax_grid_weights` instead
    turns this into an (approximate) likelihood-weighted model average, so a
    poorly-fitting mu/rho combination no longer counts as much as the best one.
    """
    out = _weighted_average_calls(
        calls,
        weights,
        id_cols=["variant_id"],
        meta_cols=["chrom", "pos", "ref", "alt"],
        post_cols=["P_E", "P_L1", "P_L2", "P_L3", "P_fixed"],
    )
    if out.empty:
        return out
    comps = out[["P_E", "P_L1", "P_L2", "P_L3"]].to_numpy()
    idx = np.argmax(comps, axis=1)
    labels = np.asarray(COMPONENTS, dtype=object)[idx]
    out["variant_inferred_component_MAP"] = labels
    out["variant_inferred_fixed_MAP"] = np.isin(labels, list(LAYER_COMPONENTS))
    out["variant_inferred_fixed_highconf_95"] = out["P_fixed"] >= 0.95
    out["variant_inferred_layer_MAP"] = np.where(out["variant_inferred_fixed_MAP"], labels, "")
    return out


def classify_organ_observations_for_fit(
    df_depth: pd.DataFrame, fit: FitResult, exp_cache: ExponentialComponentCache
) -> pd.DataFrame:
    """Per-(variant, organ) posterior classification, one row per positive observation.

    Unlike `classify_variants_for_fit`, this does NOT pool evidence across the
    organs a variant appears in -- each row's posterior uses only that organ's
    own (y, d) read counts (single-observation posterior P_js,fixed). This is the
    single-observation classification behind the per-organ counts (16, 17), so
    that the counts of different organs stay (as close as the data allow to)
    independent.
    """
    pos = df_depth[df_depth["alt_depth"] > 0].copy()
    cols = [
        "variant_id", "chrom", "pos", "ref", "alt", "sample",
        "P_E", "P_L1", "P_L2", "P_L3", "P_fixed",
        "obs_inferred_component_MAP", "obs_inferred_fixed_MAP", "obs_inferred_fixed_highconf_95",
    ]
    if pos.empty:
        return pd.DataFrame(columns=cols)

    y = pos["alt_depth"].to_numpy(dtype=int)
    d = pos["DP"].to_numpy(dtype=int)
    log_comp = fit_component_loglik(y, d, fit, exp_cache, samples=_row_samples(pos, fit))
    active_mask = active_mask_from_components(fit.active_components)
    log_pi = np.full(len(COMPONENTS), -np.inf, dtype=float)
    log_pi[active_mask] = np.log(np.clip(fit.pi[active_mask], 1e-300, None))

    log_post = log_pi[None, :] + log_comp
    log_post -= logsumexp(log_post, axis=1, keepdims=True)
    post = np.exp(log_post)
    post[:, ~active_mask] = 0.0
    row_sums = post.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    post = post / row_sums

    out = pos[["variant_id", "chrom", "pos", "ref", "alt", "sample"]].copy()
    out["P_E"] = post[:, 0]
    out["P_L1"] = post[:, 1]
    out["P_L2"] = post[:, 2]
    out["P_L3"] = post[:, 3]
    out["P_fixed"] = out["P_L1"] + out["P_L2"] + out["P_L3"]
    idx = np.argmax(post, axis=1)
    labels = np.asarray(COMPONENTS, dtype=object)[idx]
    out["obs_inferred_component_MAP"] = labels
    out["obs_inferred_fixed_MAP"] = np.isin(labels, list(LAYER_COMPONENTS))
    out["obs_inferred_fixed_highconf_95"] = out["P_fixed"] >= 0.95
    return out


def average_grid_organ_calls(calls: list[pd.DataFrame], weights: np.ndarray | None = None) -> pd.DataFrame:
    out = _weighted_average_calls(
        calls,
        weights,
        id_cols=["variant_id", "sample"],
        meta_cols=["chrom", "pos", "ref", "alt"],
        post_cols=["P_E", "P_L1", "P_L2", "P_L3", "P_fixed"],
    )
    if out.empty:
        return out
    comps = out[["P_E", "P_L1", "P_L2", "P_L3"]].to_numpy()
    idx = np.argmax(comps, axis=1)
    labels = np.asarray(COMPONENTS, dtype=object)[idx]
    out["obs_inferred_component_MAP"] = labels
    out["obs_inferred_fixed_MAP"] = np.isin(labels, list(LAYER_COMPONENTS))
    out["obs_inferred_fixed_highconf_95"] = out["P_fixed"] >= 0.95
    return out


def organ_fixed_counts(organ_calls: pd.DataFrame, samples: Sequence[str]) -> pd.DataFrame:
    """Fixed-mutation counts per organ (or replicate group) at one depth tier, from the single-observation
    posteriors: MAP counts (all fixed and per layer), high-confidence counts (P_fixed >= 0.95) and posterior-weighted
    counts (sums of P_fixed and of P_L1, P_L2, P_L3).

    Every unit in `samples` gets a row even if it has no positive fixed observation at this tier: that is a real
    zero, not a missing value.
    """
    rows = []
    by_sample = {s: g for s, g in organ_calls.groupby("sample", sort=False)} if not organ_calls.empty else {}
    for s in samples:
        g = by_sample.get(s)
        if g is None or g.empty:
            rows.append({"sample": s, **{c: 0 for c in INT_COUNT_COLUMNS}, **{c: 0.0 for c in FLOAT_COUNT_COLUMNS}})
            continue
        labels = g["obs_inferred_component_MAP"].astype(str)
        rows.append(
            {
                "sample": s,
                "MAP_fixed_count": int(g["obs_inferred_fixed_MAP"].sum()),
                "MAP_L1_count": int((labels == "L1").sum()),
                "MAP_L2_count": int((labels == "L2").sum()),
                "MAP_L3_count": int((labels == "L3").sum()),
                "highconf95_fixed_count": int(g["obs_inferred_fixed_highconf_95"].sum()),
                "posterior_weighted_fixed_count": float(g["P_fixed"].sum()),
                "posterior_weighted_L1_count": float(g["P_L1"].sum()),
                "posterior_weighted_L2_count": float(g["P_L2"].sum()),
                "posterior_weighted_L3_count": float(g["P_L3"].sum()),
            }
        )
    return pd.DataFrame(rows)


INT_COUNT_COLUMNS = ("MAP_fixed_count", "MAP_L1_count", "MAP_L2_count", "MAP_L3_count", "highconf95_fixed_count")
FLOAT_COUNT_COLUMNS = ("posterior_weighted_fixed_count", "posterior_weighted_L1_count",
                       "posterior_weighted_L2_count", "posterior_weighted_L3_count")
COUNT_COLUMNS = INT_COUNT_COLUMNS + FLOAT_COUNT_COLUMNS


def counts_from_variant_calls(calls: pd.DataFrame) -> dict[str, float]:
    if calls.empty:
        return {
            "MAP_fixed_count": 0,
            "MAP_L1_count": 0,
            "MAP_L2_count": 0,
            "MAP_L3_count": 0,
            "highconf95_fixed_count": 0,
            "posterior_weighted_fixed_count": 0.0,
            "posterior_weighted_L1_count": 0.0,
            "posterior_weighted_L2_count": 0.0,
            "posterior_weighted_L3_count": 0.0,
        }
    layer = calls["variant_inferred_layer_MAP"].fillna("").astype(str)
    return {
        "MAP_fixed_count": int(calls["variant_inferred_fixed_MAP"].sum()),
        "MAP_L1_count": int((layer == "L1").sum()),
        "MAP_L2_count": int((layer == "L2").sum()),
        "MAP_L3_count": int((layer == "L3").sum()),
        "highconf95_fixed_count": int(calls["variant_inferred_fixed_highconf_95"].sum()),
        "posterior_weighted_fixed_count": float(calls["P_fixed"].sum()),
        "posterior_weighted_L1_count": float(calls["P_L1"].sum()),
        "posterior_weighted_L2_count": float(calls["P_L2"].sum()),
        "posterior_weighted_L3_count": float(calls["P_L3"].sum()),
    }
