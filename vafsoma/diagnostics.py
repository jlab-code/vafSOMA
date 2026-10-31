"""Post-fit diagnostics (vafSOMA 0.1.6; group-aware resolvability 0.1.7). Output-only: these functions read the fitted parameters and the input
observations; they never change fits, posteriors or calls.

1. Layer resolvability (14_layer_resolvability.csv)
   For each tree, depth tier and active layer k, the model-implied recall: the probability that an observation of a
   variant truly fixed in layer k (alt reads ~ Binomial or beta-binomial(d, mu_k), conditional on >= 1 alt read) has
   maximum-a-posteriori class k under the fitted mixture (pi, lambda, mu, rho), averaged over the empirical depth
   distribution of that tier's observations. A layer whose recall averaged over the selected depth window falls below
   `min_recall` (default 0.5) is reported as unresolvable: most of its fixed variants would be absorbed by the
   background or by another layer, so its count should not be interpreted. Typical cases: a layer whose expected
   VAF lies inside the fitted background (low contribution c_k), or two layers with near-identical contributions.
   0.4.2: a layer is also reported as unresolvable when its expected VAF mu_k lies below the `background_quantile`
   (default 0.95) quantile of the fitted background allele-fraction distribution (exponential with rate lambda,
   truncated at q_max), averaged over the selected depth window. Recall alone can miss this case: with a light
   background and a small layer contribution, the layer component can take over most of the background (large pi_k,
   high recall of its own component) and the layer count then counts background. Set background_quantile to None to
   use recall only (vafSOMA <= 0.4.1).
   0.1.7: for trees analysed with replicate groups, calls are made per group, so the recall is evaluated per group:
   replicate vectors are simulated at mu_k with the depth vectors of the observed units (conditional on >= 1 positive
   replicate) and classified by the fitted group mixture (shared and private background, layers).

2. Background / mixture fit (15_vaf_fit_check.csv)
   Posterior-predictive check of the fitted mixture: observed vs expected numbers of positive observations in VAF
   bins, expected = sum over observations i of P(alt_i / d_i in bin | d_i, alt_i >= 1) under the fitted mixture.
   Reported per bin with Pearson residuals, plus a per-tree/tier summary. Because the background is modelled as a
   (truncated) exponential, a background that is heavier-tailed than exponential shows up as excess observed counts
   at low-intermediate VAF; this misfit can leak into the lowest-VAF layer (often L1), whose count should then be
   treated with caution. The summary flag `fit_warning` is raised when any bin with expected count >= `min_expected`
   has |Pearson residual| >= `max_abs_residual` (default 4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import COMPONENTS, ExponentialComponentCache, FitResult, fit_component_loglik

DEFAULT_VAF_BINS = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.0)


def _depth_support(depths: np.ndarray, max_points: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Empirical depth distribution, compressed to <= max_points support points (quantile bins)."""
    depths = np.asarray(depths, dtype=int)
    depths = depths[depths > 0]
    if depths.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    vals, cnt = np.unique(depths, return_counts=True)
    if len(vals) <= max_points:
        return vals, cnt / cnt.sum()
    qs = np.quantile(depths, (np.arange(max_points) + 0.5) / max_points)
    pts = np.unique(np.round(qs).astype(int))
    idx = np.abs(depths[:, None] - pts[None, :]).argmin(axis=1)
    w = np.bincount(idx, minlength=len(pts)).astype(float)
    return pts, w / w.sum()


def _posterior_by_y(fit: FitResult, d: int, exp_cache: ExponentialComponentCache) -> tuple[np.ndarray, np.ndarray]:
    """For one depth d: component log-likelihoods (conditional on y >= 1) for y = 1..d, and the log prior weights."""
    y = np.arange(1, d + 1)
    dd = np.full_like(y, d)
    log_comp = fit_component_loglik(y, dd, fit, exp_cache)
    with np.errstate(divide="ignore"):
        log_pi = np.log(np.clip(np.asarray(fit.pi, dtype=float), 0.0, None))
    return log_comp, log_pi


def layer_recall(fit: FitResult, depths: np.ndarray, exp_cache: ExponentialComponentCache,
                 max_points: int = 60) -> dict[str, float]:
    """Model-implied recall of each active layer, averaged over the empirical depth distribution."""
    pts, w = _depth_support(depths, max_points)
    out = {L: np.nan for L in COMPONENTS[1:]}
    if len(pts) == 0:
        return out
    acc = {L: 0.0 for L in COMPONENTS[1:] if L in fit.active_components}
    for d, wd in zip(pts, w):
        log_comp, log_pi = _posterior_by_y(fit, int(d), exp_cache)
        joint = log_comp + log_pi[None, :]
        joint[~np.isfinite(joint)] = -np.inf
        map_cls = np.argmax(joint, axis=1)
        for j, L in enumerate(COMPONENTS[1:], start=1):
            if L not in acc:
                continue
            p_y = np.exp(log_comp[:, j])          # P(y | d, layer L, y >= 1)
            acc[L] += wd * float(p_y[map_cls == j].sum())
    for L, v in acc.items():
        out[L] = v
    return out


def simulate_reads(rng: np.random.Generator, D: np.ndarray, mu: float, rho: float = 0.0,
                   read_error: float = 0.0) -> np.ndarray:
    """Alt reads of a fixed variant under the observation model of the fit (0.4.0): allele fraction
    ~ Beta(mean mu, rho) if rho > 0, read probability v (1 - e) + (1 - v) e. With rho = e = 0: Binomial(D, mu)
    (identical draws to 0.3.0)."""
    if (not rho or rho <= 0) and read_error <= 0:
        return rng.binomial(D, mu)
    v = np.broadcast_to(np.asarray(mu, dtype=float), D.shape).copy()   # 0.4.1: mu may vary by entry
    if rho and rho > 0:
        s = (1.0 - rho) / rho
        v = rng.beta(mu * s, (1.0 - mu) * s, size=D.shape)
    return rng.binomial(D, np.clip(v * (1.0 - read_error) + (1.0 - v) * read_error, 0.0, 1.0))


def group_layer_recall(row, tv: pd.DataFrame, smap: dict, exp_cache: ExponentialComponentCache, fit: FitResult,
                       max_units: int = 3000, seed: int = 1) -> dict[str, float]:
    """Model-implied recall per layer for the replicate-group model (see module docstring)."""
    from .replicates import build_units, ReplicateLikelihood, _posteriors, _active_mask5
    out = {L: np.nan for L in COMPONENTS[1:]}
    df = tv.copy()
    if "variant_id" not in df.columns:
        df["variant_id"] = df["chrom"].astype(str) + ":" + df["pos"].astype(str) + ":" + df["ref"].astype(str) + ":" + df["alt"].astype(str)
    df = df[df["sample"].astype(str).isin(smap)]
    units = build_units(df, {str(k): v for k, v in smap.items()})
    if not units.get("n"):
        return out
    rng = np.random.default_rng(seed)
    idx = np.arange(units["n"])
    if len(idx) > max_units:
        idx = rng.choice(idx, max_units, replace=False)
    D, M = units["D"][idx], units["M"][idx]
    S = units["S"][idx] if units.get("S") is not None else None
    organ = fit.mu.organ_scale is not None and S is not None
    pe = getattr(row, "pi_E_error", 0.0)
    pe = 0.0 if pe is None or (isinstance(pe, float) and np.isnan(pe)) else float(pe)
    pi5 = np.array([float(row.pi_E_shared), float(row.pi_E_private), float(row.pi_L1), float(row.pi_L2), float(row.pi_L3), pe])
    act5 = _active_mask5(fit.active_components, getattr(exp_cache, "error_component", False))
    for j, L in enumerate(COMPONENTS[1:], start=2):
        if L not in fit.active_components:
            continue
        mu_L = fit.mu.values(L, S) if organ else fit.mu.value(L)
        Y = simulate_reads(rng, D, mu_L, fit.rho, exp_cache.read_error) * M
        keep = ((Y > 0) & M).any(1)
        if not keep.any():
            out[L] = 0.0
            continue
        sim = dict(n=int(keep.sum()), Y=Y[keep], D=D[keep], M=M[keep], S=(S[keep] if S is not None else None))
        lik = ReplicateLikelihood(sim, exp_cache, by_sample=organ)
        post = _posteriors(lik.comp_loglik_units(fit.lambda_hat, fit.mu, fit.active_components, fit.rho), pi5, act5)
        p4 = np.column_stack([post[:, 0] + post[:, 1] + post[:, 5], post[:, 2], post[:, 3], post[:, 4]])
        out[L] = float((np.argmax(p4, axis=1) == j - 1).mean())
    return out


def vaf_fit_check(fit: FitResult, obs: pd.DataFrame, exp_cache: ExponentialComponentCache,
                  bins: tuple[float, ...] = DEFAULT_VAF_BINS) -> pd.DataFrame:
    """Observed vs expected positive observations per VAF bin under the fitted mixture.
    obs: rows with alt_depth, DP (positive rows are used)."""
    pos = obs[(obs["alt_depth"] > 0) & (obs["DP"] > 0)]
    edges = np.asarray(bins, dtype=float)
    nb = len(edges) - 1
    observed = np.histogram(np.clip(pos["alt_depth"] / pos["DP"], 0, 1), bins=edges)[0].astype(float)
    expected = np.zeros(nb)
    for d, n_d in pos.groupby("DP").size().items():
        log_comp, log_pi = _posterior_by_y(fit, int(d), exp_cache)
        joint = log_comp + log_pi[None, :]
        joint[~np.isfinite(joint)] = -np.inf
        m = joint.max(axis=1, keepdims=True)
        p_y = np.exp(m[:, 0]) * np.exp(joint - m).sum(axis=1)      # predictive P(y | d, y >= 1)
        p_y = p_y / p_y.sum()
        vaf = np.arange(1, int(d) + 1) / int(d)
        b = np.clip(np.searchsorted(edges, vaf, side="right") - 1, 0, nb - 1)
        expected += n_d * np.bincount(b, weights=p_y, minlength=nb)
    resid = np.where(expected > 0, (observed - expected) / np.sqrt(np.maximum(expected, 1e-12)), np.nan)
    return pd.DataFrame({"vaf_bin_lo": edges[:-1], "vaf_bin_hi": edges[1:], "observed": observed,
                         "expected": expected, "pearson_residual": resid})


def _organ_fits(fit: FitResult, obs: pd.DataFrame):
    """(organ, fit with that organ's expected VAFs, its rows) for an organ-specific composition (0.4.1)."""
    import dataclasses
    for o, g in obs.groupby(obs["sample"].astype(str), sort=True):
        yield o, dataclasses.replace(fit, mu=fit.mu.for_sample(o)), g


def layer_recall_by_organ(fit: FitResult, obs: pd.DataFrame, exp_cache: ExponentialComponentCache) -> dict[str, float]:
    """Model-implied recall with organ-specific compositions: the recall of each organ (its expected VAFs, its depths),
    averaged with weights proportional to the organ's number of observations."""
    acc, wsum = {}, 0.0
    for _, f_o, g in _organ_fits(fit, obs):
        w = float(len(g)); rec = layer_recall(f_o, g["DP"].to_numpy(), exp_cache)
        for L, v in rec.items():
            if np.isfinite(v):
                acc[L] = acc.get(L, 0.0) + w * v
        wsum += w
    out = {L: np.nan for L in COMPONENTS[1:]}
    for L, v in acc.items():
        out[L] = v / wsum if wsum > 0 else np.nan
    return out


def vaf_fit_check_by_organ(fit: FitResult, obs: pd.DataFrame, exp_cache: ExponentialComponentCache,
                           bins: tuple[float, ...] = DEFAULT_VAF_BINS) -> pd.DataFrame:
    """VAF fit check with organ-specific compositions: observed and expected counts summed over organs."""
    parts = [vaf_fit_check(f_o, g, exp_cache, bins) for _, f_o, g in _organ_fits(fit, obs)]
    if not parts:
        return vaf_fit_check(fit, obs, exp_cache, bins)
    out = parts[0][["vaf_bin_lo", "vaf_bin_hi"]].copy()
    out["observed"] = sum(p["observed"].to_numpy() for p in parts)
    out["expected"] = sum(p["expected"].to_numpy() for p in parts)
    e = out["expected"].to_numpy()
    out["pearson_residual"] = np.where(e > 0, (out["observed"].to_numpy() - e) / np.sqrt(np.maximum(e, 1e-12)), np.nan)
    return out


def background_vaf_quantile(lam: float, q_max: float, p: float) -> float:
    """0.4.2: p-quantile of the fitted background allele fraction, an exponential with rate `lam` truncated to
    (0, q_max): q_p = -log(1 - p (1 - exp(-lam q_max))) / lam."""
    lam = float(lam)
    if not np.isfinite(lam) or lam <= 0:
        return float(p * q_max)
    return float(-np.log1p(-p * (-np.expm1(-lam * q_max))) / lam)


def build_diagnostics(fit_df: pd.DataFrame, variants: pd.DataFrame, selected: pd.DataFrame, q_max: float, n_quad: int,
                      fit_from_row, read_error: float = 0.0, min_recall: float = 0.5, min_expected: float = 10.0,
                      max_abs_residual: float = 4.0, group_maps: dict | None = None,
                      background_quantile: float | None = 0.95) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (resolvability by tree x depth x layer, resolvability summary by tree x layer, VAF fit check).
    0.4.2: `background_quantile` (None: recall only) adds the background-overlap condition (module docstring)."""
    exp_cache = ExponentialComponentCache(q_max=q_max, n_quad=n_quad, read_error=read_error)
    rec_rows, fit_rows = [], []
    win = {str(s.tree): (int(s.selected_window_start), int(s.selected_window_end)) for s in selected.itertuples(index=False)}
    for r in fit_df.itertuples(index=False):
        tree, depth = str(r.tree), int(r.depth_filter)
        if tree not in win:
            continue
        a, b = win[tree]
        if not (a <= depth <= b):
            continue
        fit = fit_from_row(r)
        tv = variants[(variants["tree"] == tree) & (variants["depth_filter"] == depth)]
        if tv.empty:
            continue
        smap = (group_maps or {}).get(tree)
        if smap and "pi_E_shared" in fit_df.columns and pd.notna(getattr(r, "pi_E_shared", np.nan)):
            rec = group_layer_recall(r, tv, smap, exp_cache, fit)
        elif fit.mu.organ_scale is not None:
            rec = layer_recall_by_organ(fit, tv, exp_cache)
        else:
            rec = layer_recall(fit, tv["DP"].to_numpy(), exp_cache)
        for L in COMPONENTS[1:]:
            if L in fit.active_components:
                rec_rows.append(dict(tree=tree, depth_filter=depth, layer=L, mu=fit.mu.value(L),
                                     pi=float(fit.pi[COMPONENTS.index(L)]), model_implied_recall=rec[L],
                                     background_quantile_vaf=(background_vaf_quantile(fit.lambda_hat, q_max, background_quantile)
                                                              if background_quantile is not None else np.nan)))
        chk = vaf_fit_check_by_organ(fit, tv, exp_cache) if fit.mu.organ_scale is not None else vaf_fit_check(fit, tv, exp_cache)
        chk.insert(0, "depth_filter", depth); chk.insert(0, "tree", tree)
        fit_rows.append(chk)
    rec_df = pd.DataFrame(rec_rows)
    if rec_df.empty:
        summ = pd.DataFrame(columns=["tree", "layer", "mean_model_implied_recall", "min_recall", "background_quantile",
                                     "mean_background_quantile_vaf", "above_background", "resolvable"])
    else:
        summ = (rec_df.groupby(["tree", "layer"], as_index=False)
                .agg(mu=("mu", "first"), mean_pi=("pi", "mean"), mean_model_implied_recall=("model_implied_recall", "mean"),
                     mean_background_quantile_vaf=("background_quantile_vaf", "mean")))
        summ["min_recall"] = min_recall
        summ["background_quantile"] = background_quantile if background_quantile is not None else np.nan
        if background_quantile is None:
            summ["above_background"] = pd.array([pd.NA] * len(summ), dtype="boolean")
            summ["resolvable"] = summ["mean_model_implied_recall"] >= min_recall
        else:
            summ["above_background"] = summ["mu"].astype(float) > summ["mean_background_quantile_vaf"]
            summ["resolvable"] = (summ["mean_model_implied_recall"] >= min_recall) & summ["above_background"]
        summ = summ[["tree", "layer", "mu", "mean_pi", "mean_model_implied_recall", "min_recall", "background_quantile",
                     "mean_background_quantile_vaf", "above_background", "resolvable"]]
    chk_df = pd.concat(fit_rows, ignore_index=True) if fit_rows else pd.DataFrame()
    if not chk_df.empty:
        big = chk_df["expected"] >= min_expected
        chk_df["flag_bin"] = big & (chk_df["pearson_residual"].abs() >= max_abs_residual)
    return rec_df, summ, chk_df


def summarize_fit_check(chk_df: pd.DataFrame, min_expected: float = 10.0) -> pd.DataFrame:
    """Per tree: number of depth tiers with any flagged bin, and the worst residual (bins with expected >= min_expected)
    and its VAF bin. fit_warning = flagged bins in at least half of the selected depth tiers."""
    if chk_df.empty:
        return pd.DataFrame(columns=["tree", "n_tiers", "n_tiers_flagged", "max_abs_residual", "worst_bin", "fit_warning"])
    rows = []
    for tree, g_all in chk_df.groupby("tree"):
        per_tier = g_all.groupby("depth_filter").flag_bin.any()
        g = g_all[g_all["expected"] >= min_expected]          # worst bin among bins large enough to judge
        if g.empty:
            g = g_all
        i = g["pearson_residual"].abs().idxmax()
        rows.append(dict(tree=tree, n_tiers=int(per_tier.size), n_tiers_flagged=int(per_tier.sum()),
                         max_abs_residual=float(abs(g.loc[i, "pearson_residual"])),
                         worst_bin=f"[{g.loc[i, 'vaf_bin_lo']:.3f}, {g.loc[i, 'vaf_bin_hi']:.3f})",
                         worst_bin_obs_over_exp=float(g.loc[i, "observed"] / max(g.loc[i, "expected"], 1e-12)),
                         fit_warning=bool(per_tier.mean() >= 0.5)))
    return pd.DataFrame(rows)
