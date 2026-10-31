"""Organ-sharing recurrence test (topology-free, additive to the core EM fit).

Motivation: vafSOMA pools all organ samples into one VAF distribution for fitting, and the
per-variant `calling_status` in `06_variant_calls_augmented.csv` is likewise
computed by pooling read-level evidence across whichever organs happen to
carry alt-supporting reads. Neither step uses the fact that organs are NOT
exchangeable -- they're correlated via the tree's branching structure, so a
real shared mutation's organ-presence pattern should recur across multiple
independent loci far more than pure background noise would produce, while an
unrelated background site landing positive in the same specific multi-organ
subset by pure chance is comparatively rare.

This module builds a small, self-contained statistical test for that
recurrence, entirely from outputs the pipeline already computes (Stage 1 below
needs no new classification code at all) plus one new Monte-Carlo null
calibration step:

  Stage 1 (`organ_presence_patterns`): turn each candidate variant's
    per-organ posterior (already computed by `build_organ_variant_calls` for
    EVERY organ, not just alt_depth>0 ones) into a discrete organ-presence
    pattern.
  Stage 2 (`simulate_null_organ_patterns`): simulate what pattern frequencies
    the FITTED background-noise model alone would produce, calibrated to
    THIS analysis's own fitted lambda_hat/mu/pi (not a separately-refit
    background-only dataset -- see that function's docstring for why that
    distinction matters).
  Stage 3 (`recurrence_significance_table`): flag subsets with a
    significant, multiple-testing-corrected excess of observed over
    chance-expected recurrence.
  Stage 4 (`four_point_filter`): a purely mathematical sanity filter (the
    classical phylogenetic split-compatibility condition) on which flagged
    subsets can jointly be real splits of one tree.
  Stage 5 (`apply_recurrence_boost`): promote variants matching a validated
    subset to a new, additive calling status.

Scope, precisely: this test has
real per-variant power only for non-trivial (n_organs in [2, n_total-1))
subsets. A universal (all-organ) pattern is statistically indistinguishable
from a background q-collision at the presence-pattern level regardless of
recurrence, and is therefore never used for boosting. A singleton
(single-organ) pattern CAN show a significant *aggregate* excess (evidence
that real private mutations exist at some rate above background), but that
aggregate excess gives no way to attribute any *individual* singleton call as
real vs. noise (no compatible sibling organ to cross-check against) -- so
singletons are reported in the significance table for transparency but are
never used for per-variant boosting either.

This is an *additive* diagnostic layer: `apply_recurrence_boost` never
modifies the existing `calling_status` column, only adds new columns
alongside it. Recurrence-validated calls do not change the counts written in
03, 16 and 17.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .model import ExponentialComponentCache, FitResult, classify_organ_observations_for_fit

KEY_COLS = ["chrom", "pos", "ref", "alt"]

# calling_status values (see estimate.build_augmented_variant_calls) that already
# represent a confident, pooled-evidence "real fixed mutation" call -- never
# overridden or need boosting, regardless of organ_pattern.
ALREADY_CONFIDENT_STATUSES = {
    "called_fixed_L1",
    "called_fixed_L2",
    "called_fixed_L3",
    "called_fixed_ambiguous_layer",
}

# New calling-status-like label for a variant that was NOT already confidently
# called, but whose organ-presence pattern matches a subset the recurrence
# test validated (significant excess over the calibrated null, and pairwise
# split-compatible with every other validated subset).
RECURRENCE_VALIDATED_STATUS = "called_fixed_shared_validated"


def enumerate_nontrivial_subsets(organs: Sequence[str]) -> list[tuple[str, ...]]:
    """All non-empty, non-universal subsets of `organs` -- the 2^n - 2
    organ-presence patterns a real tree split (or a background-noise
    coincidence) can occupy. Order follows `organs`' own order within each
    subset and subsets are yielded grouped by size (singletons first)."""
    organs = list(organs)
    n = len(organs)
    subsets: list[tuple[str, ...]] = []
    for k in range(1, n):
        subsets.extend(combinations(organs, k))
    return subsets


def organ_presence_patterns(
    organ_calls_window: pd.DataFrame,
    organs: Sequence[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Per-variant organ-presence pattern from the existing per-(variant,
    organ) `mean_P_fixed` posterior (`build_organ_variant_calls`'s
    08_organ_variant_calls_window output) -- computed there for every organ a
    tree has, regardless of whether that organ ever shows alt_depth>0, so no
    new classification is needed here.

    An organ with no positive observation anywhere in the window
    (organ_calling_status evaluated_never_positive /
    not_evaluated_insufficient_depth, mean_P_fixed NaN) is treated as absent
    (P_fixed=0) -- the same convention used elsewhere in the pipeline (see
    build_augmented_variant_calls' empty_rec).

    Returns one row per (chrom,pos,ref,alt) with:
      - organ_pattern: tuple of organ ids with mean_P_fixed > threshold
        (subset of `organs`, in `organs`' order).
      - n_organs_present: len(organ_pattern).
    """
    cols = KEY_COLS + ["organ_pattern", "n_organs_present"]
    if organ_calls_window.empty:
        return pd.DataFrame(columns=cols)

    organs = list(organs)
    oc = organ_calls_window.copy()
    oc["mean_P_fixed"] = oc["mean_P_fixed"].fillna(0.0)
    piv = oc.pivot_table(index=KEY_COLS, columns="sample", values="mean_P_fixed", aggfunc="first")
    piv = piv.reindex(columns=organs).fillna(0.0)
    presence = piv > threshold

    pattern = presence.apply(lambda r: tuple(o for o, v in zip(organs, r) if v), axis=1)
    out = pattern.reset_index()
    out = out.rename(columns={0: "organ_pattern"})
    out["n_organs_present"] = out["organ_pattern"].apply(len)
    return out[cols]


def simulate_null_organ_patterns(
    fit: FitResult,
    organs: Sequence[str],
    dp_by_organ: dict[str, np.ndarray],
    q_max: float,
    n_null_sites: int,
    threshold: float,
    exp_cache: ExponentialComponentCache,
    rng: np.random.Generator,
) -> pd.Series:
    """Monte-Carlo null distribution of organ-presence PATTERNS under "pure
    background noise, no real shared mutations".

    Calibrated to THIS fit's own lambda_hat/mu/pi (via
    classify_organ_observations_for_fit -- no EM re-fit), never to a
    separately-refit background-only dataset. A separate refit would infer
    its own, much smaller pi_L1 (it has no real signal), which makes its
    classification boundary artificially conservative and produces an
    invalid, over-optimistic null for this analysis: an EM fit's inferred
    layer proportion directly sets the classification threshold's leniency,
    so a null calibrated on a different (smaller) proportion systematically
    under-counts how often the ACTUAL fitted model would call a background
    site fixed.

    q is drawn once per synthetic site and shared across every organ at that
    site (matching the model's own E-component assumption: one locus-level
    nuisance rate, not independent per organ) -- this also means the null
    correctly captures a real correlation a naive per-organ-independent
    calculation would miss: an anomalously high-q site nudges several
    organs' read counts at once, inflating the chance of any ONE multi-organ
    subset appearing once, without inflating the chance of the SAME exact
    subset recurring across independent loci (which is what the significance
    test actually looks for).

    Each organ's per-site depth is resampled from `dp_by_organ[organ]` (that
    organ's own observed DP values in the real candidate data for this
    tree/window) rather than a fixed assumed depth, so real per-organ
    coverage heterogeneity is preserved in the null.

    Returns a pandas Series indexed by organ_pattern (tuple) with the raw
    Monte-Carlo count (including the empty pattern, which the significance
    test does not use).
    """
    organs = list(organs)
    if n_null_sites <= 0 or fit.lambda_hat <= 0:
        return pd.Series(dtype=int)

    q = rng.exponential(1.0 / fit.lambda_hat, size=n_null_sites)
    q = np.clip(q, 0.0, q_max)
    site_ids = np.arange(n_null_sites)

    present = np.zeros((n_null_sites, len(organs)), dtype=bool)
    for j, organ in enumerate(organs):
        dp_pool = np.asarray(dp_by_organ.get(organ, []), dtype=np.int64)
        if dp_pool.size == 0:
            continue
        dp = rng.choice(dp_pool, size=n_null_sites, replace=True).astype(np.int64)
        alt = rng.binomial(dp, q)
        df = pd.DataFrame(
            {
                "variant_id": site_ids.astype(str),
                "chrom": "null_site",
                "pos": site_ids,
                "ref": "N",
                "alt": "N",
                "sample": organ,
                "alt_depth": alt,
                "DP": dp,
            }
        )
        oc = classify_organ_observations_for_fit(df, fit, exp_cache)
        if oc.empty:
            continue
        p_fixed = np.zeros(n_null_sites, dtype=float)
        p_fixed[oc["pos"].to_numpy()] = oc["P_fixed"].to_numpy()
        present[:, j] = p_fixed > threshold

    patterns = [tuple(o for o, v in zip(organs, row) if v) for row in present]
    return pd.Series(patterns).value_counts()


def recurrence_significance_table(
    observed_patterns: pd.Series,
    null_patterns: pd.Series,
    n_null_sites: int,
    n_candidates: int,
    subsets: Sequence[tuple[str, ...]],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """One-sided Poisson excess test per non-trivial organ subset: is the
    subset's observed recurrence count among the tree's candidate variants
    significantly higher than the calibrated null (`simulate_null_organ_
    patterns`), scaled to `n_candidates`, would predict by chance alone?

    Bonferroni-corrected across `len(subsets)` simultaneous tests (2^n - 2
    for n organs) -- deliberately conservative: a flagged subset is one the
    data supports even after paying for testing every possible subset.

    The null rate per subset uses a continuity (Haldane-Anscombe-style)
    correction, (null_count + 0.5) / (n_null_sites + 1), instead of the raw
    Monte-Carlo ratio: for a rare, high-order subset (e.g. a 3-of-4-organ
    pattern) it's common for the null simulation to draw exactly zero
    occurrences in a finite sample -- that means the true null rate is
    merely below the Monte Carlo's resolution, NOT that it's exactly zero, so
    treating a raw ratio of 0 as "impossible" and any single observed
    occurrence as infinitely significant (p=0) is overconfident and was
    caught producing exactly that false positive during development. The correction costs essentially
    nothing for well-sampled subsets and prevents this failure mode for
    poorly-sampled ones.

    Returns one row per subset with columns: subset, n_organs, observed,
    expected_null, pvalue, significant -- sorted by pvalue ascending.
    """
    # Plain dicts, not Series.get(), deliberately: a pandas Series built from a
    # dict literal with variable-length tuple keys silently coerces its index
    # into a (padded, corrupting) MultiIndex, which breaks tuple lookups. The
    # real call sites here always build these via value_counts() on a list of
    # tuples (a plain Index, unaffected), but converting to dict up front
    # makes this function correct regardless of how its caller built the input.
    observed_map = dict(observed_patterns.items())
    null_map = dict(null_patterns.items())

    bonf_alpha = alpha / max(len(subsets), 1)
    rows = []
    for s in subsets:
        observed = int(observed_map.get(s, 0))
        null_rate = (null_map.get(s, 0) + 0.5) / (n_null_sites + 1) if n_null_sites > 0 else 0.0
        expected = null_rate * n_candidates
        if expected > 0:
            pvalue = float(stats.poisson.sf(observed - 1, expected))
        else:
            pvalue = 0.0 if observed > 0 else 1.0
        rows.append(
            dict(
                subset=s,
                n_organs=len(s),
                observed=observed,
                expected_null=expected,
                pvalue=pvalue,
                significant=pvalue < bonf_alpha,
            )
        )
    out = pd.DataFrame(rows, columns=["subset", "n_organs", "observed", "expected_null", "pvalue", "significant"])
    return out.sort_values("pvalue", kind="stable").reset_index(drop=True)


def split_compatible(a: Sequence[str], b: Sequence[str], organs: Sequence[str]) -> bool:
    """Classical four-point / split-compatibility condition: two subsets of a
    common taxon set can be splits of the SAME tree only if at least one of
    A\\B, B\\A, A&B, or (universe - A - B) is empty. Any real tree's splits
    are pairwise compatible by construction; an incompatible pair is a
    purely mathematical sign that at least one is not a genuine tree split,
    independent of any developmental-biology assumption."""
    A, B, universe = set(a), set(b), set(organs)
    return not (A - B) or not (B - A) or not (A & B) or not (universe - A - B)


def four_point_filter(
    significant_subsets: Sequence[tuple[str, ...]],
    organs: Sequence[str],
) -> tuple[list[tuple[str, ...]], list[tuple[tuple[str, ...], tuple[str, ...]]]]:
    """Among the subsets the significance test flagged, find pairwise
    incompatible pairs (see `split_compatible`). Any subset involved in at
    least one incompatible pair is dropped from the returned "kept" list --
    it and its partner are reported for QC/manual review rather than
    silently resolved, since deciding which of an incompatible pair is the
    real split (if either) needs domain judgement this module doesn't have.

    Returns (kept_subsets, incompatible_pairs).
    """
    bad_pairs: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    dropped: set[tuple[str, ...]] = set()
    for a, b in combinations(significant_subsets, 2):
        if not split_compatible(a, b, organs):
            bad_pairs.append((a, b))
            dropped.add(a)
            dropped.add(b)
    kept = [s for s in significant_subsets if s not in dropped]
    return kept, bad_pairs


def apply_recurrence_boost(
    augmented: pd.DataFrame,
    pattern_df: pd.DataFrame,
    validated_subsets: Sequence[tuple[str, ...]],
) -> pd.DataFrame:
    """Left-joins each candidate variant's Stage-1 organ_pattern onto
    `augmented` (the 06_variant_calls_augmented table for one tree) and adds:
      - organ_pattern: the Stage-1 presence pattern.
      - recurrence_validated: True iff the variant was NOT already
        confidently called (see ALREADY_CONFIDENT_STATUSES) and its
        organ_pattern exactly matches one of `validated_subsets`.
      - recurrence_status: `calling_status` unchanged for anything already
        confident or not recurrence-validated; RECURRENCE_VALIDATED_STATUS
        for anything newly validated.

    `calling_status` itself is never modified -- this is purely additive,
    and the counts in 03, 16 and 17 are not changed (see module docstring).
    """
    merged = augmented.merge(pattern_df[KEY_COLS + ["organ_pattern"]], on=KEY_COLS, how="left")
    valid = set(validated_subsets)
    already_confident = merged["calling_status"].isin(ALREADY_CONFIDENT_STATUSES)
    is_validated_pattern = merged["organ_pattern"].apply(lambda p: isinstance(p, tuple) and p in valid)
    merged["recurrence_validated"] = (~already_confident) & is_validated_pattern
    merged["recurrence_status"] = np.where(
        merged["recurrence_validated"], RECURRENCE_VALIDATED_STATUS, merged["calling_status"]
    )
    return merged


@dataclass
class RecurrenceReport:
    tree: str
    organs: list[str]
    significance_table: pd.DataFrame
    validated_subsets: list[tuple[str, ...]]
    incompatible_pairs: list[tuple[tuple[str, ...], tuple[str, ...]]]
    boosted_calls: pd.DataFrame
    pattern_df: pd.DataFrame


def build_recurrence_report(
    tree: str,
    augmented: pd.DataFrame,
    organ_calls_window: pd.DataFrame,
    variants: pd.DataFrame,
    fit: FitResult,
    window_depths: Sequence[int],
    organs: Sequence[str],
    q_max: float,
    n_null_sites: int,
    presence_threshold: float,
    alpha: float,
    seed: int,
    exp_cache: ExponentialComponentCache,
) -> RecurrenceReport | None:
    """Top-level orchestration for ONE tree: Stages 1-5 of the recurrence
    test. `organs` should already exclude that tree's control organ (see
    EstimateConfig.control_organs) -- a control organ is, by the upstream
    caller's own construction, expected to show almost no alt-supporting
    reads at the candidate set, so it would appear as "always absent" in
    every pattern and contributes nothing but noise to this test.

    Returns None if there are fewer than 3 usable organs (with 2 organs,
    every non-trivial subset is either a singleton or the full set -- there
    is no >=2-organ, non-universal subset for this test to validate) or if
    there's no data to work with for this tree.
    """
    organs = list(organs)
    if len(organs) < 3:
        return None

    tv_aug = augmented[augmented["tree"] == tree].copy()
    tv_oc = organ_calls_window[organ_calls_window["tree"] == tree]
    if tv_aug.empty or tv_oc.empty or fit is None:
        return None

    pattern_df = organ_presence_patterns(tv_oc, organs, threshold=presence_threshold)
    if pattern_df.empty:
        return None
    observed_patterns = pattern_df["organ_pattern"].value_counts()
    n_candidates = len(pattern_df)

    tv = variants[(variants["tree"] == tree) & (variants["depth_filter"].isin(list(window_depths)))]
    dp_by_organ = {o: tv.loc[tv["sample"] == o, "DP"].to_numpy() for o in organs}

    rng = np.random.default_rng(seed)
    null_patterns = simulate_null_organ_patterns(
        fit, organs, dp_by_organ, q_max, n_null_sites, presence_threshold, exp_cache, rng
    )

    subsets = enumerate_nontrivial_subsets(organs)
    sig_table = recurrence_significance_table(
        observed_patterns, null_patterns, n_null_sites, n_candidates, subsets, alpha
    )
    sig_table.insert(0, "tree", tree)

    # Boostable: significant, >=2 organs, and strictly fewer than all organs
    # (excludes both singletons and the universal pattern -- see module
    # docstring for why neither carries per-variant attribution power).
    candidate_boost = sig_table[
        sig_table["significant"] & (sig_table["n_organs"] >= 2) & (sig_table["n_organs"] < len(organs))
    ]
    boostable = [s for s in candidate_boost["subset"]]
    validated, incompatible = four_point_filter(boostable, organs)

    boosted = apply_recurrence_boost(tv_aug, pattern_df, validated)
    return RecurrenceReport(tree, organs, sig_table, validated, incompatible, boosted, pattern_df)
