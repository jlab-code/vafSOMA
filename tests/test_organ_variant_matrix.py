"""Validation for build_organ_variant_calls (the 08_organ_variant_calls_window.csv
per-SNP-per-organ audit table): the organ-level analogue of
build_augmented_variant_calls, re-deriving each organ's own posterior
classification (closed-form, from saved fit parameters -- no EM re-fit) without
pooling evidence across organs.

Runs with pytest, or directly:
    PYTHONPATH=. python3 tests/test_organ_variant_matrix.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from vafsoma.estimate import EstimateConfig, build_augmented_variant_calls, build_organ_variant_calls


def _cfg(**kwargs) -> EstimateConfig:
    defaults = dict(variants="v", effective_genome_size="g", root2tip="r", dataset_label="TestDS")
    defaults.update(kwargs)
    return EstimateConfig(**defaults)


def _variant_row(tree, chrom, pos, ref, alt, sample, depth_filter, alt_depth, dp=100):
    return {
        "tree": tree, "chrom": chrom, "pos": pos, "ref": ref, "alt": alt,
        "sample": sample, "depth_filter": depth_filter,
        "ref_depth": dp - alt_depth, "alt_depth": alt_depth, "DP": dp,
        "VAF": alt_depth / dp if dp else 0.0,
    }


def _fit_row(tree, depth_filter, lambda_hat=40.0, pi=(0.95, 0.0, 0.045, 0.005)):
    return {
        "tree": tree, "depth_filter": depth_filter, "active_components": "E;L1;L2;L3",
        "mu_L1": 0.10, "mu_L2": 0.36, "mu_L3": 0.04, "layer_rho": 0.0,
        "lambda_hat": lambda_hat, "loglik": -100.0,
        "pi_E": pi[0], "pi_L1": pi[1], "pi_L2": pi[2], "pi_L3": pi[3],
        "em_n_iter": 5, "em_converged": True,
        "n_positive_rows": 10, "n_unique_positive_variants": 3,
    }


def test_organ_matrix_distinguishes_no_data_ref_only_and_fixed_from_noise():
    cfg = _cfg()
    selected = pd.DataFrame([{"tree": "treeX", "selected_window_start": 20, "selected_window_end": 22}])
    fit_df = pd.DataFrame([_fit_row("treeX", d) for d in (20, 21, 22)])

    rows = []
    # V1 @ pos 100: strong, layer-consistent VAF (~0.5, high DP) in orgA across the
    # window -> should MAP to a fixed_L* call. orgB never appears at any window
    # depth for this SNP at all -> no_data. orgC appears in the window but always
    # ref-only (alt_depth=0) -> ref_only.
    for d in (20, 21, 22):
        rows.append(_variant_row("treeX", "chr1", 100, "A", "G", "orgA", d, alt_depth=50, dp=100))
        rows.append(_variant_row("treeX", "chr1", 100, "A", "G", "orgC", d, alt_depth=0, dp=100))
    # anchor at the loosest tier (depth_filter=5) so pos=100 is in the universe,
    # and give orgB a row there only (not in the window) to prove it's a real
    # candidate organ for this tree without ever reaching the window for this SNP.
    rows.append(_variant_row("treeX", "chr1", 100, "A", "G", "orgA", 5, alt_depth=50, dp=100))
    rows.append(_variant_row("treeX", "chr1", 100, "A", "G", "orgB", 5, alt_depth=3, dp=100))

    variants = pd.DataFrame(rows)
    out = build_organ_variant_calls(variants, fit_df, selected, cfg)

    assert set(out["sample"].unique()) == {"orgA", "orgB", "orgC"}
    row_a = out[(out["pos"] == 100) & (out["sample"] == "orgA")].iloc[0]
    row_b = out[(out["pos"] == 100) & (out["sample"] == "orgB")].iloc[0]
    row_c = out[(out["pos"] == 100) & (out["sample"] == "orgC")].iloc[0]

    assert row_a["organ_calling_status"].startswith("called_fixed_")
    assert row_a["mean_P_fixed"] > 0.5
    assert row_b["organ_calling_status"] == "not_evaluated_insufficient_depth"
    assert pd.isna(row_b["mean_P_fixed"])
    assert row_c["organ_calling_status"] == "evaluated_never_positive"
    assert pd.isna(row_c["mean_P_fixed"])


def test_organ_matrix_reproduces_classify_organ_observations_for_fit_directly():
    """Cross-check: the closed-form posterior this function re-derives from the
    saved fit-parameter row must match calling classify_organ_observations_for_fit
    directly on the same data with the same (mu, lambda, pi) -- i.e. reconstructing
    a FitResult from a fit_df row loses no information relevant to classification."""
    from vafsoma.model import ExponentialComponentCache, FitResult, MuTriple, classify_organ_observations_for_fit

    cfg = _cfg()
    selected = pd.DataFrame([{"tree": "treeX", "selected_window_start": 20, "selected_window_end": 20}])
    fit_row = _fit_row("treeX", 20, lambda_hat=37.5, pi=(0.9, 0.01, 0.08, 0.01))
    fit_df = pd.DataFrame([fit_row])

    rows = [_variant_row("treeX", "chr1", 300, "A", "G", "orgA", 20, alt_depth=18, dp=40)]
    rows.append(_variant_row("treeX", "chr1", 300, "A", "G", "orgA", 5, alt_depth=18, dp=40))  # loosest-tier anchor
    variants = pd.DataFrame(rows)

    out = build_organ_variant_calls(variants, fit_df, selected, cfg)
    got = out[(out["pos"] == 300) & (out["sample"] == "orgA")].iloc[0]

    exp_cache = ExponentialComponentCache(q_max=cfg.q_max, n_quad=cfg.n_quad)
    fit = FitResult(
        mu=MuTriple(L1=fit_row["mu_L1"], L2=fit_row["mu_L2"], L3=fit_row["mu_L3"]),
        lambda_hat=fit_row["lambda_hat"],
        pi=np.array([fit_row["pi_E"], fit_row["pi_L1"], fit_row["pi_L2"], fit_row["pi_L3"]]),
        loglik=fit_row["loglik"], n_positive_rows=10, n_unique_positive_variants=3, grouped=pd.DataFrame(),
    )
    df_depth = variants[(variants["depth_filter"] == 20)].copy()
    df_depth["variant_id"] = df_depth["chrom"] + ":" + df_depth["pos"].astype(str) + ">" + df_depth["alt"]
    direct = classify_organ_observations_for_fit(df_depth, fit, exp_cache)
    direct_row = direct.iloc[0]

    assert abs(got["mean_P_fixed"] - direct_row["P_fixed"]) < 1e-9
    assert abs(got["mean_P_L2"] - direct_row["P_L2"]) < 1e-9


CALLING_STATUS_VOCAB = {
    "not_evaluated_insufficient_depth",
    "evaluated_never_positive",
    "evaluated_unstable",
    "called_noise",
    "called_fixed_L1",
    "called_fixed_L2",
    "called_fixed_L3",
    "called_fixed_ambiguous_layer",
}


def test_organ_status_vocabulary_matches_snp_level_calling_status_exactly():
    """The two matrices (SNP-level 06, organ-level 08) must use the identical
    status vocabulary so they're directly comparable cell-for-cell -- an organ
    matrix cell and the SNP-level calling_status for the same row should read
    the same way if you ignore the pooling-across-organs difference."""
    cfg = _cfg()
    selected = pd.DataFrame([{"tree": "treeX", "selected_window_start": 20, "selected_window_end": 22}])
    fit_df = pd.DataFrame([_fit_row("treeX", d) for d in (20, 21, 22)])
    rows = []
    for d in (20, 21, 22):
        rows.append(_variant_row("treeX", "chr1", 100, "A", "G", "orgA", d, alt_depth=50, dp=100))
        rows.append(_variant_row("treeX", "chr1", 100, "A", "G", "orgC", d, alt_depth=0, dp=100))
    rows.append(_variant_row("treeX", "chr1", 100, "A", "G", "orgA", 5, alt_depth=50, dp=100))
    rows.append(_variant_row("treeX", "chr1", 100, "A", "G", "orgB", 5, alt_depth=3, dp=100))
    variants = pd.DataFrame(rows)
    out = build_organ_variant_calls(variants, fit_df, selected, cfg)

    org_statuses = set(out["organ_calling_status"].unique())
    assert org_statuses <= CALLING_STATUS_VOCAB, f"unexpected label(s) not in the shared vocabulary: {org_statuses - CALLING_STATUS_VOCAB}"

    variant_summary = pd.DataFrame(columns=[
        "tree", "chrom", "pos", "ref", "alt", "n_dp_cutoffs", "fraction_MAP_fixed",
        "fraction_MAP_L1", "fraction_MAP_L2", "fraction_MAP_L3", "mean_P_E", "mean_P_L1",
        "mean_P_L2", "mean_P_L3", "mean_P_fixed", "stable_fixed_call", "stable_layer_call",
    ])
    aug = build_augmented_variant_calls(variants, variant_summary, selected, cfg)
    snp_statuses = set(aug["calling_status"].unique())
    assert snp_statuses <= CALLING_STATUS_VOCAB, f"unexpected label(s) not in the shared vocabulary: {snp_statuses - CALLING_STATUS_VOCAB}"


if __name__ == "__main__":
    test_organ_matrix_distinguishes_no_data_ref_only_and_fixed_from_noise()
    print("  PASS test_organ_matrix_distinguishes_no_data_ref_only_and_fixed_from_noise")
    test_organ_matrix_reproduces_classify_organ_observations_for_fit_directly()
    print("  PASS test_organ_matrix_reproduces_classify_organ_observations_for_fit_directly")
    test_organ_status_vocabulary_matches_snp_level_calling_status_exactly()
    print("  PASS test_organ_status_vocabulary_matches_snp_level_calling_status_exactly")
    print("All tests passed.")
