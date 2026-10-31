"""Validation for build_candidate_fixed_mutations (09_candidate_fixed_mutations_<tree>.csv):
the researcher-facing headline table -- filtered to SNPs with fixation evidence,
one row each, with most-likely-layer plus every organ's status/P_fixed inlined.

Runs with pytest, or directly:
    PYTHONPATH=. python3 tests/test_candidate_fixed_mutations.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from vafsoma.estimate import CANDIDATE_CALLING_STATUSES, build_candidate_fixed_mutations


def _augmented_row(tree, pos, calling_status, mean_P_L1=0.0, mean_P_L2=0.0, mean_P_L3=0.0,
                    mean_P_fixed=None, stable_layer_call="", n_organs_positive=1, organs_positive="orgA"):
    if mean_P_fixed is None:
        mean_P_fixed = mean_P_L1 + mean_P_L2 + mean_P_L3
    return {
        "dataset_label": "TestDS", "tree": tree, "chrom": "chr1", "pos": pos, "ref": "A", "alt": "G",
        "selected_window_start": 20, "selected_window_end": 30, "window_n_dp_cutoffs": 3,
        "calling_status": calling_status, "n_dp_cutoffs_present_in_window": 3,
        "n_organs_positive_in_window": n_organs_positive, "organs_positive_in_window": organs_positive,
        "max_alt_depth_in_window": 10.0, "max_VAF_in_window": 0.3,
        "n_dp_cutoffs_classified": 3, "fraction_MAP_fixed": 1.0 if calling_status.startswith("called_fixed") else 0.5,
        "mean_P_E": max(0.0, 1 - mean_P_fixed), "mean_P_L1": mean_P_L1, "mean_P_L2": mean_P_L2, "mean_P_L3": mean_P_L3,
        "mean_P_fixed": mean_P_fixed, "stable_fixed_call": calling_status.startswith("called_fixed"),
        "stable_layer_call": stable_layer_call,
    }


def _organ_row(tree, pos, sample, organ_calling_status, mean_P_fixed=None):
    return {
        "dataset_label": "TestDS", "tree": tree, "chrom": "chr1", "pos": pos, "ref": "A", "alt": "G",
        "sample": sample, "selected_window_start": 20, "selected_window_end": 30,
        "n_dp_cutoffs_present_in_window": 3, "organ_calling_status": organ_calling_status,
        "n_dp_cutoffs_positive": 3 if mean_P_fixed is not None else 0,
        "fraction_MAP_fixed": None, "fraction_MAP_L1": None, "fraction_MAP_L2": None, "fraction_MAP_L3": None,
        "mean_P_E": None, "mean_P_L1": None, "mean_P_L2": None, "mean_P_L3": None,
        "mean_P_fixed": mean_P_fixed, "stable_fixed_call": False, "stable_layer_call": "",
    }


def test_filters_to_candidate_statuses_only():
    augmented = pd.DataFrame([
        _augmented_row("treeX", 100, "called_fixed_L2", mean_P_L2=0.95, stable_layer_call="L2"),
        _augmented_row("treeX", 200, "called_noise"),
        _augmented_row("treeX", 300, "not_evaluated_insufficient_depth", mean_P_fixed=float("nan"),
                        mean_P_L1=float("nan"), mean_P_L2=float("nan"), mean_P_L3=float("nan")),
        _augmented_row("treeX", 400, "evaluated_never_positive", mean_P_fixed=float("nan"),
                        mean_P_L1=float("nan"), mean_P_L2=float("nan"), mean_P_L3=float("nan")),
        _augmented_row("treeX", 500, "evaluated_unstable", mean_P_L2=0.4),
    ])
    organ_calls = pd.DataFrame([
        _organ_row("treeX", 100, "orgA", "called_fixed_L2", 0.95),
        _organ_row("treeX", 200, "orgA", "called_noise", 0.02),
        _organ_row("treeX", 500, "orgA", "evaluated_unstable", 0.4),
    ])

    result = build_candidate_fixed_mutations(augmented, organ_calls)
    assert set(result.keys()) == {"treeX"}
    out = result["treeX"]
    assert set(out["pos"]) == {100, 500}, "only called_fixed_* and evaluated_unstable rows should survive the filter"
    assert set(CANDIDATE_CALLING_STATUSES) == {
        "called_fixed_L1", "called_fixed_L2", "called_fixed_L3", "called_fixed_ambiguous_layer", "evaluated_unstable",
    }


def test_most_likely_layer_reported_even_when_not_stable():
    augmented = pd.DataFrame([
        _augmented_row("treeX", 500, "evaluated_unstable", mean_P_L1=0.1, mean_P_L2=0.4, mean_P_L3=0.05, stable_layer_call=""),
    ])
    organ_calls = pd.DataFrame([_organ_row("treeX", 500, "orgA", "evaluated_unstable", 0.55)])
    out = build_candidate_fixed_mutations(augmented, organ_calls)["treeX"]
    row = out.iloc[0]
    assert row["most_likely_layer"] == "L2"  # argmax(0.1, 0.4, 0.05) even though stable_layer_call is empty
    assert row["stable_layer_call"] == "" or pd.isna(row["stable_layer_call"])


def test_organ_columns_inlined_and_n_organs_called_fixed_counts_correctly():
    augmented = pd.DataFrame([
        _augmented_row("treeX", 100, "called_fixed_L2", mean_P_L2=0.9, stable_layer_call="L2",
                        n_organs_positive=2, organs_positive="orgA;orgB"),
    ])
    organ_calls = pd.DataFrame([
        _organ_row("treeX", 100, "orgA", "called_fixed_L2", 0.9),
        _organ_row("treeX", 100, "orgB", "evaluated_unstable", 0.4),
        _organ_row("treeX", 100, "orgC", "not_evaluated_insufficient_depth", None),
    ])
    out = build_candidate_fixed_mutations(augmented, organ_calls)["treeX"]
    row = out.iloc[0]

    assert "orgA_status" in out.columns and "orgA_P_fixed" in out.columns
    assert "orgB_status" in out.columns and "orgC_status" in out.columns
    assert row["orgA_status"] == "called_fixed_L2"
    assert row["orgA_P_fixed"] == 0.9
    assert row["orgB_status"] == "evaluated_unstable"
    assert row["orgC_status"] == "not_evaluated_insufficient_depth"
    assert pd.isna(row["orgC_P_fixed"])
    assert row["n_organs_called_fixed"] == 1
    assert row["organs_called_fixed"] == "orgA"


def test_different_trees_get_separate_tables_with_their_own_organ_columns():
    augmented = pd.DataFrame([
        _augmented_row("tree1", 100, "called_fixed_L2", mean_P_L2=0.9, stable_layer_call="L2"),
        _augmented_row("tree2", 200, "called_fixed_L3", mean_P_L3=0.9, stable_layer_call="L3"),
    ])
    organ_calls = pd.DataFrame([
        _organ_row("tree1", 100, "leaf09", "called_fixed_L2", 0.9),
        _organ_row("tree2", 200, "leaf04", "called_fixed_L3", 0.9),
    ])
    result = build_candidate_fixed_mutations(augmented, organ_calls)
    assert set(result.keys()) == {"tree1", "tree2"}
    assert "leaf09_status" in result["tree1"].columns
    assert "leaf09_status" not in result["tree2"].columns
    assert "leaf04_status" in result["tree2"].columns


if __name__ == "__main__":
    test_filters_to_candidate_statuses_only()
    print("  PASS test_filters_to_candidate_statuses_only")
    test_most_likely_layer_reported_even_when_not_stable()
    print("  PASS test_most_likely_layer_reported_even_when_not_stable")
    test_organ_columns_inlined_and_n_organs_called_fixed_counts_correctly()
    print("  PASS test_organ_columns_inlined_and_n_organs_called_fixed_counts_correctly")
    test_different_trees_get_separate_tables_with_their_own_organ_columns()
    print("  PASS test_different_trees_get_separate_tables_with_their_own_organ_columns")
    print("All tests passed.")
