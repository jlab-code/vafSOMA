"""Validation for perturb_layer_proportions / sweep_layer_proportions: the
layer-proportion sensitivity sweep the pipeline uses to test how stable a rate
estimate is to the ASSUMED (never-fit) layer_proportions input, as opposed to
the mu-grid's softmax-blended averaging used for genuinely uncertain nuisance
values (see the conversation this was built for -- layer_proportions describes
an externally-known biological fact, so its stability is tested with
independent refits at each perturbed value, not blended into one estimate).

Runs with pytest, or directly:
    PYTHONPATH=. python3 tests/test_layer_proportion_sensitivity.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from vafsoma.utils import perturb_layer_proportions, sweep_layer_proportions


def test_two_layer_case_other_layer_absorbs_entire_complementary_change():
    """Pseudobulk-style L1=0.1/L2=0.9: perturbing L1 by +20% must move it to
    0.12 and L2 down to exactly 0.88 (the only other active layer absorbs the
    whole complementary change), with the total unchanged at 1.0."""
    props = {"L1": 0.1, "L2": 0.9, "L3": 0.0}
    out = perturb_layer_proportions(props, "L1", 0.2)
    assert np.isclose(out["L1"], 0.12)
    assert np.isclose(out["L2"], 0.88)
    assert np.isclose(out["L3"], 0.0)
    assert np.isclose(sum(out.values()), sum(props.values()))


def test_two_layer_case_negative_step():
    props = {"L1": 0.1, "L2": 0.9, "L3": 0.0}
    out = perturb_layer_proportions(props, "L1", -0.2)
    assert np.isclose(out["L1"], 0.08)
    assert np.isclose(out["L2"], 0.92)


def test_three_layer_case_others_split_proportionally_to_original_ratio():
    """Walnut-style L1=0.20/L2=0.72/L3=0.08: perturbing L2 by +20% moves L2 to
    0.864, and the remaining 0.136 must split between L1 and L3 in their
    ORIGINAL 0.20:0.08 (=5:2) ratio, not equally."""
    props = {"L1": 0.20, "L2": 0.72, "L3": 0.08}
    out = perturb_layer_proportions(props, "L2", 0.2)
    assert np.isclose(out["L2"], 0.864)
    remaining = 1.0 - 0.864
    expected_L1 = remaining * (0.20 / 0.28)
    expected_L3 = remaining * (0.08 / 0.28)
    assert np.isclose(out["L1"], expected_L1)
    assert np.isclose(out["L3"], expected_L3)
    assert np.isclose(sum(out.values()), 1.0)


def test_zero_relative_step_returns_unchanged_copy():
    props = {"L1": 0.20, "L2": 0.72, "L3": 0.08}
    out = perturb_layer_proportions(props, "L2", 0.0)
    assert out == props
    assert out is not props  # a copy, not the same object


def test_perturbing_an_inactive_layer_raises():
    props = {"L1": 0.20, "L2": 0.72, "L3": 0.0}
    try:
        perturb_layer_proportions(props, "L3", 0.2)
        assert False, "expected ValueError for perturbing an inactive (zero) layer"
    except ValueError:
        pass


def test_step_that_would_make_target_negative_raises():
    props = {"L1": 0.20, "L2": 0.72, "L3": 0.08}
    try:
        perturb_layer_proportions(props, "L2", -1.5)  # 0.72 * (1-1.5) = -0.36
        assert False, "expected ValueError for a negative resulting proportion"
    except ValueError:
        pass


def test_unknown_target_layer_raises():
    props = {"L1": 0.20, "L2": 0.72, "L3": 0.08}
    try:
        perturb_layer_proportions(props, "L4", 0.1)
        assert False, "expected ValueError for an unknown layer key"
    except ValueError:
        pass


def test_sweep_two_layer_default_steps_baseline_appears_once():
    props = {"L1": 0.1, "L2": 0.9, "L3": 0.0}
    out = sweep_layer_proportions(props)
    # default active layers = L1, L2 (L3 excluded, it's 0); default steps has one
    # zero among 5 -- baseline should appear exactly once across the whole sweep,
    # not once per swept layer.
    baseline_rows = [r for r in out if r["relative_step"] == 0.0]
    assert len(baseline_rows) == 1
    assert baseline_rows[0]["layer_swept"] == "baseline"
    assert baseline_rows[0]["layer_proportions"] == props
    # 2 layers x 5 steps, minus the 1 duplicate baseline collapsed away = 9 rows
    # total, but L2=0.9 can't take a +20% relative step (0.9*1.2=1.08 > total
    # 1.0) -- that one point is gracefully skipped (layer_proportions=None),
    # not raised, so it's still present as a row, just marked infeasible.
    assert len(out) == 9
    swept_layers = {r["layer_swept"] for r in out}
    assert swept_layers == {"baseline", "L1", "L2"}
    infeasible = [r for r in out if r["layer_proportions"] is None]
    assert len(infeasible) == 1
    assert infeasible[0]["layer_swept"] == "L2" and infeasible[0]["relative_step"] == 0.2
    assert infeasible[0]["skipped_reason"] is not None
    # every non-skipped point is a real, valid perturbation
    for r in out:
        if r["layer_proportions"] is not None:
            assert np.isclose(sum(r["layer_proportions"].values()), 1.0)


def test_sweep_three_layer_covers_all_active_layers_and_matches_perturb():
    props = {"L1": 0.20, "L2": 0.72, "L3": 0.08}
    out = sweep_layer_proportions(props, relative_steps=(-0.2, 0.0, 0.2))
    # 3 layers x 3 steps, minus 2 duplicate baselines collapsed = 7 rows
    assert len(out) == 7
    for row in out:
        if row["layer_swept"] == "baseline":
            continue
        expected = perturb_layer_proportions(props, row["layer_swept"], row["relative_step"])
        assert row["layer_proportions"] == expected
        # every sweep point must still sum to the original total
        assert np.isclose(sum(row["layer_proportions"].values()), sum(props.values()))


def test_sweep_respects_explicit_layers_subset():
    props = {"L1": 0.20, "L2": 0.72, "L3": 0.08}
    out = sweep_layer_proportions(props, relative_steps=(-0.2, 0.0, 0.2), layers=["L2"])
    swept_layers = {r["layer_swept"] for r in out}
    assert swept_layers == {"baseline", "L2"}
    assert len(out) == 3  # 1 layer x 3 steps, baseline not duplicated (only 1 layer anyway)


if __name__ == "__main__":
    test_two_layer_case_other_layer_absorbs_entire_complementary_change()
    print("  PASS test_two_layer_case_other_layer_absorbs_entire_complementary_change")
    test_two_layer_case_negative_step()
    print("  PASS test_two_layer_case_negative_step")
    test_three_layer_case_others_split_proportionally_to_original_ratio()
    print("  PASS test_three_layer_case_others_split_proportionally_to_original_ratio")
    test_zero_relative_step_returns_unchanged_copy()
    print("  PASS test_zero_relative_step_returns_unchanged_copy")
    test_perturbing_an_inactive_layer_raises()
    print("  PASS test_perturbing_an_inactive_layer_raises")
    test_step_that_would_make_target_negative_raises()
    print("  PASS test_step_that_would_make_target_negative_raises")
    test_unknown_target_layer_raises()
    print("  PASS test_unknown_target_layer_raises")
    test_sweep_two_layer_default_steps_baseline_appears_once()
    print("  PASS test_sweep_two_layer_default_steps_baseline_appears_once")
    test_sweep_three_layer_covers_all_active_layers_and_matches_perturb()
    print("  PASS test_sweep_three_layer_covers_all_active_layers_and_matches_perturb")
    test_sweep_respects_explicit_layers_subset()
    print("  PASS test_sweep_respects_explicit_layers_subset")
    print("All tests passed.")
