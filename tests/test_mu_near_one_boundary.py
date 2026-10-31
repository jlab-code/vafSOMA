"""Validation for the phased single-layer mu-near-1 grid mechanism (see
MU_NEAR_ONE_TRIGGER / near_one_mu_grid / _layer_mu_grid /
_fixed_mu_with_boundary_grid in estimate.py).

Real single-layer sequencing (layer_specific_seq="yes") of phased data
(vaf_phasing="phased") with the default active_layer_proportion=1.0 derives
mu = vaf_scale(1.0) * 1.0 = 1.0 for the active layer -- a fixed point right at
the numerically degenerate boundary (a single ref-allele read from ordinary
sequencing/mapping error is then enough to force misclassification). Rather
than require the user to remember to switch --mu-mode to "grid" for exactly
this case, make_mu_grid auto-detects a derived mu at/above
mu_near_one_trigger and expands it into a small softmax-blended grid
approaching 1 from below, regardless of mu_mode -- this is the "use the grid
close to 1" mechanism for phased single-layer fits.

Runs with pytest, or directly:
    PYTHONPATH=. python3 tests/test_mu_near_one_boundary.py
"""
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from vafsoma.estimate import (
    MU_NEAR_ONE_OFFSETS,
    EstimateConfig,
    build_model_design,
    make_mu_grid,
    near_one_mu_grid,
    run_estimate,
)


def test_near_one_mu_grid_values_below_one_and_sorted():
    grid = near_one_mu_grid()
    assert len(grid) == len(set(grid)) == len(MU_NEAR_ONE_OFFSETS)
    assert grid == sorted(grid)
    assert all(0.0 < v < 1.0 for v in grid)
    # The largest offset (1e-2) should give the smallest value, closest to 0.99.
    assert np.isclose(grid[0], 0.99)
    # The smallest offset (1e-5) should give the value closest to 1.
    assert np.isclose(grid[-1], 1.0 - 1e-5)


def test_near_one_mu_grid_respects_custom_offsets():
    grid = near_one_mu_grid([0.5, 0.1])
    assert np.isclose(sorted(grid)[0], 0.5)
    assert np.isclose(sorted(grid)[1], 0.9)


def test_fixed_mode_phased_single_layer_at_proportion_one_auto_expands_to_grid():
    """layer_specific_seq=yes + phased + active_layer_proportion=1.0 (the
    default) derives mu=1.0 for the active layer -- must NOT collapse to a
    single fixed mu=1.0 point even though mu_mode defaults to 'fixed'."""
    cfg = EstimateConfig(
        variants="v", effective_genome_size="g", root2tip="r",
        layer_specific_seq="yes", vaf_phasing="phased", active_layer="L2",
    )
    design = build_model_design(cfg)
    assert design.derived_mu.L2 == 1.0
    grid = make_mu_grid(cfg, design)
    assert len(grid) > 1, "mu=1.0 must trigger the near-one grid even in fixed mode"
    l2_values = [mu.L2 for mu in grid]
    assert all(v is not None and v < 1.0 for v in l2_values)
    assert all(mu.L1 is None and mu.L3 is None for mu in grid)  # inactive layers stay None


def test_fixed_mode_ordinary_purity_unaffected():
    """A purity well below the boundary (e.g. 0.7) must keep the ordinary
    single fixed-point behavior -- the near-one mechanism must not fire for
    every layer-specific run, only the genuinely boundary-adjacent ones."""
    cfg = EstimateConfig(
        variants="v", effective_genome_size="g", root2tip="r",
        layer_specific_seq="yes", vaf_phasing="phased", active_layer="L2",
        active_layer_proportion=0.7,
    )
    design = build_model_design(cfg)
    assert design.derived_mu.L2 == 0.7
    grid = make_mu_grid(cfg, design)
    assert len(grid) == 1
    assert grid[0].L2 == 0.7


def test_fixed_mode_bulk_default_proportions_unaffected():
    """Regression guard: the existing bulk 3-layer default (L1=0.20/L2=0.72/
    L3=0.08, unphased) is nowhere near the boundary and must keep returning
    exactly one grid point, unchanged from before this feature existed."""
    cfg = EstimateConfig(variants="v", effective_genome_size="g", root2tip="r")
    design = build_model_design(cfg)
    grid = make_mu_grid(cfg, design)
    assert len(grid) == 1
    assert grid[0].L1 == design.derived_mu.L1
    assert grid[0].L2 == design.derived_mu.L2
    assert grid[0].L3 == design.derived_mu.L3


def test_explicit_mu_grid_override_still_wins_over_near_one():
    cfg = EstimateConfig(
        variants="v", effective_genome_size="g", root2tip="r",
        layer_specific_seq="yes", vaf_phasing="phased", active_layer="L2",
        mu_L2_grid="0.9,0.95",
    )
    design = build_model_design(cfg)
    grid = make_mu_grid(cfg, design)
    assert sorted(mu.L2 for mu in grid) == [0.9, 0.95]


def test_grid_mode_also_uses_near_one_grid_not_plusminus_20pct():
    """mu_mode='grid' must route the boundary layer through the same
    near_one_mu_grid, not the ordinary +/-20% local grid (which would try
    1.0*1.2=1.2, clipped down to collide with the base point)."""
    cfg = EstimateConfig(
        variants="v", effective_genome_size="g", root2tip="r",
        layer_specific_seq="yes", vaf_phasing="phased", active_layer="L2",
        mu_mode="grid",
    )
    design = build_model_design(cfg)
    grid_fixed_default = make_mu_grid(cfg, design)
    cfg_fixed = EstimateConfig(
        variants="v", effective_genome_size="g", root2tip="r",
        layer_specific_seq="yes", vaf_phasing="phased", active_layer="L2",
    )
    grid_fixed = make_mu_grid(cfg_fixed, design)
    assert sorted(mu.L2 for mu in grid_fixed_default) == sorted(mu.L2 for mu in grid_fixed)


def test_custom_trigger_and_offsets_are_respected():
    cfg = EstimateConfig(
        variants="v", effective_genome_size="g", root2tip="r",
        layer_specific_seq="yes", vaf_phasing="phased", active_layer="L2",
        active_layer_proportion=0.9,  # below the default 0.999 trigger
        mu_near_one_trigger=0.85,     # but above this custom, looser trigger
        mu_near_one_offsets="0.05,0.01",
    )
    design = build_model_design(cfg)
    assert design.derived_mu.L2 == 0.9
    grid = make_mu_grid(cfg, design)
    assert sorted(mu.L2 for mu in grid) == [0.95, 0.99]


def _build_phased_single_layer_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Synthetic real-single-layer-sequencing-style data: every organ IS the
    L2 layer (no cross-layer dilution), phased convention -- the
    phased single-layer fitting scenario this feature was built for."""
    rng = np.random.default_rng(11)
    samples = ["S1", "S2", "S3"]
    var_dir = tmp_path / "variants"
    var_dir.mkdir()
    for depth in [10, 11]:
        rows = []
        pos = 2000
        # p near 0 -> noise/E; p near 1 -> a real single-layer (phased) fixed
        # mutation, which is exactly the case mu=1.0 fixed cannot handle safely.
        for p, nvar in [(0.01, 12), (0.97, 5)]:
            for _ in range(nvar):
                pos += 1
                for sample in samples:
                    d = int(rng.integers(depth + 15, depth + 30))
                    y = int(rng.binomial(d, p))
                    rows.append({
                        "sample": sample, "chrom": "chr1", "pos": pos, "ref": "A", "alt": "T",
                        "ref_depth": d - y, "alt_depth": y, "DP": d, "VAF": y / d,
                    })
        pd.DataFrame(rows).to_csv(var_dir / f"treeX_dp_{depth}_vaf.csv", index=False)

    variant_zip = tmp_path / "variants.zip"
    with zipfile.ZipFile(variant_zip, "w") as zf:
        for f in var_dir.glob("*.csv"):
            zf.write(f, f.name)

    genome = tmp_path / "genome.csv"
    pd.DataFrame({"tree": ["treeX"] * 2, "depth_filter": [10, 11], "effective_genome_size": [500_000] * 2}).to_csv(genome, index=False)
    root = tmp_path / "root.csv"
    pd.DataFrame({"sample": samples, "root2tip_length": [8, 9, 7]}).to_csv(root, index=False)
    return variant_zip, genome, root


def test_pipeline_runs_end_to_end_for_phased_single_layer_at_proportion_one():
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        variant_zip, genome, root = _build_phased_single_layer_inputs(tmp_path)
        outputs = run_estimate(
            EstimateConfig(
                variants=str(variant_zip),
                effective_genome_size=str(genome),
                root2tip=str(root),
                dataset_label="phased_test",
                outdir=str(tmp_path / "out"),
                layer_specific_seq="yes",
                vaf_phasing="phased",
                active_layer="L2",
                lambda_grid="30",
                lambda_refine=False,
                em_n_starts=2,
                selected_dp_window="10:11",
                write_plots=False,
            )
        )
        fit_df = pd.read_csv(outputs["fit_parameters"])
        # More than one (mu, rho) grid point per depth confirms the near-one
        # grid actually engaged, not a single fixed mu=1.0 point.
        assert fit_df.groupby("depth_filter").size().min() > 1
        assert (fit_df["mu_L2"] < 1.0).all()
        tier_df = pd.read_csv(outputs["fixed_counts_by_tier"])
        assert (tier_df["sum_MAP_fixed_count"] >= 0).all()
        assert np.isfinite(tier_df["fixed_call_density"]).all()


if __name__ == "__main__":
    test_near_one_mu_grid_values_below_one_and_sorted()
    print("  PASS test_near_one_mu_grid_values_below_one_and_sorted")
    test_near_one_mu_grid_respects_custom_offsets()
    print("  PASS test_near_one_mu_grid_respects_custom_offsets")
    test_fixed_mode_phased_single_layer_at_proportion_one_auto_expands_to_grid()
    print("  PASS test_fixed_mode_phased_single_layer_at_proportion_one_auto_expands_to_grid")
    test_fixed_mode_ordinary_purity_unaffected()
    print("  PASS test_fixed_mode_ordinary_purity_unaffected")
    test_fixed_mode_bulk_default_proportions_unaffected()
    print("  PASS test_fixed_mode_bulk_default_proportions_unaffected")
    test_explicit_mu_grid_override_still_wins_over_near_one()
    print("  PASS test_explicit_mu_grid_override_still_wins_over_near_one")
    test_grid_mode_also_uses_near_one_grid_not_plusminus_20pct()
    print("  PASS test_grid_mode_also_uses_near_one_grid_not_plusminus_20pct")
    test_custom_trigger_and_offsets_are_respected()
    print("  PASS test_custom_trigger_and_offsets_are_respected")
    test_pipeline_runs_end_to_end_for_phased_single_layer_at_proportion_one()
    print("  PASS test_pipeline_runs_end_to_end_for_phased_single_layer_at_proportion_one")
    print("All tests passed.")
