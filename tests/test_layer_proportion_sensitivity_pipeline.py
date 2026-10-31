"""Integration test for the layer-proportion sensitivity sweep wired into
run_estimate() (see layer_proportion_sensitivity_sweep in estimate.py and
perturb_layer_proportions/sweep_layer_proportions in utils.py).

Unlike tests/test_layer_proportion_sensitivity.py (pure math on the
perturbation/sweep functions themselves), this exercises the full pipeline
path: EstimateConfig -> run_estimate -> 10_layer_proportion_sensitivity.csv,
including that the relative_step=0.0 baseline row reproduces the same
per-depth fit (and therefore the same mean counts) as the ordinary,
unperturbed selected-window summary, since both use the exact same single-
point mu grid and EM random seed for a given (tree, depth).

Runs with pytest, or directly:
    PYTHONPATH=. python3 tests/test_layer_proportion_sensitivity_pipeline.py
"""
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from vafsoma.estimate import EstimateConfig, run_estimate


def _build_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(7)
    samples = ["S1", "S2", "S3"]
    var_dir = tmp_path / "variants"
    var_dir.mkdir()
    for depth in [12, 13]:
        rows = []
        pos = 1000
        # p=0.02 -> noise-like (E); p=0.36 -> L2-like fixed mutations, so the
        # L2 sweep in particular should visibly move the counts as its assumed
        # proportion (and therefore derived mu_L2) is perturbed.
        for p, nvar in [(0.02, 15), (0.36, 6)]:
            for _ in range(nvar):
                pos += 1
                for sample in samples:
                    d = int(rng.integers(depth + 15, depth + 35))
                    y = int(rng.binomial(d, p))
                    rows.append(
                        {
                            "sample": sample,
                            "chrom": "chr1",
                            "pos": pos,
                            "ref": "A",
                            "alt": "T",
                            "ref_depth": d - y,
                            "alt_depth": y,
                            "DP": d,
                            "VAF": y / d,
                        }
                    )
        pd.DataFrame(rows).to_csv(var_dir / f"tree1_dp_{depth}_vaf.csv", index=False)

    variant_zip = tmp_path / "variants.zip"
    with zipfile.ZipFile(variant_zip, "w") as zf:
        for f in var_dir.glob("*.csv"):
            zf.write(f, f.name)

    genome = tmp_path / "genome.csv"
    pd.DataFrame(
        {"tree": ["tree1"] * 2, "depth_filter": [12, 13], "effective_genome_size": [1_000_000] * 2}
    ).to_csv(genome, index=False)
    root = tmp_path / "root.csv"
    pd.DataFrame({"sample": samples, "root2tip_length": [10, 11, 9]}).to_csv(root, index=False)
    return variant_zip, genome, root


def _run(tmp_path: Path, **overrides):
    variant_zip, genome, root = _build_inputs(tmp_path)
    kwargs = dict(
        variants=str(variant_zip),
        effective_genome_size=str(genome),
        root2tip=str(root),
        dataset_label="test",
        outdir=str(tmp_path / "out"),
        lambda_grid="30",
        lambda_refine=False,
        em_n_starts=2,
        selected_dp_window="12:13",
        write_plots=False,
        layer_proportion_sensitivity=True,
        layer_proportion_sensitivity_steps="-0.1,0.0,0.1",
        layer_proportion_sensitivity_layers="L1,L2",
    )
    kwargs.update(overrides)
    return run_estimate(EstimateConfig(**kwargs))


def test_sensitivity_sweep_output_written_when_enabled():
    with tempfile.TemporaryDirectory() as td:
        outputs = _run(Path(td))
        assert "layer_proportion_sensitivity" in outputs
        assert outputs["layer_proportion_sensitivity"].exists()


def test_sensitivity_sweep_off_by_default():
    with tempfile.TemporaryDirectory() as td:
        outputs = _run(Path(td), layer_proportion_sensitivity=False)
        assert "layer_proportion_sensitivity" not in outputs
        assert not (Path(td) / "out" / "10_layer_proportion_sensitivity.csv").exists()


def test_sensitivity_sweep_covers_expected_points_and_baseline_matches_selected_window():
    with tempfile.TemporaryDirectory() as td:
        outputs = _run(Path(td))
        sweep = pd.read_csv(outputs["layer_proportion_sensitivity"])
        selected = pd.read_csv(outputs["selected_window_summary"])

        # 1 baseline + (L1: -0.1,0.1) + (L2: -0.1,0.1) = 5 rows for the one tree.
        assert len(sweep) == 5
        assert set(sweep["layer_swept"]) == {"baseline", "L1", "L2"}
        baseline = sweep[sweep["layer_swept"] == "baseline"].iloc[0]
        assert baseline["relative_step"] == 0.0
        assert baseline["skipped_reason"] == "" or pd.isna(baseline["skipped_reason"])
        assert baseline["n_window_depths_used"] == 2

        # The baseline sweep point uses the exact same (single-point) mu grid
        # and EM seeding as the ordinary run, so it must reproduce the same
        # mean counts reported in 05_selected_window_summary.csv, not just something close.
        sel_row = selected[selected["tree"] == "tree1"].iloc[0]
        for c in ("mean_MAP_fixed_count", "mean_posterior_weighted_L2_count"):
            assert np.isclose(baseline[c], sel_row[c], rtol=1e-12, atol=1e-12)

        # Every non-baseline, non-skipped row is a real perturbation: its
        # layer_proportions string differs from the baseline's.
        for _, row in sweep[sweep["layer_swept"] != "baseline"].iterrows():
            assert pd.isna(row["skipped_reason"]) or row["skipped_reason"] == ""
            assert row["layer_proportions"] != baseline["layer_proportions"]
            assert row["mean_MAP_fixed_count"] >= 0


def test_sensitivity_sweep_records_infeasible_points_gracefully():
    with tempfile.TemporaryDirectory() as td:
        # L2=0.9 out of a 1.0 total can't take a +20% relative step (0.9*1.2=1.08
        # exceeds the total) -- default steps include +0.2, so this must show up
        # as a skipped (not crashing) row with NaN counts.
        outputs = _run(
            Path(td),
            layer_proportions="L1=0.10,L2=0.90,L3=0.0",
            layer_proportion_sensitivity_steps=None,  # default -0.2,-0.1,0.0,0.1,0.2
            layer_proportion_sensitivity_layers="L2",
        )
        sweep = pd.read_csv(outputs["layer_proportion_sensitivity"])
        infeasible = sweep[(sweep["layer_swept"] == "L2") & np.isclose(sweep["relative_step"], 0.2)]
        assert len(infeasible) == 1
        row = infeasible.iloc[0]
        assert pd.isna(row["mean_MAP_fixed_count"])
        assert isinstance(row["skipped_reason"], str) and len(row["skipped_reason"]) > 0
        assert row["n_window_depths_used"] == 0


if __name__ == "__main__":
    test_sensitivity_sweep_output_written_when_enabled()
    print("  PASS test_sensitivity_sweep_output_written_when_enabled")
    test_sensitivity_sweep_off_by_default()
    print("  PASS test_sensitivity_sweep_off_by_default")
    test_sensitivity_sweep_covers_expected_points_and_baseline_matches_selected_window()
    print("  PASS test_sensitivity_sweep_covers_expected_points_and_baseline_matches_selected_window")
    test_sensitivity_sweep_records_infeasible_points_gracefully()
    print("  PASS test_sensitivity_sweep_records_infeasible_points_gracefully")
    print("All tests passed.")
