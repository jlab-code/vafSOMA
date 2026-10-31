"""Tests for the replicate-group generalization (vafsoma.replicates)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vafsoma.model import ExponentialComponentCache, MuTriple, make_component_loglik  # noqa: E402
from vafsoma.replicates import (  # noqa: E402
    ReplicateLikelihood,
    build_units,
    collapse_pi,
    group_lengths,
    read_replicate_groups,
    sample_to_group,
)

MU = MuTriple(0.10, 0.3625, 0.0375)
ACTIVE = ("E", "L1", "L2", "L3")


def _toy(n_var=300, samples=("A", "B", "C", "D"), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(n_var):
        vaf = rng.choice([0.01, 0.0375, 0.10, 0.3625])
        for s in samples:
            d = int(30 + rng.poisson(5))
            y = int(rng.binomial(d, vaf))
            rows.append(dict(variant_id=f"v{v}", chrom="c1", pos=v, ref="A", alt="T", sample=s, alt_depth=y, DP=d))
    return pd.DataFrame(rows)


def test_singleton_groups_reduce_to_per_organ_likelihood():
    """With one organ per group, both background components equal the base E density
    and the layer components equal the base positive-only layer densities, exactly."""
    df = _toy()
    smap = {s: s for s in df["sample"].unique()}
    units = build_units(df, smap)
    cache = ExponentialComponentCache(q_max=0.5, n_quad=96)
    lik = ReplicateLikelihood(units, cache)
    for lam in (5.0, 60.0, 300.0):
        lc = lik.comp_loglik_units(lam, MU, ACTIVE)
        y = units["Y"][:, 0]
        d = units["D"][:, 0]
        base = make_component_loglik(y, d, MU, lam, cache, active_components=ACTIVE)
        assert np.allclose(lc[:, 0], base[:, 0], atol=1e-8)          # E_shared == E
        assert np.allclose(lc[:, 1], base[:, 0], atol=1e-8)          # E_private == E
        assert np.allclose(lc[:, 2:5], base[:, 1:], atol=1e-8)        # L1..L3


def test_units_only_positive_and_zeros_kept():
    df = _toy()
    smap = {"A": "g1", "B": "g1", "C": "g2", "D": "g2"}
    units = build_units(df, smap)
    assert (units["n_pos"] >= 1).all()
    assert (units["n_rep"] == 2).all()
    # zeros inside a positive unit are retained (fixed components use them)
    assert ((units["Y"] == 0) & units["M"]).any()


def test_dedup_counts_cover_all_units():
    df = _toy()
    units = build_units(df, {"A": "g1", "B": "g1", "C": "g2", "D": "g2"})
    lik = ReplicateLikelihood(units, ExponentialComponentCache())
    assert lik.counts.sum() == units["n"]
    assert len(lik.inverse) == units["n"]


def test_fixed_component_penalizes_missing_replicate():
    """A dominant-layer pattern with one replicate at zero is far less likely under L2
    than the same pattern with both replicates positive."""
    df = pd.DataFrame([
        dict(variant_id="v1", chrom="c", pos=1, ref="A", alt="T", sample="A", alt_depth=11, DP=30),
        dict(variant_id="v1", chrom="c", pos=1, ref="A", alt="T", sample="B", alt_depth=10, DP=30),
        dict(variant_id="v2", chrom="c", pos=2, ref="A", alt="T", sample="A", alt_depth=11, DP=30),
        dict(variant_id="v2", chrom="c", pos=2, ref="A", alt="T", sample="B", alt_depth=0, DP=30),
    ])
    units = build_units(df, {"A": "g", "B": "g"})
    lc = ReplicateLikelihood(units, ExponentialComponentCache()).comp_loglik_units(50.0, MU, ACTIVE)
    vids = units["meta"]["variant_id"].tolist()
    l2_both, l2_missing = lc[vids.index("v1"), 3], lc[vids.index("v2"), 3]
    assert l2_both - l2_missing > 10


def test_controls_are_singletons_and_lengths_averaged(tmp_path):
    f = tmp_path / "groups.tsv"
    pd.DataFrame({"tree": ["t1"] * 3, "sample": ["A", "B", "C"], "group": ["g1", "g1", "g2"]}).to_csv(f, sep="\t", index=False)
    rg = read_replicate_groups(f)
    smap = sample_to_group(rg, "t1", ["A", "B", "C", "N"], control_set={"N"})
    assert smap == {"A": "g1", "B": "g1", "C": "g2", "N": "control:N"}
    L = group_lengths({"A": 10.0, "B": 12.0, "C": 8.0, "N": 5.0}, smap)
    assert L["g1"] == 11.0 and L["g2"] == 8.0


def test_collapse_pi():
    assert np.allclose(collapse_pi(np.array([0.3, 0.2, 0.1, 0.3, 0.1])), [0.5, 0.1, 0.3, 0.1])


if __name__ == "__main__":
    import tempfile
    test_singleton_groups_reduce_to_per_organ_likelihood()
    test_units_only_positive_and_zeros_kept()
    test_dedup_counts_cover_all_units()
    test_fixed_component_penalizes_missing_replicate()
    with tempfile.TemporaryDirectory() as d:
        test_controls_are_singletons_and_lengths_averaged(Path(d))
    test_collapse_pi()
    print("all replicate tests passed")


def test_group_stability_uses_cutoffs_where_unit_is_present():
    """0.1.7: a unit observed at only some cutoffs of the window (site depth filter) and MAP L2 at all of them is
    called fixed in L2 (0.1.6 divided by all cutoffs of the window and returned evaluated_unstable)."""
    import pandas as pd
    from vafsoma.replicates import summarize_group_calls_window
    rows = []
    for d in (10, 11, 12, 13):                       # unit A present at all four cutoffs, unit B only at the first two
        for vid, pos, present in (("A", 1, True), ("B", 2, d <= 11)):
            if not present:
                continue
            rows.append(dict(dataset_label="t", tree="t1", depth_filter=d, variant_id=vid, chrom="c", pos=pos, ref="A",
                             alt="T", replicate_group="g1", n_replicates=2, n_positive_replicates=2,
                             obs_inferred_component_MAP="L2", obs_inferred_fixed_MAP=True, P_E=0.0, P_E_shared=0.0,
                             P_E_private=0.0, P_L1=0.0, P_L2=1.0, P_L3=0.0, P_fixed=1.0))
    oc = pd.DataFrame(rows)
    sel = pd.DataFrame([dict(tree="t1", selected_window_start=10, selected_window_end=13)])
    out = summarize_group_calls_window(oc, sel).set_index("pos")
    assert out.loc[1, "calling_status"] == "called_fixed_L2"
    assert out.loc[2, "calling_status"] == "called_fixed_L2"
    assert out.loc[2, "n_dp_cutoffs_present"] == 2
