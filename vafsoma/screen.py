"""All-organs screen applied before the fit (vafSOMA 0.4.2, opt-in).

A candidate with alternative reads in every sampled non-control organ is removed before the mixture is fitted. In a
design where no lineage after the divergence from the control organ (matched normal) is shared by all sampled
organs, a mutation cannot be present in all of them; such candidates are recurrent artefacts (mapping, paralogs,
library-wide errors). Removing them before the fit, rather than from the calls afterwards, keeps them out of the
fitted background: otherwise they make the background heavier and true layer-fixed mutations are absorbed as noise.

Rule (per tree): at the lowest depth tier of the input, a variant is screened when every non-control sample of the
tree has at least `min_alt` alternative reads (samples without a row count as 0). The variant is then removed from
every tier and every sample of that tree. Design check: the tree needs at least one control organ
(EstimateConfig.control_organs) and at least three non-control samples; otherwise the screen raises an error.

The screen is only valid when the sampling design satisfies the condition above; it removes true mutations that are
fixed in every sampled organ (for example, when all samples come from one branch, or when a mutation drifted to
fixation in all sampled lineages but not in the control's lineage). It is therefore off by default.
"""
from __future__ import annotations

import pandas as pd

SCREEN_SUMMARY_COLUMNS = ["tree", "depth_filter_used", "n_noncontrol_samples", "min_alt", "n_variants_lowest_tier",
                          "n_screened", "fraction_screened"]


def all_organs_screen(variants: pd.DataFrame, control_by_tree: dict[str, set[str]], min_alt: int = 1
                      ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (filtered variants, per-tree summary, screened sites). `control_by_tree`: {tree: set of control ids}."""
    if int(min_alt) < 1:
        raise ValueError("all-organs screen: min_alt must be >= 1")
    keep_parts, summ, sites = [], [], []
    for tree, tv in variants.groupby("tree", sort=True):
        tree = str(tree)
        ctrl = set(control_by_tree.get(tree, set()))
        samples = sorted(tv["sample"].astype(str).unique())
        if not ctrl:
            raise ValueError(f"all-organs screen: tree {tree!r} has no control organ; the screen needs the matched "
                             "normal (control_organs) to define the organs that must not share a lineage")
        noncontrol = [s for s in samples if s not in ctrl]
        if len(noncontrol) < 3:
            raise ValueError(f"all-organs screen: tree {tree!r} has {len(noncontrol)} non-control samples; at least 3 "
                             "are needed (with fewer organs, true mutations shared by all of them are common)")
        d0 = int(tv["depth_filter"].min())
        t0 = tv[(tv["depth_filter"] == d0) & tv["sample"].astype(str).isin(noncontrol)]
        pos = t0[t0["alt_depth"] >= int(min_alt)].groupby("variant_id")["sample"].nunique()
        screened = set(pos[pos == len(noncontrol)].index)
        n_var = int(tv.loc[tv["depth_filter"] == d0, "variant_id"].nunique())
        summ.append(dict(tree=tree, depth_filter_used=d0, n_noncontrol_samples=len(noncontrol), min_alt=int(min_alt),
                         n_variants_lowest_tier=n_var, n_screened=len(screened),
                         fraction_screened=(len(screened) / n_var) if n_var else 0.0))
        if screened:
            s0 = (tv[(tv["depth_filter"] == d0) & tv["variant_id"].isin(screened)]
                  .drop_duplicates("variant_id")[["tree", "variant_id", "chrom", "pos", "ref", "alt"]])
            sites.append(s0)
        keep_parts.append(tv[~tv["variant_id"].isin(screened)])
    out = pd.concat(keep_parts, ignore_index=True) if keep_parts else variants.iloc[0:0].copy()
    sites_df = (pd.concat(sites, ignore_index=True) if sites
                else pd.DataFrame(columns=["tree", "variant_id", "chrom", "pos", "ref", "alt"]))
    return out, pd.DataFrame(summ, columns=SCREEN_SUMMARY_COLUMNS), sites_df
