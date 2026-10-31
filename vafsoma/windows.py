from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import adjacent_relative_change, safe_sd


# Depth-window selection. The window metric is the fixed-call density of a tier: the mean over the (non-control)
# organs or replicate groups of N_s / (2 G L_s), with N_s the MAP count of fixed calls of unit s, G the effective
# genome size of the tier and L_s the root-to-tip length of the unit. It is used only to find a stretch of depth tiers
# in which the calls are stable; it is reported in 03 and 04 but is not an estimate of anything.
# (vafSOMA <= 0.2.3 called the same quantity rate_MAP_fixed; the arithmetic is unchanged, so windows are identical.)
DENSITY_COLUMN = "fixed_call_density"
DENSITY_COUNT_COLUMN = "sum_MAP_fixed_count"


def fixed_call_density(per_unit_counts: pd.DataFrame, length_by_unit: dict[str, float], G: float) -> float:
    """Mean over units of MAP_fixed_count / (2 G L_s); units without a positive finite L_s are skipped."""
    ratios = []
    for _, row in per_unit_counts.iterrows():
        L_s = length_by_unit.get(str(row["sample"]))
        if L_s is None or not np.isfinite(L_s) or L_s <= 0:
            continue
        ratios.append(float(row["MAP_fixed_count"]) / (2.0 * G * L_s))
    return float(np.mean(ratios)) if ratios else float("nan")


def scan_windows(
    tier_df: pd.DataFrame,
    dataset_label: str,
    min_start: int = 12,
    min_cutoffs: int = 4,
    max_cv: float = 0.15,
    max_adjacent_change: float = 0.25,
) -> pd.DataFrame:
    rows: list[dict] = []
    for tree, gtree in tier_df.groupby("tree", sort=True):
        gtree = gtree.sort_values("depth_filter")
        depths = gtree["depth_filter"].astype(int).to_numpy()
        vals = gtree[DENSITY_COLUMN].astype(float).to_numpy()
        counts = gtree[DENSITY_COUNT_COLUMN].astype(float).to_numpy()
        n = len(depths)
        for i in range(n):
            for j in range(i, n):
                dwin = depths[i : j + 1]
                vwin = vals[i : j + 1]
                cwin = counts[i : j + 1]
                passes = False if dwin[0] < min_start else len(dwin) >= min_cutoffs
                mean_v = float(np.nanmean(vwin)) if len(vwin) else float("nan")
                sd_v = safe_sd(vwin)
                cv_v = float(sd_v / mean_v) if mean_v > 0 else float("inf")
                max_adj = adjacent_relative_change(vwin)
                passes = bool(passes and cv_v <= max_cv and max_adj <= max_adjacent_change)
                rows.append(
                    {
                        "dataset_label": dataset_label,
                        "tree": tree,
                        "window_start": int(dwin[0]),
                        "window_end": int(dwin[-1]),
                        "n_dp_cutoffs": int(len(dwin)),
                        "mean_density": mean_v,
                        "sd_density": sd_v,
                        "cv_density": cv_v,
                        "min_density": float(np.nanmin(vwin)),
                        "max_density": float(np.nanmax(vwin)),
                        "max_adjacent_relative_change": max_adj,
                        "mean_count": float(np.nanmean(cwin)) if len(cwin) else float("nan"),
                        "sd_count": safe_sd(cwin),
                        "passes_window_rule": passes,
                    }
                )
    return pd.DataFrame(rows)


def _score_window(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "n_dp_cutoffs": df["n_dp_cutoffs"].iloc[0],
            "mean_cv": df["cv_density"].mean(),
            "mean_adjacent": df["max_adjacent_relative_change"].mean(),
            "mean_density": df["mean_density"].mean(),
        }
    )


def select_windows(
    window_scan: pd.DataFrame,
    tier_df: pd.DataFrame,
    selected_window: tuple[int, int] | None = None,
    strategy: str = "auto",
) -> pd.DataFrame:
    trees = sorted(tier_df["tree"].astype(str).unique())

    if selected_window is not None:
        a, b = selected_window
        return pd.DataFrame(
            [
                {
                    "tree": tree,
                    "selected_window_start": int(a),
                    "selected_window_end": int(b),
                    "window_selection_type": "user_specified",
                }
                for tree in trees
            ]
        )

    ws = window_scan[window_scan["passes_window_rule"]] if not window_scan.empty else window_scan
    if ws.empty:
        # Fallback: use the widest available depth span per tree.
        rows = []
        for tree, g in tier_df.groupby("tree"):
            depths = sorted(g["depth_filter"].astype(int).unique())
            rows.append(
                {
                    "tree": str(tree),
                    "selected_window_start": int(depths[0]),
                    "selected_window_end": int(depths[-1]),
                    "window_selection_type": "fallback_all_depths_no_passing_window",
                }
            )
        return pd.DataFrame(rows)

    if strategy in {"auto", "common"}:
        grouped = ws.groupby(["window_start", "window_end"], as_index=False).filter(
            lambda x: set(x["tree"].astype(str)) == set(trees)
        )
        if not grouped.empty:
            score_rows = []
            for (wa, wb), gw in grouped.groupby(["window_start", "window_end"], sort=False):
                sc = _score_window(gw).to_dict()
                sc["window_start"] = wa
                sc["window_end"] = wb
                score_rows.append(sc)
            scores = pd.DataFrame(score_rows).sort_values(
                ["n_dp_cutoffs", "mean_cv", "mean_adjacent"], ascending=[False, True, True]
            )
            best = scores.iloc[0]
            a, b = int(best["window_start"]), int(best["window_end"])
            return pd.DataFrame(
                [
                    {
                        "tree": tree,
                        "selected_window_start": a,
                        "selected_window_end": b,
                        "window_selection_type": "common_auto",
                    }
                    for tree in trees
                ]
            )
        if strategy == "common":
            # common requested but none found: explicit fallback to tree-specific.
            strategy = "tree-specific"

    rows = []
    for tree, g in ws.groupby("tree"):
        best = g.sort_values(["n_dp_cutoffs", "cv_density", "max_adjacent_relative_change"], ascending=[False, True, True]).iloc[0]
        rows.append(
            {
                "tree": str(tree),
                "selected_window_start": int(best["window_start"]),
                "selected_window_end": int(best["window_end"]),
                "window_selection_type": "tree_specific_auto",
            }
        )
    # If some trees had no passing window, add fallback spans for them.
    done = {r["tree"] for r in rows}
    for tree, g in tier_df.groupby("tree"):
        if str(tree) not in done:
            depths = sorted(g["depth_filter"].astype(int).unique())
            rows.append(
                {
                    "tree": str(tree),
                    "selected_window_start": int(depths[0]),
                    "selected_window_end": int(depths[-1]),
                    "window_selection_type": "fallback_all_depths_tree_no_passing_window",
                }
            )
    return pd.DataFrame(rows)
