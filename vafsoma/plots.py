from __future__ import annotations

from pathlib import Path

import pandas as pd

LAYERS = ("L1", "L2", "L3")


def write_count_plots(tier_df: pd.DataFrame, selected: pd.DataFrame, outdir: str | Path) -> None:
    """Plot the posterior-weighted fixed-mutation counts per layer (sums over non-control units) against the depth
    cutoff, with the selected window shaded. One PNG per tree in <outdir>/diagnostic_plots/.

    Plotting is non-essential: without matplotlib the other outputs are still written.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    out = Path(outdir) / "diagnostic_plots"
    out.mkdir(parents=True, exist_ok=True)
    for tree, g in tier_df.groupby("tree", sort=True):
        g = g.sort_values("depth_filter")
        srow = selected[selected["tree"].astype(str) == str(tree)]
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        for layer in LAYERS:
            col = f"sum_posterior_weighted_{layer}_count"
            if col in g.columns and g[col].abs().sum() > 0:
                ax.plot(g["depth_filter"], g[col], marker="o", ms=3, label=layer)
        if not srow.empty:
            ax.axvspan(int(srow.iloc[0]["selected_window_start"]), int(srow.iloc[0]["selected_window_end"]), alpha=0.15)
        ax.set_xlabel("depth cutoff")
        ax.set_ylabel("posterior-weighted fixed calls")
        ax.set_title(f"{tree}: fixed calls per depth tier (shaded: selected window)", fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"{tree}_fixed_counts_by_tier.png", dpi=180)
        plt.close(fig)
