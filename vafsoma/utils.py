from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


COMPONENTS = ("E", "L1", "L2", "L3")
LAYER_COMPONENTS = ("L1", "L2", "L3")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_float_list(text: str | None, default: Sequence[float]) -> list[float]:
    if text is None or str(text).strip() == "":
        return list(default)
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_lambda_grid(text: str | None, grid_min: float, grid_max: float, grid_n: int) -> np.ndarray:
    if text is not None and str(text).strip():
        vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
        vals = sorted(set(vals))
        if any(v <= 0 for v in vals):
            raise ValueError("All lambda grid values must be > 0.")
        return np.asarray(vals, dtype=float)
    if grid_min <= 0 or grid_max <= 0 or grid_n < 1:
        raise ValueError("Lambda grid bounds must be > 0 and grid size must be >= 1.")
    if grid_n == 1:
        return np.asarray([grid_min], dtype=float)
    return np.geomspace(grid_min, grid_max, grid_n)


def parse_window(text: str | None) -> tuple[int, int] | None:
    if text is None or str(text).strip() == "":
        return None
    m = re.match(r"^\s*(\d+)\s*[:,-]\s*(\d+)\s*$", str(text))
    if not m:
        raise ValueError("--selected-dp-window must have form START:END, e.g. 28:36")
    a, b = int(m.group(1)), int(m.group(2))
    if b < a:
        raise ValueError("selected DP window end must be >= start")
    return a, b


def parse_control_organs(text: str | None) -> dict[str, str | list[str]]:
    """Parse per-tree control/reference organ ids like 'tree1=leaf14,tree2=leaf19'.

    Used to mark the organ(s) the upstream variant caller (e.g. Mutect2 tumor-normal
    mode) treated as the baseline the other organs' variants were called against
    -- see EstimateConfig.control_organs for why these organs are excluded from the
    depth-window metric. A tree with no entry is unaffected.
    A tree with more than one control organ (e.g. a joint multi-sample Mutect2 call
    with two normal samples) is written with '+' between them, e.g.
    'tree2=leaf19+leaf17,tree3=leaf08+leaf19'."""
    if text is None or str(text).strip() == "":
        return {}
    out: dict[str, str | list[str]] = {}
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("--control-organs must look like tree1=leaf14,tree2=leaf19[+leaf17]")
        tree, organ = item.split("=", 1)
        tree, organ = tree.strip(), organ.strip()
        if not tree or not organ:
            raise ValueError("--control-organs must look like tree1=leaf14,tree2=leaf19[+leaf17]")
        if "+" in organ:
            organs = [o.strip() for o in organ.split("+") if o.strip()]
            if not organs:
                raise ValueError("--control-organs must look like tree1=leaf14,tree2=leaf19[+leaf17]")
            out[tree] = organs
        else:
            out[tree] = organ
    return out


def infer_tree_depth_from_name(name: str) -> tuple[str | None, int | None]:
    base = Path(name).name
    tree = None
    depth = None

    # Prefer explicit treeNN tokens.
    m_tree = re.search(r"(?i)(tree[_-]?\d+|tree[_-]?[A-Za-z0-9]+)", base)
    if m_tree:
        tree = m_tree.group(1).replace("_", "").replace("-", "")

    # Explicit depth-filter tokens. Avoid accidentally taking coverage columns.
    patterns = [
        r"(?i)(?:depth[_-]?filter|dp[_-]?filter|filter|cutoff|dp|depth)[_-]?(\d+)",
        r"(?i)(?:^|[_-])d(\d+)(?:[_\\.-]|$)",
    ]
    for pat in patterns:
        m = re.search(pat, base)
        if m:
            depth = int(m.group(1))
            break
    return tree, depth


def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def rename_by_aliases(df: pd.DataFrame, aliases: dict[str, Iterable[str]]) -> pd.DataFrame:
    df = normalize_colnames(df)
    lower_to_actual = {c.lower(): c for c in df.columns}
    ren = {}
    for target, names in aliases.items():
        if target in df.columns:
            continue
        for name in names:
            key = name.lower()
            if key in lower_to_actual:
                ren[lower_to_actual[key]] = target
                break
    if ren:
        df = df.rename(columns=ren)
    return df


def variant_key_frame(df: pd.DataFrame) -> pd.Series:
    return (
        df["chrom"].astype(str)
        + ":"
        + df["pos"].astype(str)
        + ":"
        + df["ref"].astype(str)
        + ">"
        + df["alt"].astype(str)
    )


def safe_sd(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(np.nanstd(arr, ddof=1))


def adjacent_relative_change(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1:
        return 0.0
    denom = np.maximum(np.abs(arr[:-1]), 1e-300)
    return float(np.nanmax(np.abs(np.diff(arr)) / denom))


def format_missing(values: Iterable[object]) -> str:
    vals = sorted({str(v) for v in values if pd.notna(v)})
    return ";".join(vals)


def parse_layer_proportions(text: str | None, default: dict[str, float] | None = None) -> dict[str, float]:
    """Parse layer proportions like 'L1=0.20,L2=0.72,L3=0.08'."""
    if text is None or str(text).strip() == "":
        if default is None:
            raise ValueError("Layer proportions are required.")
        props = dict(default)
    else:
        props = {}
        for item in str(text).split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError("--layer-proportions must look like L1=0.20,L2=0.72,L3=0.08")
            k, v = item.split("=", 1)
            layer = k.strip().upper()
            if layer not in LAYER_COMPONENTS:
                raise ValueError(f"Unknown layer in --layer-proportions: {layer}")
            props[layer] = float(v.strip())
    for layer in LAYER_COMPONENTS:
        props.setdefault(layer, 0.0)
    bad = {k: v for k, v in props.items() if not math.isfinite(float(v)) or float(v) < 0 or float(v) > 1}
    if bad:
        raise ValueError(f"Layer proportions must be finite values in [0,1]; bad values: {bad}")
    total = sum(float(props[k]) for k in LAYER_COMPONENTS)
    if total <= 0:
        raise ValueError("At least one layer proportion must be > 0.")
    # Do not force exact sum=1, but catch likely mistakes.
    if total > 1.000001:
        raise ValueError(f"Layer proportions sum to {total:.6g}, which is > 1.")
    return {k: float(props[k]) for k in LAYER_COMPONENTS}


def format_layer_proportions(props: dict[str, float]) -> str:
    return ";".join(f"{k}={float(props.get(k, 0.0)):.12g}" for k in LAYER_COMPONENTS)


def perturb_layer_proportions(props: dict[str, float], target_layer: str, relative_step: float) -> dict[str, float]:
    """Perturb one layer's assumed proportion by a relative fraction, redistributing
    the remainder proportionally among the OTHER active layers so the total is preserved.

    This is for sensitivity/stability sweeps against layer_proportions, which the
    pipeline otherwise always treats as a fixed, externally-known input (never
    fit) -- see the conversation this was built for: layer_proportions describes
    an assumed biological fact (organ composition), not a nuisance parameter to
    blend via the mu-grid's softmax averaging, so testing its stability calls
    for independent refits at each perturbed value, not a blended point estimate.

    target_layer's new value is old_value * (1 + relative_step). The other
    layers with a nonzero original proportion absorb the complementary change,
    each scaled to keep their ORIGINAL relative ratios among themselves --
    e.g. for L1=0.20/L2=0.72/L3=0.08, perturbing L2 by +20% moves L2 to 0.864
    and splits the remaining 0.136 between L1 and L3 in their original 0.20:0.08
    (=5:2) ratio, not equally. For a two-layer mixture (e.g. L1=0.1/L2=0.9),
    the one other layer necessarily absorbs the entire complementary change.

    relative_step=0.0 returns props unchanged (as a copy). Raises ValueError if
    target_layer isn't present/active, if the perturbed value would be negative,
    or if there is no other active layer to absorb a nonzero remainder.
    """
    if target_layer not in props:
        raise ValueError(f"Unknown target_layer {target_layer!r}; expected one of {list(props.keys())}")
    old_val = float(props[target_layer])
    if old_val <= 0:
        raise ValueError(f"Cannot perturb {target_layer!r}: its proportion is {old_val}, not active (>0).")
    if relative_step == 0.0:
        return dict(props)

    new_val = old_val * (1.0 + relative_step)
    if new_val < 0:
        raise ValueError(
            f"relative_step={relative_step} on {target_layer}={old_val} would give a negative proportion."
        )

    total = sum(float(v) for v in props.values())
    others = {k: float(v) for k, v in props.items() if k != target_layer}
    others_total_old = sum(v for v in others.values() if v > 0)
    remaining = total - new_val
    if remaining < -1e-12:
        raise ValueError(
            f"Perturbed {target_layer} proportion ({new_val:.6g}) exceeds the total ({total:.6g})."
        )
    remaining = max(remaining, 0.0)

    if others_total_old <= 0:
        if remaining > 1e-12:
            raise ValueError(
                f"No other active layer to absorb the remaining {remaining:.6g} after perturbing {target_layer}."
            )
        new_props = dict(props)
        new_props[target_layer] = new_val
        return new_props

    scale = remaining / others_total_old
    new_props = {k: (new_val if k == target_layer else float(v) * scale) for k, v in props.items()}
    return new_props


def sweep_layer_proportions(
    props: dict[str, float],
    relative_steps: Sequence[float] = (-0.2, -0.1, 0.0, 0.1, 0.2),
    layers: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Build a layer-proportion sensitivity sweep: for each active layer (default:
    every layer with a nonzero proportion in `props`), perturb it independently
    across `relative_steps` via `perturb_layer_proportions`, holding the others'
    RELATIVE ratios fixed. The relative_step=0.0 baseline is included exactly
    once (not once per layer), since it's the same point regardless of which
    layer is nominally "being swept".

    A step that is mathematically infeasible -- most commonly a positive
    relative_step on a layer whose proportion is already close to the total
    (e.g. L2=0.9 out of 1.0 can only increase by ~11% relative before
    exceeding the total, so a +20% step has no valid redistribution) -- is
    skipped rather than raised, since a routine sweep run across many
    (tree, depth) groups should degrade gracefully at the boundary instead of
    aborting the whole sweep. Each skipped point is recorded in the returned
    list with "layer_proportions": None and a "skipped_reason" string, so
    callers can see what was infeasible rather than have it silently vanish.

    Returns a list of {"layer_swept": str, "relative_step": float,
    "layer_proportions": dict | None, "skipped_reason": str | None} entries,
    in the order layers x relative_steps (baseline first, wherever it first
    appears in relative_steps).
    """
    if layers is None:
        layers = [layer for layer in LAYER_COMPONENTS if props.get(layer, 0.0) > 0]
    results: list[dict[str, object]] = []
    seen_baseline = False
    for layer in layers:
        for step in relative_steps:
            if step == 0.0:
                if seen_baseline:
                    continue
                seen_baseline = True
                results.append({
                    "layer_swept": "baseline", "relative_step": 0.0,
                    "layer_proportions": dict(props), "skipped_reason": None,
                })
                continue
            try:
                perturbed = perturb_layer_proportions(props, layer, step)
            except ValueError as exc:
                results.append({
                    "layer_swept": layer, "relative_step": float(step),
                    "layer_proportions": None, "skipped_reason": str(exc),
                })
                continue
            results.append({
                "layer_swept": layer, "relative_step": float(step),
                "layer_proportions": perturbed, "skipped_reason": None,
            })
    return results


def finite_or_nan(x: float) -> float:
    try:
        if math.isfinite(float(x)):
            return float(x)
    except Exception:
        pass
    return float("nan")
