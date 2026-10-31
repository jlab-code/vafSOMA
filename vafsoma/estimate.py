from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .io import read_effective_genome, read_root2tip, read_variants, root_lengths_by_sample
from .diagnostics import build_diagnostics, summarize_fit_check
from .screen import all_organs_screen
from .model import (
    ExponentialComponentCache,
    FitResult,
    MuTriple,
    average_grid_organ_calls,
    average_grid_variant_calls,
    classify_organ_observations_for_fit,
    classify_variants_for_fit,
    counts_from_variant_calls,
    fit_one_depth,
    COUNT_COLUMNS,
    organ_fixed_counts,
    softmax_grid_weights,
)
from .plots import write_count_plots
from .recurrence import RecurrenceReport, build_recurrence_report
from .replicates import (
    classify_group_units,
    classify_variants_pooled,
    fit_replicate_depth,
    group_lengths,
    sample_to_group,
    summarize_group_calls_window,
    tree_has_replicates,
)
from .utils import (
    COMPONENTS,
    LAYER_COMPONENTS,
    ensure_dir,
    format_layer_proportions,
    format_missing,
    parse_float_list,
    parse_lambda_grid,
    parse_layer_proportions,
    parse_window,
    sweep_layer_proportions,
    variant_key_frame,
)
from .windows import fixed_call_density, scan_windows, select_windows


DEFAULT_LAYER_PROPORTIONS = {"L1": 0.20, "L2": 0.72, "L3": 0.08}

# Version of the output specification (vafSOMA_docs/outputs.md): the files and columns that downstream analyses can
# rely on.
OUTPUT_SPEC_VERSION = "1"

# Phased single-layer sequencing boundary: when data is genuinely phased
# (vaf_phasing="phased", vaf_scale=1.0) AND the sample IS the layer itself --
# real single-layer sequencing with no cross-layer dilution, i.e.
# layer_specific_seq="yes" with active_layer_proportion at or near 1.0 -- the
# derived mu for the active layer is vaf_scale * proportion, which sits at or
# very near 1.0. A FIXED mu=1.0 point is numerically degenerate there: any
# single ref-allele read from ordinary sequencing/mapping error is then
# enough to force that observation's classification away from the layer
# component, however strong the alt-supporting evidence otherwise is. Rather
# than fit a single point on that knife-edge, a derived mu at/above this
# threshold is automatically expanded into a small grid of values approaching
# 1 from below (see near_one_mu_grid), blended the same softmax-weighted way
# as any other multi-point mu grid -- this reflects genuine numerical
# uncertainty about exactly how close to the boundary to place mu, which is
# exactly what the mu-grid mechanism already exists to handle (contrast with
# the layer_proportions sensitivity sweep, which instead uses independent,
# unblended refits, because layer_proportions is an assumed-known quantity,
# not a numerically uncertain one -- see layer_proportion_sensitivity_sweep).
# This override fires regardless of --mu-mode, since a fixed point here isn't
# a stylistic default so much as simply the wrong thing to fit.
MU_NEAR_ONE_TRIGGER = 0.999
MU_NEAR_ONE_OFFSETS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)


def near_one_mu_grid(offsets: Sequence[float] = MU_NEAR_ONE_OFFSETS) -> list[float]:
    """Grid of mu values approaching 1.0 from below, log-spaced like the
    lambda grid, for the phased single-layer boundary case (see
    MU_NEAR_ONE_TRIGGER). Deduplicated and sorted ascending."""
    vals = {float(np.clip(1.0 - float(off), 1e-6, 1 - 1e-9)) for off in offsets}
    return sorted(vals)


@dataclass
class ModelDesign:
    layer_specific_seq: str
    vaf_phasing: str
    vaf_scale: float
    active_components: tuple[str, ...]
    layer_proportions: dict[str, float]
    derived_mu: MuTriple
    # 0.4.1: organ-specific layer composition, {tree or "*": {sample: (s_L1, s_L2, s_L3)}} with s_k = a_k(organ) / a_k;
    # None = one composition for all organs (as in 0.4.0)
    organ_scale_by_tree: dict | None = None


@dataclass
class EstimateConfig:
    variants: str
    effective_genome_size: str
    root2tip: str
    dataset_label: str = "dataset"
    outdir: str = "vafsoma_results"

    # User-facing biological/VAF-scale design.
    layer_specific_seq: str = "no"
    vaf_phasing: str = "unphased"
    layer_proportions: str | None = None
    # 0.4.1: organ-specific layer proportions (bulk organs only): a table file (tab- or comma-separated; columns sample,
    # L1, L2, L3, optional tree) or a dict {sample: {"L1": .., "L2": .., "L3": ..}} or {(tree, sample): {...}}. Organs
    # not listed use layer_proportions. None = one composition for all organs.
    organ_layer_proportions: object = None
    active_layer: str | None = None
    active_layer_proportion: float | None = None

    # mu_source=derived calculates mu from layer proportions and VAF phasing.
    # mu_source=manual keeps legacy/debug behavior via --mu-L* or --mu-L*-grid.
    mu_source: str = "derived"
    mu_mode: str = "fixed"
    mu_L1: float = 0.10
    mu_L2: float = 0.36
    mu_L3: float = 0.04
    mu_L1_grid: str | None = None
    mu_L2_grid: str | None = None
    mu_L3_grid: str | None = None

    # Phased single-layer boundary handling (see MU_NEAR_ONE_TRIGGER's module
    # docstring): a derived active-layer mu at/above this threshold is
    # automatically expanded into a small softmax-blended grid approaching 1
    # from below, regardless of mu_mode, since a fixed point there is
    # numerically degenerate. mu_near_one_offsets overrides the default
    # log-spaced offsets (parsed like the other mu grids); both are exposed
    # mainly for testing/tuning, not everyday use.
    mu_near_one_trigger: float = 0.999
    mu_near_one_offsets: str | None = None

    # Beta-binomial overdispersion for the L1/L2/L3 layer components (optional
    # extension of the binomial model; see model.layer_loglik).
    # "fixed" with layer_rho=0.0 reproduces the original binomial model exactly.
    # 0.4.0: observation model: symmetric per-read sequencing error in every component; the optional provenance of
    # the calibration (preset name, config_sha256, source version) is recorded in the metadata.
    read_error: float = 0.0
    obs_preset: str | None = None
    obs_config_sha256: str | None = None
    obs_version: str | None = None
    layer_dispersion_mode: str = "fixed"
    layer_rho: float = 0.0
    layer_rho_grid: str | None = None

    q_max: float = 0.5
    n_quad: int = 96
    lambda_grid: str | None = None
    lambda_grid_min: float = 1.0
    lambda_grid_max: float = 500.0
    # This is now the COARSE grid only (lambda_refine below adds local
    # continuous refinement on top of it), so it no longer needs to be large:
    # validation on both synthetic and real data found 15 coarse points +
    # refinement matches or beats a 100-300 point fixed grid, in 2-4x less
    # time (see tests/test_lambda_refine.py and the empirical validation in
    # the PR/issue this change accompanies). The old default of 80 is kept
    # available via --lambda-grid-n for anyone who wants to reproduce
    # pre-refinement runs bit-for-bit with --no-lambda-refine.
    lambda_grid_n: int = 15
    # lambda_refine=True (default) replaces pure fixed-grid profiling with a
    # coarse-grid + local continuous-optimizer refinement (see
    # model._profile_lambda_search); this is both faster (fewer, cheaper EM
    # refits) and more accurate (not limited to the grid's resolution) than a
    # fixed grid alone. lambda_refine=False reproduces the old pure-grid
    # behavior exactly (useful for reproducing older results bit-for-bit).
    lambda_refine: bool = True
    selected_dp_window: str | None = None
    window_selection: str = "auto"
    min_window_start: int = 12
    min_window_cutoffs: int = 4
    max_window_cv: float = 0.15
    max_window_adjacent_change: float = 0.25
    n_jobs: int = 1
    em_max_iter: int = 1000
    em_tol: float = 1e-8
    em_n_starts: int = 16
    em_random_seed: int = 1
    stable_call_fraction: float = 1.0
    write_plots: bool = True
    # 0.1.6: post-fit diagnostics (output only; fits and calls are unchanged)
    write_diagnostics: bool = True
    resolvability_min_recall: float = 0.5
    fit_check_max_abs_residual: float = 4.0
    # 0.4.2: a layer is also unresolvable when mu_k lies below this quantile of the fitted background (None: recall
    # only, as in <= 0.4.1); see vafsoma.diagnostics.
    resolvability_background_quantile: float | None = 0.95
    # 0.4.2: all-organs screen before the fit (opt-in; needs control organs and >= 3 other samples per tree); see
    # vafsoma.screen.
    all_organs_screen: bool = False
    all_organs_screen_min_alt: int = 1

    # Per-tree control organ(s): the matched normal(s) used by the upstream caller (e.g. Mutect2). By construction
    # they show almost no alternative reads at the candidate sites, so they are excluded from the depth-window metric.
    # Their calls and counts are still written, flagged with is_control_organ / is_control. Value: one sample id or a
    # list of ids per tree (CLI: "tree2=leaf19+leaf17"); see _control_organ_set.
    control_organs: dict[str, str | list[str]] = field(default_factory=dict)

    # Layer-proportion sensitivity sweep: layer_proportions (design.layer_proportions)
    # is always treated as a fixed, externally-known input to the model -- it is
    # never fit -- so this is a robustness check, not part of the primary
    # estimate. Unlike the mu-grid (which blends genuinely-uncertain nuisance
    # values into ONE estimate via softmax weighting), this performs INDEPENDENT
    # refits at each perturbed layer_proportions value (default: every active
    # layer x 5 relative steps) and reports each one separately, so a real
    # sensitivity is visible rather than averaged away -- see
    # layer_proportion_sensitivity_sweep's docstring.
    #
    # This is a "routine" pipeline capability (always available, not a one-off
    # script) but is opt-IN (default False / --layer-proportion-sensitivity to
    # turn on), because it multiplies the number of EM refits by roughly
    # (n_active_layers * n_nonzero_steps + 1) on top of the main fit -- with
    # the default grid/em_n_starts settings that easily dwarfs the cost of the
    # primary estimate, so it should be requested deliberately rather than
    # silently slow down every ordinary run.
    layer_proportion_sensitivity: bool = False
    layer_proportion_sensitivity_steps: str | None = None  # default set in the sweep function: -0.2,-0.1,0,0.1,0.2
    layer_proportion_sensitivity_layers: str | None = None  # default: all layers with a nonzero proportion

    # Organ-sharing recurrence test (see the vafsoma.recurrence module docstring): an
    # additive diagnostic that flags which non-trivial organ subsets recur
    # across candidate loci significantly more than the fitted background-noise
    # model would produce by chance, and uses validated subsets to promote a
    # small number of borderline per-variant calls to a new calling status
    # (called_fixed_shared_validated) alongside (never replacing) the existing
    # calling_status. Opt-in (default False) since it adds a Monte-Carlo null
    # simulation per tree on top of the main fit, and is new/less battle-tested
    # than the rest of the pipeline -- it does not change the counts in 03/16/17.
    enable_recurrence_test: bool = False
    recurrence_null_sites: int = 20000
    recurrence_presence_threshold: float = 0.5
    recurrence_alpha: float = 0.05
    recurrence_seed: int = 0

    # Replicate groups (see vafsoma.replicates): optional {tree: {sample: group}}
    # map grouping organs that derive from the same meristem (e.g. neighbouring
    # leaves of one branch). For a tree listed here the mixture is fitted to
    # (variant, group) units -- one latent state per unit, zeros included for
    # fixed components, background split into shared and private parts -- and
    # groups are the units of the counts (16, 17) and of the depth-window metric. Trees not
    # listed (and every tree when this is None/empty) run the base model code
    # path unchanged; the group model with singleton groups is mathematically
    # identical to the base model.
    replicate_groups: dict[str, dict[str, str]] | None = None
    # 0.2.0: tree label for input tables without a 'tree' column or 'treeNN' file-name token (config input.tree)
    tree_label: str | None = None


def _norm_yes_no(x: str) -> str:
    y = str(x).strip().lower()
    if y not in {"yes", "no"}:
        raise ValueError("layer_specific_seq must be 'yes' or 'no'.")
    return y


def _norm_phasing(x: str) -> str:
    y = str(x).strip().lower()
    if y not in {"phased", "unphased"}:
        raise ValueError("vaf_phasing must be 'phased' or 'unphased'.")
    return y


def _norm_layer(x: str | None) -> str | None:
    if x is None or str(x).strip() == "":
        return None
    y = str(x).strip().upper()
    if y not in LAYER_COMPONENTS:
        raise ValueError("active_layer must be one of L1, L2, L3.")
    return y


def _control_organ_set(control_organs: dict, tree: str) -> set[str]:
    """Normalize an EstimateConfig.control_organs entry (str | list[str] | None)
    to a set of sample ids. See EstimateConfig.control_organs' docstring: a
    tree may have zero, one, or multiple control organs (e.g. a joint
    multi-sample Mutect2 call with two normal samples)."""
    val = control_organs.get(str(tree))
    if val is None:
        return set()
    if isinstance(val, (list, tuple, set)):
        return {str(v) for v in val}
    return {str(val)}


def read_organ_layer_proportions(src) -> dict[tuple[str, str], dict[str, float]]:
    """Organ-specific layer proportions (0.4.1) as {(tree or "*", sample): {L1, L2, L3}}; see
    EstimateConfig.organ_layer_proportions."""
    if src is None:
        return {}
    rows: dict[tuple[str, str], dict[str, float]] = {}
    if isinstance(src, (str, Path)):
        path = Path(src)
        if not path.exists():
            raise ValueError(f"organ_layer_proportions: file not found: {path}")
        sep = "\t" if path.read_text().splitlines()[0].count("\t") else ","
        tab = pd.read_csv(path, sep=sep, dtype={"sample": str, "tree": str})
        if "sample" not in tab.columns or not set(tab.columns) & set(LAYER_COMPONENTS):
            raise ValueError("organ_layer_proportions: the table needs a 'sample' column and L1, L2 and/or L3")
        for r in tab.to_dict("records"):
            tree = str(r["tree"]) if "tree" in tab.columns and pd.notna(r.get("tree")) else "*"
            rows[(tree, str(r["sample"]))] = {L: float(r[L]) for L in LAYER_COMPONENTS if L in tab.columns and pd.notna(r[L])}
    elif isinstance(src, dict):
        for k, v in src.items():
            tree, sample = (str(k[0]), str(k[1])) if isinstance(k, tuple) else ("*", str(k))
            rows[(tree, sample)] = {str(L).upper(): float(x) for L, x in dict(v).items()}
    else:
        raise ValueError("organ_layer_proportions must be a table path or a dict")
    if not rows:
        raise ValueError("organ_layer_proportions is empty")
    return rows


def organ_scales(props: dict[str, float], table: dict[tuple[str, str], dict[str, float]],
                 active_layers: Sequence[str]) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Scale factors s_k = a_k(organ) / a_k for the modelled layers (1 for the others), by tree ("*" = all trees)."""
    out: dict[str, dict[str, tuple[float, float, float]]] = {}
    for (tree, sample), lp in table.items():
        bad = set(lp) - set(LAYER_COMPONENTS)
        if bad:
            raise ValueError(f"organ_layer_proportions[{sample}]: unknown layers {sorted(bad)}")
        if any(not np.isfinite(v) or v < 0 or v > 1 for v in lp.values()) or sum(lp.values()) > 1.000001:
            raise ValueError(f"organ_layer_proportions[{sample}]: proportions must be in [0, 1] and sum to at most 1")
        sc = []
        for L in LAYER_COMPONENTS:
            if L in active_layers:
                if lp.get(L, 0.0) <= 0:
                    raise ValueError(f"organ_layer_proportions[{sample}]: modelled layer {L} needs a proportion > 0")
                sc.append(float(lp[L]) / float(props[L]))
            else:
                sc.append(1.0)
        out.setdefault(tree, {})[sample] = tuple(sc)
    return out


def organ_scale_for_tree(design: "ModelDesign", tree: str, samples: Sequence[str] | None = None) -> tuple | None:
    """MuTriple.organ_scale for one tree (tree-specific entries override "*"), or None."""
    if not design.organ_scale_by_tree:
        return None
    m = dict(design.organ_scale_by_tree.get("*", {}))
    own = design.organ_scale_by_tree.get(str(tree), {})
    if samples is not None and own:
        missing = set(own) - {str(x) for x in samples}
        if missing:
            raise ValueError(f"organ_layer_proportions for tree {tree!r}: samples not in the data: {sorted(missing)}")
    m.update(own)
    if samples is not None:
        m = {k: v for k, v in m.items() if k in {str(x) for x in samples}}
    return tuple(sorted(m.items())) or None


def build_model_design(cfg: EstimateConfig) -> ModelDesign:
    layer_specific = _norm_yes_no(cfg.layer_specific_seq)
    phasing = _norm_phasing(cfg.vaf_phasing)
    active_layer = _norm_layer(cfg.active_layer)
    vaf_scale = 1.0 if phasing == "phased" else 0.5

    if layer_specific == "no":
        if active_layer is not None:
            raise ValueError(
                "--active-layer can only be used when --layer-specific-seq yes. "
                "For bulk/unlayered data, active layers are inferred from --layer-proportions."
            )
        if cfg.active_layer_proportion is not None:
            raise ValueError("--active-layer-proportion can only be used when --layer-specific-seq yes.")
        props = parse_layer_proportions(cfg.layer_proportions, default=DEFAULT_LAYER_PROPORTIONS)
        active_layers = tuple(layer for layer in LAYER_COMPONENTS if props.get(layer, 0.0) > 0)
        if not active_layers:
            raise ValueError("At least one layer must have a positive layer proportion.")
        active_components = ("E",) + active_layers
        mu = MuTriple(
            *(vaf_scale * props[layer] if layer in active_layers else None for layer in LAYER_COMPONENTS)
        )
    else:
        if active_layer is None:
            raise ValueError("--active-layer is required when --layer-specific-seq yes.")
        if cfg.layer_proportions is not None and str(cfg.layer_proportions).strip() != "":
            raise ValueError("--layer-proportions cannot be used when --layer-specific-seq yes; use --active-layer-proportion instead.")
        prop = 1.0 if cfg.active_layer_proportion is None else float(cfg.active_layer_proportion)
        if not np.isfinite(prop) or prop <= 0 or prop > 1:
            raise ValueError("--active-layer-proportion must be in (0,1].")
        props = {layer: 0.0 for layer in LAYER_COMPONENTS}
        props[active_layer] = prop
        active_components = ("E", active_layer)
        vals = {layer: None for layer in LAYER_COMPONENTS}
        vals[active_layer] = vaf_scale * prop
        mu = MuTriple(vals["L1"], vals["L2"], vals["L3"])

    organ_scale_by_tree = None
    if cfg.organ_layer_proportions is not None:
        if layer_specific != "no":
            raise ValueError("organ_layer_proportions can only be used for bulk organs (layer_specific_seq no)")
        organ_scale_by_tree = organ_scales(props, read_organ_layer_proportions(cfg.organ_layer_proportions),
                                           [L for L in LAYER_COMPONENTS if L in active_components])
    return ModelDesign(
        layer_specific_seq=layer_specific,
        vaf_phasing=phasing,
        vaf_scale=vaf_scale,
        active_components=tuple(active_components),
        layer_proportions=props,
        derived_mu=mu,
        organ_scale_by_tree=organ_scale_by_tree,
    )


def _mu_with_inactive_none(mu: MuTriple, active_components: Sequence[str]) -> MuTriple:
    active = set(active_components)
    vals = {}
    for layer in LAYER_COMPONENTS:
        vals[layer] = getattr(mu, layer) if layer in active else None
    return MuTriple(vals["L1"], vals["L2"], vals["L3"])


def _layer_mu_grid(layer: str, cfg: EstimateConfig, base_value: float | None) -> list[float]:
    """Per-layer candidate mu values for the (derived, grid-mode) softmax-
    blended mu grid. An explicit --mu-L{layer}-grid always wins. Otherwise:
    an inactive layer (base_value None) contributes a single dummy 0.0
    (discarded by the active-layer filter in make_mu_grid, kept only so
    product() stays well-formed); a base value at/above MU_NEAR_ONE_TRIGGER
    uses near_one_mu_grid() instead of the ordinary local grid, since a
    boundary value isn't safely explored by a +/-20% multiplicative window
    (1.2x would exceed 1, and clipping collapses it onto the base point);
    every other base value keeps the existing local +/-20% grid.
    """
    text = getattr(cfg, f"mu_{layer}_grid")
    if text is not None and str(text).strip():
        return parse_float_list(text, [])
    if base_value is None:
        return [0.0]
    m = float(base_value)
    if m >= cfg.mu_near_one_trigger:
        return near_one_mu_grid(parse_float_list(cfg.mu_near_one_offsets, list(MU_NEAR_ONE_OFFSETS)))
    vals = [m * 0.8, m, m * 1.2]
    return [float(np.clip(v, 1e-6, 1 - 1e-6)) for v in vals]


def _fixed_mu_with_boundary_grid(cfg: EstimateConfig, design: ModelDesign) -> list[MuTriple] | None:
    """None if no active layer's derived mu sits at/above the phased single-
    layer boundary (MU_NEAR_ONE_TRIGGER); otherwise a small grid where each
    boundary layer varies over near_one_mu_grid() and every other active
    layer stays fixed at its single derived value -- i.e. "fixed" mode's
    ordinary single-point behavior, except automatically promoted to a grid
    for the specific layer(s) that need it. See MU_NEAR_ONE_TRIGGER's
    docstring for why this override applies even when --mu-mode is "fixed"."""
    base = design.derived_mu.as_dict()
    active = set(design.active_components)
    near_one_layers = [
        layer
        for layer in LAYER_COMPONENTS
        if layer in active and base[layer] is not None and float(base[layer]) >= cfg.mu_near_one_trigger
    ]
    if not near_one_layers:
        return None
    per_layer: dict[str, list[float | None]] = {}
    for layer in LAYER_COMPONENTS:
        if layer not in active:
            per_layer[layer] = [None]
        elif layer in near_one_layers:
            # Routed through _layer_mu_grid (not near_one_mu_grid directly) so
            # an explicit --mu-L{layer}-grid override still wins here too.
            per_layer[layer] = _layer_mu_grid(layer, cfg, base[layer])
        else:
            per_layer[layer] = [float(base[layer])]
    grid = [MuTriple(a, b, c) for a, b, c in product(per_layer["L1"], per_layer["L2"], per_layer["L3"])]
    return grid


def make_mu_grid(cfg: EstimateConfig, design: ModelDesign) -> list[MuTriple]:
    mu_source = str(cfg.mu_source).strip().lower()
    mu_mode = str(cfg.mu_mode).strip().lower()
    if mu_source not in {"derived", "manual"}:
        raise ValueError("mu_source must be 'derived' or 'manual'.")
    if mu_mode not in {"fixed", "grid"}:
        raise ValueError("mu_mode must be 'fixed' or 'grid'.")

    active = set(design.active_components)

    if mu_source == "derived" and mu_mode == "fixed":
        boundary_grid = _fixed_mu_with_boundary_grid(cfg, design)
        if boundary_grid is not None:
            return boundary_grid
        return [_mu_with_inactive_none(design.derived_mu, design.active_components)]

    if mu_source == "manual" and mu_mode == "fixed":
        return [_mu_with_inactive_none(MuTriple(cfg.mu_L1, cfg.mu_L2, cfg.mu_L3), design.active_components)]

    if mu_source == "manual" and mu_mode == "grid":
        l1 = parse_float_list(cfg.mu_L1_grid, [0.08, 0.10, 0.12])
        l2 = parse_float_list(cfg.mu_L2_grid, [0.32, 0.36, 0.40])
        l3 = parse_float_list(cfg.mu_L3_grid, [0.03, 0.04, 0.05])
    else:
        # Derived grid mode: use explicit grids if supplied, otherwise a small
        # local grid around each derived active mu value (or, at the phased
        # single-layer boundary, near_one_mu_grid() -- see _layer_mu_grid).
        # Inactive layers remain None.
        base = design.derived_mu.as_dict()
        l1 = _layer_mu_grid("L1", cfg, base["L1"])
        l2 = _layer_mu_grid("L2", cfg, base["L2"])
        l3 = _layer_mu_grid("L3", cfg, base["L3"])

    grid = []
    for a, b, c in product(l1, l2, l3):
        mu = MuTriple(a if "L1" in active else None, b if "L2" in active else None, c if "L3" in active else None)
        grid.append(mu)
    return grid


def make_rho_grid(cfg: EstimateConfig) -> list[float]:
    """Beta-binomial overdispersion grid for the L1/L2/L3 layer components.

    mode="fixed" (the default) returns a single value -- layer_rho, itself
    defaulting to 0.0 -- which reproduces the plain binomial model
    exactly and changes nothing for existing configs/runs.
    """
    mode = str(cfg.layer_dispersion_mode).strip().lower()
    if mode not in {"fixed", "grid"}:
        raise ValueError("layer_dispersion_mode must be 'fixed' or 'grid'.")
    if mode == "fixed":
        return [float(cfg.layer_rho)]
    grid = parse_float_list(cfg.layer_rho_grid, [0.0, 0.005, 0.01, 0.02, 0.05])
    bad = [r for r in grid if r < 0 or r >= 1]
    if bad:
        raise ValueError(f"layer_rho_grid values must be in [0, 1): {bad}")
    return [float(r) for r in grid]


def input_validation_summary(variants: pd.DataFrame, root: pd.DataFrame, genome: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    rows = []
    for tree, g in variants.groupby("tree", sort=True):
        samples = sorted(g["sample"].astype(str).unique())
        depths = sorted(g["depth_filter"].astype(int).unique())
        rt = root[root["tree"].astype(str) == str(tree)] if "tree" in root.columns else root
        root_samples = sorted(rt["sample"].astype(str).unique()) if "sample" in rt.columns else []
        if root_samples:
            missing_samples = sorted(set(samples) - set(root_samples))
            samples_match = len(missing_samples) == 0
            n_samples_root = len(root_samples)
        else:
            missing_samples = []
            samples_match = True
            n_samples_root = int(len(rt))
        ggen = genome[genome["tree"].astype(str) == str(tree)]
        genome_depths = set(ggen["depth_filter"].astype(int))
        missing_depths = sorted(set(depths) - genome_depths)
        rows.append(
            {
                "tree": str(tree),
                "dataset_label": dataset_label,
                "n_samples_variant": len(samples),
                "n_samples_root2tip": n_samples_root,
                "samples_match": bool(samples_match),
                "n_dp_cutoffs": len(depths),
                "dp_min": int(min(depths)) if depths else np.nan,
                "dp_max": int(max(depths)) if depths else np.nan,
                "effective_genome_size_available": len(missing_depths) == 0,
                "missing_samples": format_missing(missing_samples),
                "missing_depths": format_missing(missing_depths),
            }
        )
    return pd.DataFrame(rows)


def organ_lengths_for_group(
    df_depth: pd.DataFrame,
    genome: pd.DataFrame,
    root: pd.DataFrame,
) -> tuple[float, dict[str, float], list[str], bool]:
    """G of this tier and {sample: L_s} (root-to-tip lengths), for the depth-window metric and the outputs."""
    tree = str(df_depth["tree"].iloc[0])
    depth = int(df_depth["depth_filter"].iloc[0])
    gmatch = genome[(genome["tree"].astype(str) == tree) & (genome["depth_filter"].astype(int) == depth)]
    if gmatch.empty:
        raise ValueError(f"Missing effective genome size for tree={tree}, depth_filter={depth}")
    G = float(gmatch.iloc[0]["effective_genome_size"])
    samples = sorted(df_depth["sample"].astype(str).unique())
    length_by_sample, sample_specific = root_lengths_by_sample(root, tree, samples)
    missing = [s for s in samples if s not in length_by_sample]
    if missing:
        raise ValueError(
            f"Missing root-to-tip lengths for tree={tree}, samples={missing}. "
            "Either provide sample-specific lengths for all samples or an average-length table."
        )
    return G, length_by_sample, samples, sample_specific


def _fit_row(cfg: EstimateConfig, design: ModelDesign, tree: str, depth: int, fit: FitResult) -> dict:
    """Fit-parameter row (02 output); same keys as the base code path."""
    return {
        "dataset_label": cfg.dataset_label,
        "tree": str(tree),
        "depth_filter": int(depth),
        "layer_specific_seq": design.layer_specific_seq,
        "vaf_phasing": design.vaf_phasing,
        "vaf_scale": design.vaf_scale,
        "layer_proportions": format_layer_proportions(design.layer_proportions),
        "active_components": ";".join(design.active_components),
        "mu_source": cfg.mu_source,
        "mu_mode": cfg.mu_mode,
        "mu_L1": fit.mu.L1,
        "mu_L2": fit.mu.L2,
        "mu_L3": fit.mu.L3,
        "layer_dispersion_mode": cfg.layer_dispersion_mode,
        "layer_rho": fit.rho,
        "lambda_hat": fit.lambda_hat,
        "loglik": fit.loglik,
        "pi_E": fit.pi[0],
        **({"pi_E_error": fit.pi_E_error} if cfg.read_error > 0 else {}),
        "pi_L1": fit.pi[1],
        "pi_L2": fit.pi[2],
        "pi_L3": fit.pi[3],
        "em_n_starts": cfg.em_n_starts,
        "em_n_iter": fit.n_iter,
        "em_converged": fit.converged,
        "n_positive_rows": fit.n_positive_rows,
        "n_unique_positive_variants": fit.n_unique_positive_variants,
    }


def fit_tree_depth_task(
    key: tuple[str, int],
    df_depth: pd.DataFrame,
    genome: pd.DataFrame,
    root: pd.DataFrame,
    cfg: EstimateConfig,
    design: ModelDesign,
    mu_grid: list[MuTriple],
    rho_grid: list[float],
    lambda_grid: np.ndarray,
) -> tuple[list[dict], pd.DataFrame, dict, pd.DataFrame, list[dict]]:
    """Fit one (tree, depth tier) and return: fit rows (02), variant calls, the tier summary row (03), the per-unit
    posteriors (17) and the per-unit counts (16)."""
    tree, depth = key
    exp_cache = ExponentialComponentCache(q_max=cfg.q_max, n_quad=cfg.n_quad, read_error=cfg.read_error)
    G, length_by_sample, samples, sample_specific_length = organ_lengths_for_group(df_depth, genome, root)
    # 0.4.1: organ-specific composition -> every mu of the grid carries the organ scale factors of this tree
    organ_scale = organ_scale_for_tree(design, tree, samples)
    if organ_scale is not None:
        mu_grid = [dataclasses.replace(m, organ_scale=organ_scale) for m in mu_grid]

    # Control organs (matched normals) are classified like every organ but excluded from the window metric.
    control_set = _control_organ_set(cfg.control_organs, tree)
    unit_samples = [s for s in samples if s not in control_set]
    missing_control = control_set - set(samples)
    if missing_control:
        raise ValueError(
            f"control_organs[{tree!r}]={sorted(missing_control)!r} not among this tree's samples {samples}; "
            "check the configured control-organ id(s) for typos."
        )

    replicate_mode = tree_has_replicates(cfg.replicate_groups, tree)
    if replicate_mode:
        smap = sample_to_group(cfg.replicate_groups, tree, samples, control_set)
        # Groups replace organs as units of the counts and of the window metric; control organs are singleton
        # groups and stay excluded from the metric.
        group_length = group_lengths(length_by_sample, smap)
        units = sorted({smap[s] for s in unit_samples})
        control_units = {smap[s] for s in control_set}

    fit_rows = []
    calls_by_grid: list[pd.DataFrame] = []
    organ_calls_by_grid: list[pd.DataFrame] = []
    loglik_by_grid: list[float] = []
    grid_points = list(product(mu_grid, rho_grid))
    for gi, (mu, rho) in enumerate(grid_points):
        if replicate_mode:
            fit, rep_extras, rep_units, rep_lik = fit_replicate_depth(
                df_depth,
                smap,
                mu=mu,
                lambda_grid=lambda_grid,
                exp_cache=exp_cache,
                active_components=design.active_components,
                rho=rho,
                em_n_starts=cfg.em_n_starts,
                em_random_seed=cfg.em_random_seed + int(depth) + 1000 * gi,
                em_max_iter=cfg.em_max_iter,
                em_tol=cfg.em_tol,
                lambda_refine=cfg.lambda_refine,
            )
            calls_by_grid.append(classify_variants_pooled(fit, rep_extras, rep_units, rep_lik))
            organ_calls_by_grid.append(classify_group_units(fit, rep_extras, rep_units, rep_lik))
            loglik_by_grid.append(fit.loglik)
            fit_rows.append(_fit_row(cfg, design, tree, depth, fit) | {
                "replicate_mode": True,
                "pi_E_shared": rep_extras["pi_E_shared"],
                "pi_E_private": rep_extras["pi_E_private"],
                **({"pi_E_error": rep_extras["pi_E_error"]} if cfg.read_error > 0 else {}),
                "n_replicate_units": rep_extras["n_units"],
                "n_multi_replicate_units": rep_extras["n_multi_replicate_units"],
            })
            continue
        fit = fit_one_depth(
            df_depth,
            mu=mu,
            lambda_grid=lambda_grid,
            exp_cache=exp_cache,
            em_max_iter=cfg.em_max_iter,
            em_tol=cfg.em_tol,
            active_components=design.active_components,
            em_n_starts=cfg.em_n_starts,
            em_random_seed=cfg.em_random_seed + int(depth) + 1000 * gi,
            rho=rho,
            lambda_refine=cfg.lambda_refine,
        )
        calls_by_grid.append(classify_variants_for_fit(df_depth, fit, exp_cache))
        organ_calls_by_grid.append(classify_organ_observations_for_fit(df_depth, fit, exp_cache))
        loglik_by_grid.append(fit.loglik)
        fit_rows.append(
            {
                "dataset_label": cfg.dataset_label,
                "tree": str(tree),
                "depth_filter": int(depth),
                "layer_specific_seq": design.layer_specific_seq,
                "vaf_phasing": design.vaf_phasing,
                "vaf_scale": design.vaf_scale,
                "layer_proportions": format_layer_proportions(design.layer_proportions),
                "active_components": ";".join(design.active_components),
                "mu_source": cfg.mu_source,
                "mu_mode": cfg.mu_mode,
                "mu_L1": fit.mu.L1,
                "mu_L2": fit.mu.L2,
                "mu_L3": fit.mu.L3,
                "layer_dispersion_mode": cfg.layer_dispersion_mode,
                "layer_rho": fit.rho,
                "lambda_hat": fit.lambda_hat,
                "loglik": fit.loglik,
                "pi_E": fit.pi[0],
                **({"pi_E_error": fit.pi_E_error} if cfg.read_error > 0 else {}),
                "pi_L1": fit.pi[1],
                "pi_L2": fit.pi[2],
                "pi_L3": fit.pi[3],
                "em_n_starts": cfg.em_n_starts,
                "em_n_iter": fit.n_iter,
                "em_converged": fit.converged,
                "n_positive_rows": fit.n_positive_rows,
                "n_unique_positive_variants": fit.n_unique_positive_variants,
            }
        )

    if len(grid_points) == 1:
        variant_calls = calls_by_grid[0]
        organ_calls = organ_calls_by_grid[0]
    else:
        weights = softmax_grid_weights(loglik_by_grid)
        variant_calls = average_grid_variant_calls(calls_by_grid, weights=weights)
        organ_calls = average_grid_organ_calls(organ_calls_by_grid, weights=weights)

    empty_variant_cols = [
        "variant_id", "chrom", "pos", "ref", "alt", "P_E", "P_L1", "P_L2", "P_L3", "P_fixed",
        "variant_inferred_component_MAP", "variant_inferred_fixed_MAP",
        "variant_inferred_fixed_highconf_95", "variant_inferred_layer_MAP",
    ]
    if variant_calls.empty:
        variant_calls = pd.DataFrame(columns=empty_variant_cols)
    variant_calls.insert(0, "depth_filter", int(depth))
    variant_calls.insert(0, "tree", str(tree))
    variant_calls.insert(0, "dataset_label", cfg.dataset_label)

    empty_organ_cols = [
        "variant_id", "chrom", "pos", "ref", "alt", "sample", "P_E", "P_L1", "P_L2", "P_L3", "P_fixed",
        "obs_inferred_component_MAP", "obs_inferred_fixed_MAP", "obs_inferred_fixed_highconf_95",
    ]
    if organ_calls.empty:
        organ_calls = pd.DataFrame(columns=empty_organ_cols)
    organ_calls.insert(0, "depth_filter", int(depth))
    organ_calls.insert(0, "tree", str(tree))
    organ_calls.insert(0, "dataset_label", cfg.dataset_label)
    # Flag (don't drop) the control organs' own rows: they are written with the other units, flagged.
    if replicate_mode:
        organ_calls["is_control_organ"] = organ_calls["sample"].astype(str).isin(control_units)
    else:
        organ_calls["is_control_organ"] = organ_calls["sample"].astype(str).isin(control_set)

    # Background level per layer at this tier: the mean P_L{n} of observations whose MAP state is the background
    # (pooled over all units, control included). A property of the fit, reported in 03.
    background_by_layer: dict[str, float] = {}
    if not organ_calls.empty and "obs_inferred_component_MAP" in organ_calls.columns:
        noise_rows = organ_calls[organ_calls["obs_inferred_component_MAP"].astype(str) == "E"]
        for layer in ("L1", "L2", "L3"):
            background_by_layer[layer] = float(noise_rows[f"P_{layer}"].mean()) if not noise_rows.empty else 0.0
    else:
        for layer in ("L1", "L2", "L3"):
            background_by_layer[layer] = 0.0

    # Unique-variant counts (variant-level posteriors pooled over organs).
    uniquevariant_counts = counts_from_variant_calls(
        variant_calls if not variant_calls.empty else pd.DataFrame(columns=empty_variant_cols)
    )

    if replicate_mode:
        unit_type = "replicate_group"
        all_units = sorted({smap[s] for s in samples})
        control_ids = control_units
        unit_members = {u: sorted(s for s in samples if smap[s] == u) for u in all_units}
        unit_length = group_length
    else:
        unit_type = "organ"
        all_units = list(samples)
        control_ids = control_set
        unit_members = {s: [s] for s in samples}
        unit_length = length_by_sample
        units = unit_samples
    fixation_criterion = "replicate_group" if replicate_mode else "single_organ"
    unit_counts = organ_fixed_counts(organ_calls, all_units)
    metric_counts = unit_counts[unit_counts["sample"].isin(units)].set_index("sample").loc[units].reset_index()
    density = fixed_call_density(metric_counts, unit_length, G)

    tier_row = {
        "dataset_label": cfg.dataset_label,
        "tree": str(tree),
        "depth_filter": int(depth),
        "effective_genome_size": G,
        "n_samples": len(samples),
        "control_organ": ";".join(sorted(control_set)),
        "unit_type": unit_type,
        "fixation_criterion": fixation_criterion,
        "n_units": len(units),
        "root2tip_mode": "sample_specific" if sample_specific_length else "tree_or_global_average",
        "layer_specific_seq": design.layer_specific_seq,
        "vaf_phasing": design.vaf_phasing,
        "vaf_scale": design.vaf_scale,
        "active_components": ";".join(design.active_components),
        "layer_proportions": format_layer_proportions(design.layer_proportions),
        "uniquevariant_MAP_fixed_count": uniquevariant_counts["MAP_fixed_count"],
        "uniquevariant_highconf95_fixed_count": uniquevariant_counts["highconf95_fixed_count"],
        "uniquevariant_posterior_weighted_fixed_count": uniquevariant_counts["posterior_weighted_fixed_count"],
        "background_P_L1": background_by_layer["L1"],
        "background_P_L2": background_by_layer["L2"],
        "background_P_L3": background_by_layer["L3"],
    }
    # Sums over the non-control units (the units of the window metric).
    for c in COUNT_COLUMNS:
        tier_row[f"sum_{c}"] = float(metric_counts[c].sum()) if not metric_counts.empty else 0.0
    tier_row["fixed_call_density"] = density

    unit_rows = []
    for r in unit_counts.itertuples(index=False):
        u = str(r.sample)
        row = {
            "dataset_label": cfg.dataset_label,
            "tree": str(tree),
            "depth_filter": int(depth),
            "unit": u,
            "unit_type": unit_type,
            "fixation_criterion": fixation_criterion,
            "members": ";".join(unit_members.get(u, [u])),
            "is_control": u in control_ids,
            "root2tip_length": float(unit_length.get(u, np.nan)),
            "effective_genome_size": G,
        }
        for c in COUNT_COLUMNS:
            row[c] = getattr(r, c)
        unit_rows.append(row)
    if organ_scale is not None:
        for fr in fit_rows:
            fr["organ_layer_scale"] = json.dumps({k: list(v) for k, v in organ_scale})
    return fit_rows, variant_calls, tier_row, organ_calls, unit_rows


def annotate_summary_with_diagnostics(summary: pd.DataFrame, resolvability: pd.DataFrame,
                                      fit_summary: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add per-layer `{layer}_model_implied_recall`, `{layer}_resolvable` and `{layer}_status` ("reportable" or
    "unresolvable"; "no_diagnostics" when the layer has no recall) and the tree's `fit_warning` to the selected-window
    summary. Pure annotation: no count is changed or removed."""
    out = summary.copy()
    if out.empty:
        return out
    trees = out["tree"].astype(str)
    rv = resolvability.copy() if resolvability is not None else pd.DataFrame()
    if not rv.empty:
        rv["tree"] = rv["tree"].astype(str)
    for layer in ("L1", "L2", "L3"):
        rec, res = {}, {}
        if not rv.empty:
            sub = rv[rv["layer"].astype(str) == layer]
            rec = dict(zip(sub["tree"], sub["mean_model_implied_recall"].astype(float)))
            res = dict(zip(sub["tree"], sub["resolvable"].astype(bool)))
        out[f"{layer}_model_implied_recall"] = [rec.get(x, np.nan) for x in trees]
        out[f"{layer}_resolvable"] = pd.array([res.get(x, pd.NA) for x in trees], dtype="boolean")
        out[f"{layer}_status"] = ["no_diagnostics" if x not in res else ("reportable" if res[x] else "unresolvable")
                                       for x in trees]
    if fit_summary is not None and not fit_summary.empty:
        fw = dict(zip(fit_summary["tree"].astype(str), fit_summary["fit_warning"].astype(bool)))
        out["fit_warning"] = pd.array([fw.get(x, pd.NA) for x in trees], dtype="boolean")
    return out


def summarize_selected_windows(
    tier_df: pd.DataFrame,
    selected: pd.DataFrame,
    calls_all: pd.DataFrame,
    cfg: EstimateConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per tree: the selected window with the mean fixed-mutation counts per tier (sums over non-control units), and
    the per-variant summary over the window."""
    summary_rows = []
    variant_summary_rows = []

    for s in selected.itertuples(index=False):
        tree = str(s.tree)
        a = int(s.selected_window_start)
        b = int(s.selected_window_end)
        twin = tier_df[
            (tier_df["tree"].astype(str) == tree)
            & (tier_df["depth_filter"].astype(int) >= a)
            & (tier_df["depth_filter"].astype(int) <= b)
        ].sort_values("depth_filter")
        depths = [int(x) for x in twin["depth_filter"].tolist()]
        row = {
            "dataset_label": cfg.dataset_label,
            "tree": tree,
            "selected_window_start": a,
            "selected_window_end": b,
            "window_selection_type": s.window_selection_type,
            "n_window_cutoffs": len(depths),
            "unit_type": twin["unit_type"].iloc[0] if not twin.empty else "",
            "fixation_criterion": twin["fixation_criterion"].iloc[0] if not twin.empty else "",
            "n_units": int(twin["n_units"].iloc[0]) if not twin.empty else 0,
        }
        for c in COUNT_COLUMNS:
            row[f"mean_{c}"] = float(twin[f"sum_{c}"].mean()) if not twin.empty else np.nan
        summary_rows.append(row)

        ctree = calls_all[
            (calls_all["tree"].astype(str) == tree)
            & (calls_all["depth_filter"].astype(int) >= a)
            & (calls_all["depth_filter"].astype(int) <= b)
        ].copy()
        if not ctree.empty:
            total_selected_cutoffs = len(depths)
            ctree["is_MAP_L1"] = ctree["variant_inferred_layer_MAP"].astype(str) == "L1"
            ctree["is_MAP_L2"] = ctree["variant_inferred_layer_MAP"].astype(str) == "L2"
            ctree["is_MAP_L3"] = ctree["variant_inferred_layer_MAP"].astype(str) == "L3"
            for vid, g in ctree.groupby("variant_id", sort=False):
                first = g.iloc[0]
                n_present = int(g["depth_filter"].nunique())
                frac_fixed = float(g["variant_inferred_fixed_MAP"].mean())
                frac_l1 = float(g["is_MAP_L1"].mean())
                frac_l2 = float(g["is_MAP_L2"].mean())
                frac_l3 = float(g["is_MAP_L3"].mean())
                layer_fracs = {"L1": frac_l1, "L2": frac_l2, "L3": frac_l3}
                best_layer, best_layer_frac = max(layer_fracs.items(), key=lambda kv: kv[1])
                stable_fixed = frac_fixed >= cfg.stable_call_fraction
                stable_layer = best_layer if stable_fixed and best_layer_frac >= cfg.stable_call_fraction else ""
                variant_summary_rows.append(
                    {
                        "dataset_label": cfg.dataset_label,
                        "tree": tree,
                        "chrom": first["chrom"],
                        "pos": first["pos"],
                        "ref": first["ref"],
                        "alt": first["alt"],
                        "n_dp_cutoffs": n_present,
                        "n_selected_dp_cutoffs": total_selected_cutoffs,
                        "fraction_MAP_fixed": frac_fixed,
                        "fraction_MAP_L1": frac_l1,
                        "fraction_MAP_L2": frac_l2,
                        "fraction_MAP_L3": frac_l3,
                        "mean_P_E": float(g["P_E"].mean()),
                        "mean_P_L1": float(g["P_L1"].mean()),
                        "mean_P_L2": float(g["P_L2"].mean()),
                        "mean_P_L3": float(g["P_L3"].mean()),
                        "mean_P_fixed": float(g["P_fixed"].mean()),
                        "stable_fixed_call": bool(stable_fixed),
                        "stable_layer_call": stable_layer,
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(variant_summary_rows)


def layer_proportion_sensitivity_sweep(
    variants: pd.DataFrame,
    genome: pd.DataFrame,
    root: pd.DataFrame,
    selected: pd.DataFrame,
    cfg: EstimateConfig,
    design: ModelDesign,
    rho_grid: list[float],
    lambda_grid: np.ndarray,
) -> pd.DataFrame:
    """Robustness check for the ASSUMED, never-fit design.layer_proportions:
    for each tree's already-selected DP window, independently refit at each
    perturbed layer_proportions point from sweep_layer_proportions (default
    +/-10%/+/-20% relative steps) and report how the fixed-mutation counts move, instead
    of blending perturbed points into one estimate the way the mu-grid does
    for genuinely uncertain nuisance values -- see sweep_layer_proportions'
    and perturb_layer_proportions' docstrings for why that distinction matters
    here (layer_proportions describes an externally-known biological fact, so
    testing its stability calls for independent refits, not softmax blending).

    Refitting is bounded to each tree's already-selected window depths only
    (not the full DP range), since the window has already been chosen as the
    stable region under the baseline (unperturbed) proportions -- this keeps
    the extra cost to one additional fit per (window depth, sweep point)
    rather than repeating the full window-scan per perturbation.

    A sweep point that is mathematically infeasible (e.g. perturbing a layer
    whose proportion is already close to the total) is still reported, with
    the counts as NaN and skipped_reason set, matching
    sweep_layer_proportions' own graceful-skip behavior -- a routine sweep run
    across many trees should degrade gracefully at the boundary, not silently
    drop rows.
    """
    if selected.empty:
        return pd.DataFrame()

    steps = parse_float_list(cfg.layer_proportion_sensitivity_steps, [-0.2, -0.1, 0.0, 0.1, 0.2])
    layers_cfg = None
    if cfg.layer_proportion_sensitivity_layers is not None and str(cfg.layer_proportion_sensitivity_layers).strip():
        layers_cfg = [
            x.strip().upper() for x in str(cfg.layer_proportion_sensitivity_layers).split(",") if x.strip()
        ]

    rows: list[dict] = []
    for s in selected.itertuples(index=False):
        tree = str(s.tree)
        a, b = int(s.selected_window_start), int(s.selected_window_end)
        window_depths = sorted(
            int(d)
            for d in variants.loc[
                (variants["tree"].astype(str) == tree)
                & (variants["depth_filter"].astype(int) >= a)
                & (variants["depth_filter"].astype(int) <= b),
                "depth_filter",
            ].unique()
        )
        if not window_depths:
            continue

        sweep_points = sweep_layer_proportions(design.layer_proportions, relative_steps=steps, layers=layers_cfg)

        for point in sweep_points:
            layer_swept = point["layer_swept"]
            rel_step = float(point["relative_step"])
            props = point["layer_proportions"]

            base_row = {
                "dataset_label": cfg.dataset_label,
                "tree": tree,
                "selected_window_start": a,
                "selected_window_end": b,
                "layer_swept": layer_swept,
                "relative_step": rel_step,
            }

            if props is None:
                rows.append({
                    **base_row,
                    "layer_proportions": "",
                    "skipped_reason": point["skipped_reason"],
                    "n_window_depths_used": 0,
                    **{f"mean_{c}": np.nan for c in COUNT_COLUMNS},
                })
                continue

            active_layers = tuple(layer for layer in LAYER_COMPONENTS if props.get(layer, 0.0) > 0)
            active_components = ("E",) + active_layers
            mu_vals = {
                layer: (design.vaf_scale * props[layer] if layer in active_layers else None)
                for layer in LAYER_COMPONENTS
            }
            perturbed_mu = MuTriple(mu_vals["L1"], mu_vals["L2"], mu_vals["L3"])
            perturbed_design = ModelDesign(
                layer_specific_seq=design.layer_specific_seq,
                vaf_phasing=design.vaf_phasing,
                vaf_scale=design.vaf_scale,
                active_components=active_components,
                layer_proportions=props,
                derived_mu=perturbed_mu,
                organ_scale_by_tree=design.organ_scale_by_tree,   # organ ratios kept (0.4.1)
            )
            mu_grid_point = [perturbed_mu]

            tier_rows: list[dict] = []
            for depth in window_depths:
                df_depth = variants[
                    (variants["tree"].astype(str) == tree) & (variants["depth_filter"].astype(int) == depth)
                ].copy()
                if df_depth.empty:
                    continue
                _, _, tier_row, _, _ = fit_tree_depth_task(
                    (tree, depth), df_depth, genome, root, cfg, perturbed_design, mu_grid_point, rho_grid, lambda_grid
                )
                tier_rows.append(tier_row)

            rows.append({
                **base_row,
                "layer_proportions": format_layer_proportions(props),
                "skipped_reason": "",
                "n_window_depths_used": len(tier_rows),
                **{f"mean_{c}": (float(np.nanmean([r[f"sum_{c}"] for r in tier_rows])) if tier_rows else np.nan)
                   for c in COUNT_COLUMNS},
            })

    return pd.DataFrame(rows)


def build_augmented_variant_calls(
    variants: pd.DataFrame,
    variant_summary: pd.DataFrame,
    selected: pd.DataFrame,
    cfg: EstimateConfig,
) -> pd.DataFrame:
    """Per-SNP audit table: every candidate variant in the loosest depth tier
    (the smallest depth_filter present per tree) for a given tree, left-joined
    with its within-selected-window classification from `variant_summary`
    (the same per-variant summary that becomes 07_variant_calls_selected_window.csv).

    Every row gets an explicit `calling_status` rather than a variant simply
    being absent when it wasn't called:

      - not_evaluated_insufficient_depth: the variant never has any organ
        meeting the per-organ depth cutoff for ANY depth tier within the
        selected window, i.e. it never reaches the region where the model is
        stable/trusted. This is a QC/attrition signal, not a model output.
      - evaluated_never_positive: the variant reaches the window (has rows in
        the depth-filtered tables there) but alt_depth==0 for every organ at
        every window depth, so there was no positive observation to classify.
      - evaluated_unstable: positively observed and classified somewhere in
        the window, but the MAP call is not consistent across the window's
        depth cutoffs (fraction_MAP_fixed is neither ~0 nor ~1 at the
        cfg.stable_call_fraction threshold).
      - called_noise: consistently classified as NOT fixed (background/noise
        component) across the window.
      - called_fixed_L1 / _L2 / _L3: consistently classified as a fixed
        (somatic) mutation in that histogenic layer across the window.
      - called_fixed_ambiguous_layer: consistently classified as fixed, but
        no single layer reaches the stability threshold (rare; only possible
        when cfg.stable_call_fraction < 1.0).

    No call is ever derived from outside the selected window -- depth tiers
    below the window are exactly the ones with unstable, inflated counts of
    fixed calls, so a variant only observed there is reported as not evaluated,
    never given a status derived from that unstable regime.

    `organs_positive_in_window` names which organ(s)/sample(s) actually
    carried the alt allele (alt_depth > 0) at any depth tier within the
    window -- a semicolon-joined list, e.g. "leaf09;leaf21" -- so the table
    answers "detected in which organ", not just "detected in how many".
    """
    if selected.empty:
        return pd.DataFrame()

    variants = variants.copy()
    for c in ("chrom", "ref", "alt"):
        variants[c] = variants[c].astype(str)
    variants["pos"] = variants["pos"].astype(int)
    variants["tree"] = variants["tree"].astype(str)
    variants["depth_filter"] = variants["depth_filter"].astype(int)
    variants["alt_depth"] = variants["alt_depth"].astype(float)

    vs = variant_summary.copy()
    vs_indexed = None
    if not vs.empty:
        for c in ("chrom", "ref", "alt"):
            vs[c] = vs[c].astype(str)
        vs["pos"] = vs["pos"].astype(int)
        vs["tree"] = vs["tree"].astype(str)
        vs_indexed = vs.set_index(["tree", "chrom", "pos", "ref", "alt"])

    empty_rec = {
        "n_dp_cutoffs_classified": 0,
        "fraction_MAP_fixed": np.nan,
        "fraction_MAP_L1": np.nan,
        "fraction_MAP_L2": np.nan,
        "fraction_MAP_L3": np.nan,
        "mean_P_E": np.nan,
        "mean_P_L1": np.nan,
        "mean_P_L2": np.nan,
        "mean_P_L3": np.nan,
        "mean_P_fixed": np.nan,
        "stable_fixed_call": False,
        "stable_layer_call": "",
    }

    rows: list[dict] = []
    key_cols = ["chrom", "pos", "ref", "alt"]

    for s in selected.itertuples(index=False):
        tree = str(s.tree)
        a, b = int(s.selected_window_start), int(s.selected_window_end)
        tv = variants[variants["tree"] == tree]
        if tv.empty:
            continue

        min_depth = int(tv["depth_filter"].min())
        universe = tv.loc[tv["depth_filter"] == min_depth, key_cols].drop_duplicates()

        win = tv[tv["depth_filter"].between(a, b)]
        window_n_dp_cutoffs = int(win["depth_filter"].nunique())

        if not win.empty:
            presence = win.groupby(key_cols)["depth_filter"].nunique()
            pos_win = win[win["alt_depth"] > 0]
            n_organs_pos = pos_win.groupby(key_cols)["sample"].nunique()
            organs_pos_list = pos_win.groupby(key_cols)["sample"].apply(
                lambda s: ";".join(sorted(set(s.astype(str))))
            )
            max_alt = win.groupby(key_cols)["alt_depth"].max()
            max_vaf = win.groupby(key_cols)["VAF"].max() if "VAF" in win.columns else pd.Series(dtype=float)
        else:
            presence = pd.Series(dtype=int)
            n_organs_pos = pd.Series(dtype=int)
            organs_pos_list = pd.Series(dtype=object)
            max_alt = pd.Series(dtype=float)
            max_vaf = pd.Series(dtype=float)

        for row in universe.itertuples(index=False):
            key3 = (row.chrom, row.pos, row.ref, row.alt)
            n_present = int(presence.get(key3, 0))
            reaches_window = n_present > 0

            vkey = (tree,) + key3
            vs_row = vs_indexed.loc[vkey] if (vs_indexed is not None and vkey in vs_indexed.index) else None

            if vs_row is not None:
                frac_fixed = float(vs_row["fraction_MAP_fixed"])
                stable_fixed = bool(vs_row["stable_fixed_call"])
                stable_layer = str(vs_row["stable_layer_call"]) if pd.notna(vs_row["stable_layer_call"]) else ""
                if stable_fixed and stable_layer in {"L1", "L2", "L3"}:
                    status = f"called_fixed_{stable_layer}"
                elif stable_fixed:
                    status = "called_fixed_ambiguous_layer"
                elif frac_fixed <= (1.0 - cfg.stable_call_fraction):
                    status = "called_noise"
                else:
                    status = "evaluated_unstable"
                rec = {
                    "n_dp_cutoffs_classified": int(vs_row["n_dp_cutoffs"]),
                    "fraction_MAP_fixed": frac_fixed,
                    "fraction_MAP_L1": float(vs_row["fraction_MAP_L1"]),
                    "fraction_MAP_L2": float(vs_row["fraction_MAP_L2"]),
                    "fraction_MAP_L3": float(vs_row["fraction_MAP_L3"]),
                    "mean_P_E": float(vs_row["mean_P_E"]),
                    "mean_P_L1": float(vs_row["mean_P_L1"]),
                    "mean_P_L2": float(vs_row["mean_P_L2"]),
                    "mean_P_L3": float(vs_row["mean_P_L3"]),
                    "mean_P_fixed": float(vs_row["mean_P_fixed"]),
                    "stable_fixed_call": stable_fixed,
                    "stable_layer_call": stable_layer,
                }
            elif reaches_window:
                status = "evaluated_never_positive"
                rec = empty_rec
            else:
                status = "not_evaluated_insufficient_depth"
                rec = empty_rec

            rows.append(
                {
                    "dataset_label": cfg.dataset_label,
                    "tree": tree,
                    "chrom": row.chrom,
                    "pos": row.pos,
                    "ref": row.ref,
                    "alt": row.alt,
                    "selected_window_start": a,
                    "selected_window_end": b,
                    "window_n_dp_cutoffs": window_n_dp_cutoffs,
                    "calling_status": status,
                    "n_dp_cutoffs_present_in_window": n_present,
                    "n_organs_positive_in_window": int(n_organs_pos.get(key3, 0)),
                    "organs_positive_in_window": organs_pos_list.get(key3, ""),
                    "max_alt_depth_in_window": float(max_alt.get(key3, 0.0)),
                    "max_VAF_in_window": float(max_vaf.get(key3, 0.0)) if not max_vaf.empty else 0.0,
                    **rec,
                }
            )

    cols = [
        "dataset_label", "tree", "chrom", "pos", "ref", "alt",
        "selected_window_start", "selected_window_end", "window_n_dp_cutoffs",
        "calling_status", "n_dp_cutoffs_present_in_window", "n_organs_positive_in_window",
        "organs_positive_in_window", "max_alt_depth_in_window", "max_VAF_in_window",
        "n_dp_cutoffs_classified", "fraction_MAP_fixed", "fraction_MAP_L1", "fraction_MAP_L2",
        "fraction_MAP_L3", "mean_P_E", "mean_P_L1", "mean_P_L2", "mean_P_L3", "mean_P_fixed",
        "stable_fixed_call", "stable_layer_call",
    ]
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=cols)
    return out[cols]


def _fit_result_from_fit_row(row) -> FitResult:
    """Reconstruct a (classification-only) FitResult from a saved
    02_fit_parameters_by_tree_depth.csv row. No EM is re-run -- lambda_hat/pi/mu
    are just read back and used to re-evaluate the (closed-form) posterior, so
    this exactly reproduces the original classification. `grouped` is left
    empty since classify_organ_observations_for_fit never reads it (only
    fit_one_depth's own EM does).

    An inactive layer (not in active_components, e.g. L3 for a two-layer
    L1/L2-only bulk mixture such as the 1:9 pseudobulk design) has mu=None in
    the original fit -- classify_* never reads mu for a component outside
    active_components, so it's reconstructed here as 0.0 rather than left
    None/NaN, purely so MuTriple/float() round-trip cleanly.
    """
    def _mu_or_zero(v: object) -> float:
        return 0.0 if pd.isna(v) else float(v)

    osc = getattr(row, "organ_layer_scale", None)          # 0.4.1: organ-specific composition, JSON {sample: [s1, s2, s3]}
    osc = tuple(sorted((k, tuple(v)) for k, v in json.loads(osc).items())) if isinstance(osc, str) and osc else None
    return FitResult(
        mu=MuTriple(L1=_mu_or_zero(row.mu_L1), L2=_mu_or_zero(row.mu_L2), L3=_mu_or_zero(row.mu_L3), organ_scale=osc),
        lambda_hat=float(row.lambda_hat),
        pi=np.array([float(row.pi_E), float(row.pi_L1), float(row.pi_L2), float(row.pi_L3)]),
        pi_E_error=(0.0 if pd.isna(getattr(row, "pi_E_error", np.nan)) else float(row.pi_E_error)),
        loglik=float(row.loglik),
        n_positive_rows=int(row.n_positive_rows),
        n_unique_positive_variants=int(row.n_unique_positive_variants),
        grouped=pd.DataFrame(),
        active_components=tuple(str(row.active_components).split(";")),
        rho=float(row.layer_rho),
        n_iter=int(row.em_n_iter),
        converged=bool(row.em_converged),
    )


def build_organ_variant_calls(
    variants: pd.DataFrame,
    fit_df: pd.DataFrame,
    selected: pd.DataFrame,
    cfg: EstimateConfig,
) -> pd.DataFrame:
    """Per-(SNP, organ) audit table: the organ-level analogue of
    build_augmented_variant_calls. Rows are the same loosest-tier candidate-SNP
    universe per tree, crossed with every organ/sample that tree actually has;
    each cell's classification is re-derived (closed-form, no EM re-fit) from
    the saved per-depth lambda_hat/pi/mu in fit_df, applied to that organ's own
    reads only (never pooled across organs -- same single-observation posterior
    the per-organ counts (16, 17) use).

    `organ_calling_status` per (SNP, organ) reuses the exact same vocabulary as
    the SNP-level `calling_status` in build_augmented_variant_calls, just
    evaluated per organ instead of pooled across organs:
      - not_evaluated_insufficient_depth: this organ never has a row for this
        SNP at any window depth (its own per-organ depth never reached the
        window's cutoff there).
      - evaluated_never_positive: has window rows, but alt_depth==0 in all of
        them (negative).
      - evaluated_unstable: positively observed, but the MAP call isn't
        consistent across the window's depth cutoffs.
      - called_noise / called_fixed_L1 / called_fixed_L2 / called_fixed_L3 /
        called_fixed_ambiguous_layer: consistently classified.

    `mean_P_fixed` (plus mean_P_L1/L2/L3) is carried alongside the categorical
    status specifically so a "partial"/"almost fixed" signal in one organ isn't
    hidden just because its MAP call is "noise" or "unstable" -- two organs can
    both MAP to non-fixed while having very different posterior mass on fixed.
    """
    if selected.empty:
        return pd.DataFrame()

    variants = variants.copy()
    for c in ("chrom", "ref", "alt", "sample"):
        variants[c] = variants[c].astype(str)
    variants["pos"] = variants["pos"].astype(int)
    variants["tree"] = variants["tree"].astype(str)
    variants["depth_filter"] = variants["depth_filter"].astype(int)
    variants["alt_depth"] = variants["alt_depth"].astype(float)
    if "variant_id" not in variants.columns:
        # classify_organ_observations_for_fit requires this; read_variants() normally
        # already adds it (see io.variant_key_frame), but build it defensively here too.
        variants["variant_id"] = variant_key_frame(variants)

    fit_df = fit_df.copy()
    fit_df["tree"] = fit_df["tree"].astype(str)
    fit_df["depth_filter"] = fit_df["depth_filter"].astype(int)
    fit_by_key = {(r.tree, r.depth_filter): r for r in fit_df.itertuples(index=False)}

    exp_cache = ExponentialComponentCache(q_max=cfg.q_max, n_quad=cfg.n_quad, read_error=cfg.read_error)
    key_cols = ["chrom", "pos", "ref", "alt"]

    all_rows: list[dict] = []

    for s in selected.itertuples(index=False):
        tree = str(s.tree)
        a, b = int(s.selected_window_start), int(s.selected_window_end)
        tv = variants[variants["tree"] == tree]
        if tv.empty:
            continue

        organs = sorted(tv["sample"].unique())
        control_set = _control_organ_set(cfg.control_organs, tree)
        min_depth = int(tv["depth_filter"].min())
        universe = tv.loc[tv["depth_filter"] == min_depth, key_cols].drop_duplicates()

        win = tv[tv["depth_filter"].between(a, b)]
        window_depths = sorted(win["depth_filter"].unique())

        # Raw presence (any row, positive or not) per (variant, organ) within the window.
        presence = (
            win.groupby(key_cols + ["sample"])["depth_filter"].nunique()
            if not win.empty
            else pd.Series(dtype=int)
        )

        # Re-derive the per-organ posterior at each window depth from the saved fit
        # parameters (closed-form; no EM), then stack across depths.
        organ_calls_parts = []
        for d in window_depths:
            key = (tree, d)
            if key not in fit_by_key:
                continue
            df_depth = win[win["depth_filter"] == d]
            if df_depth.empty:
                continue
            fit = _fit_result_from_fit_row(fit_by_key[key])
            oc = classify_organ_observations_for_fit(df_depth, fit, exp_cache)
            if oc.empty:
                continue
            oc = oc.copy()
            oc["depth_filter"] = d
            organ_calls_parts.append(oc)

        oc_all = pd.concat(organ_calls_parts, ignore_index=True) if organ_calls_parts else pd.DataFrame()
        if not oc_all.empty:
            oc_all["chrom"] = oc_all["chrom"].astype(str)
            oc_all["pos"] = oc_all["pos"].astype(int)
            oc_all["ref"] = oc_all["ref"].astype(str)
            oc_all["alt"] = oc_all["alt"].astype(str)
            oc_all["sample"] = oc_all["sample"].astype(str)
            grouped_oc = {k: g for k, g in oc_all.groupby(key_cols + ["sample"], sort=False)}
        else:
            grouped_oc = {}

        for row in universe.itertuples(index=False):
            key4 = (row.chrom, row.pos, row.ref, row.alt)
            for organ in organs:
                key5 = key4 + (organ,)
                n_present = int(presence.get(key5, 0))
                g = grouped_oc.get(key5)

                if g is not None and not g.empty:
                    frac_fixed = float(g["obs_inferred_fixed_MAP"].mean())
                    is_l1 = (g["obs_inferred_component_MAP"] == "L1").mean()
                    is_l2 = (g["obs_inferred_component_MAP"] == "L2").mean()
                    is_l3 = (g["obs_inferred_component_MAP"] == "L3").mean()
                    layer_fracs = {"L1": float(is_l1), "L2": float(is_l2), "L3": float(is_l3)}
                    best_layer, best_layer_frac = max(layer_fracs.items(), key=lambda kv: kv[1])
                    stable_fixed = frac_fixed >= cfg.stable_call_fraction
                    stable_layer = best_layer if stable_fixed and best_layer_frac >= cfg.stable_call_fraction else ""
                    if stable_fixed and stable_layer:
                        status = f"called_fixed_{stable_layer}"
                    elif stable_fixed:
                        status = "called_fixed_ambiguous_layer"
                    elif frac_fixed <= (1.0 - cfg.stable_call_fraction):
                        status = "called_noise"
                    else:
                        status = "evaluated_unstable"
                    rec = {
                        "n_dp_cutoffs_positive": int(g["depth_filter"].nunique()),
                        "fraction_MAP_fixed": frac_fixed,
                        "fraction_MAP_L1": float(is_l1),
                        "fraction_MAP_L2": float(is_l2),
                        "fraction_MAP_L3": float(is_l3),
                        "mean_P_E": float(g["P_E"].mean()),
                        "mean_P_L1": float(g["P_L1"].mean()),
                        "mean_P_L2": float(g["P_L2"].mean()),
                        "mean_P_L3": float(g["P_L3"].mean()),
                        "mean_P_fixed": float(g["P_fixed"].mean()),
                        "stable_fixed_call": bool(stable_fixed),
                        "stable_layer_call": stable_layer,
                    }
                elif n_present > 0:
                    status = "evaluated_never_positive"
                    rec = {
                        "n_dp_cutoffs_positive": 0, "fraction_MAP_fixed": np.nan,
                        "fraction_MAP_L1": np.nan, "fraction_MAP_L2": np.nan, "fraction_MAP_L3": np.nan,
                        "mean_P_E": np.nan, "mean_P_L1": np.nan, "mean_P_L2": np.nan,
                        "mean_P_L3": np.nan, "mean_P_fixed": np.nan,
                        "stable_fixed_call": False, "stable_layer_call": "",
                    }
                else:
                    status = "not_evaluated_insufficient_depth"
                    rec = {
                        "n_dp_cutoffs_positive": 0, "fraction_MAP_fixed": np.nan,
                        "fraction_MAP_L1": np.nan, "fraction_MAP_L2": np.nan, "fraction_MAP_L3": np.nan,
                        "mean_P_E": np.nan, "mean_P_L1": np.nan, "mean_P_L2": np.nan,
                        "mean_P_L3": np.nan, "mean_P_fixed": np.nan,
                        "stable_fixed_call": False, "stable_layer_call": "",
                    }

                all_rows.append(
                    {
                        "dataset_label": cfg.dataset_label,
                        "tree": tree,
                        "chrom": row.chrom,
                        "pos": row.pos,
                        "ref": row.ref,
                        "alt": row.alt,
                        "sample": organ,
                        "selected_window_start": a,
                        "selected_window_end": b,
                        "n_dp_cutoffs_present_in_window": n_present,
                        "organ_calling_status": status,
                        "is_control_organ": organ in control_set,
                        **rec,
                    }
                )

    cols = [
        "dataset_label", "tree", "chrom", "pos", "ref", "alt", "sample",
        "selected_window_start", "selected_window_end", "n_dp_cutoffs_present_in_window",
        "organ_calling_status", "is_control_organ", "n_dp_cutoffs_positive", "fraction_MAP_fixed",
        "fraction_MAP_L1", "fraction_MAP_L2", "fraction_MAP_L3",
        "mean_P_E", "mean_P_L1", "mean_P_L2", "mean_P_L3", "mean_P_fixed",
        "stable_fixed_call", "stable_layer_call",
    ]
    out = pd.DataFrame(all_rows)
    if out.empty:
        return pd.DataFrame(columns=cols)
    return out[cols]


CANDIDATE_CALLING_STATUSES = {
    "called_fixed_L1",
    "called_fixed_L2",
    "called_fixed_L3",
    "called_fixed_ambiguous_layer",
    "evaluated_unstable",
}


def build_candidate_fixed_mutations(
    augmented: pd.DataFrame,
    organ_calls_window: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """The researcher-facing headline table: one row per candidate fixed SNP,
    answering in a single place (per tree, since each tree has its own organs):

      1. which SNPs show evidence for layer-fixation (rows are pre-filtered to
         calling_status in CANDIDATE_CALLING_STATUSES -- called_fixed_* plus
         evaluated_unstable, since a real fixed-layer signal doesn't require
         hitting the strict stable_call_fraction=1.0 bar at every organ; see
         the corresponding case in build_organ_variant_calls's tests
         for why evaluated_unstable can still be a genuine, layer-specific
         partial signal worth surfacing, not noise).
      2. most_likely_layer: the soft argmax over mean_P_L1/L2/L3, reported
         even when stable_layer_call is empty (i.e. for evaluated_unstable
         rows, where there's no single layer stable at every window depth but
         there's still a best guess).
      3 & 4. one status column and one P_fixed column per organ, pivoted from
         build_organ_variant_calls's long-format output, so the full per-organ
         calling status and posterior confidence for this SNP are visible
         without a separate join.

    Returns {tree: DataFrame} since organs differ by tree, so a single
    combined table would need one all-NaN column block per tree/organ pair
    that doesn't apply to a given row -- one table per tree is the workable
    "easy to read" shape. The full unfiltered data (every SNP, not just
    candidates) remains available in `augmented` (06) and
    `organ_calls_window` (08) for deeper statistical follow-up.
    """
    if augmented.empty:
        return {}

    candidates = augmented[augmented["calling_status"].isin(CANDIDATE_CALLING_STATUSES)].copy()
    if candidates.empty:
        return {t: pd.DataFrame() for t in sorted(augmented["tree"].unique())}

    def _most_likely_layer(row) -> str:
        vals = {"L1": row["mean_P_L1"], "L2": row["mean_P_L2"], "L3": row["mean_P_L3"]}
        if all(pd.isna(v) for v in vals.values()):
            return ""
        return max(vals, key=lambda k: (vals[k] if pd.notna(vals[k]) else -1.0))

    candidates["most_likely_layer"] = candidates.apply(_most_likely_layer, axis=1)
    candidates["variant"] = (
        candidates["chrom"] + ":" + candidates["pos"].astype(str) + candidates["ref"] + ">" + candidates["alt"]
    )

    key_cols = ["tree", "chrom", "pos", "ref", "alt"]
    oc = organ_calls_window.merge(candidates[key_cols], on=key_cols, how="inner").copy()

    fixed_flag = oc["organ_calling_status"].str.startswith("called_fixed")
    n_fixed = oc.assign(_fixed=fixed_flag).groupby(key_cols)["_fixed"].sum().rename("n_organs_called_fixed")
    organs_fixed = (
        oc[fixed_flag].groupby(key_cols)["sample"].apply(lambda s: ";".join(sorted(s))).rename("organs_called_fixed")
    )

    front_cols = [
        "dataset_label", "tree", "chrom", "pos", "ref", "alt", "variant",
        "calling_status", "most_likely_layer", "stable_layer_call",
        "mean_P_fixed", "mean_P_L1", "mean_P_L2", "mean_P_L3", "fraction_MAP_fixed",
        "n_organs_positive_in_window", "organs_positive_in_window",
    ]

    out_by_tree: dict[str, pd.DataFrame] = {}
    for tree, cg in candidates.groupby("tree", sort=True):
        cg = cg.merge(n_fixed, on=key_cols, how="left").merge(organs_fixed, on=key_cols, how="left")
        cg["n_organs_called_fixed"] = cg["n_organs_called_fixed"].fillna(0).astype(int)
        cg["organs_called_fixed"] = cg["organs_called_fixed"].fillna("")

        oc_tree = oc[oc["tree"] == tree]
        organs = sorted(oc_tree["sample"].unique())
        # dropna=False: pivot_table silently drops a column that's entirely NaN
        # (e.g. an organ with zero positive observations across every candidate
        # SNP for this tree), which would desync status_wide/pfixed_wide's organ
        # columns from each other and from `organs` below -- keep every organ
        # column present regardless of whether it's all-NaN for this subset.
        status_wide = oc_tree.pivot_table(
            index=key_cols, columns="sample", values="organ_calling_status", aggfunc="first", dropna=False
        )
        pfixed_wide = oc_tree.pivot_table(
            index=key_cols, columns="sample", values="mean_P_fixed", aggfunc="first", dropna=False
        )
        status_wide.columns = [f"{c}_status" for c in status_wide.columns]
        pfixed_wide.columns = [f"{c}_P_fixed" for c in pfixed_wide.columns]

        wide = cg.merge(status_wide.reset_index(), on=key_cols, how="left")
        wide = wide.merge(pfixed_wide.reset_index(), on=key_cols, how="left")

        organ_cols: list[str] = []
        for organ in organs:
            organ_cols.append(f"{organ}_status")
            organ_cols.append(f"{organ}_P_fixed")

        cols = front_cols + ["n_organs_called_fixed", "organs_called_fixed"] + organ_cols
        wide = wide.sort_values(["calling_status", "mean_P_fixed"], ascending=[True, False]).reset_index(drop=True)
        out_by_tree[str(tree)] = wide[cols]

    return out_by_tree


def metadata_dict(cfg: EstimateConfig, design: ModelDesign) -> dict[str, object]:
    return {
        "dataset_label": cfg.dataset_label,
        "layer.specific.seq": design.layer_specific_seq,
        "vaf.phasing": design.vaf_phasing,
        "vaf.scale": design.vaf_scale,
        "active.components": list(design.active_components),
        "layer.proportions": design.layer_proportions,
        "organ.layer.scale": ({t: {k: list(v) for k, v in m.items()} for t, m in design.organ_scale_by_tree.items()}
                              if design.organ_scale_by_tree else None),
        "derived.mu": design.derived_mu.as_dict(),
        "mu.source": cfg.mu_source,
        "mu.mode": cfg.mu_mode,
        "layer.dispersion.mode": cfg.layer_dispersion_mode,
        "layer.rho": cfg.layer_rho,
        "observation.read_error": cfg.read_error,
        "observation.error_component": "error-only background (q = 0) with its own weight, active because read_error > 0" if cfg.read_error > 0 else "none (read_error = 0)",
        "observation.preset": cfg.obs_preset,
        "observation.config_sha256": cfg.obs_config_sha256,
        "observation.source_version": cfg.obs_version,
        "output.spec.version": OUTPUT_SPEC_VERSION,
        "counts": (
            "per depth tier and organ (or replicate group): MAP counts, high-confidence counts (P_fixed >= 0.95) and "
            "posterior-weighted counts (sums of P_fixed, P_L1, P_L2, P_L3) of the single-observation posteriors"
        ),
        "depth.window.metric": "mean over non-control units of MAP_fixed_count / (2 * effective_genome_size * L_s)",
        "mu.grid.averaging": "likelihood-weighted (softmax of grid-point total loglik) when the (mu, rho) grid has more than one point",
        "mu.near.one.trigger": cfg.mu_near_one_trigger,
        "mu.near.one.offsets": cfg.mu_near_one_offsets or ",".join(str(o) for o in MU_NEAR_ONE_OFFSETS),
        "mu.near.one.note": (
            "A derived active-layer mu at/above mu.near.one.trigger (the phased single-layer "
            "sequencing boundary, mu -> 1.0) is automatically expanded into a softmax-blended "
            "grid of values approaching 1 from below, regardless of mu_mode; see "
            "MU_NEAR_ONE_TRIGGER in estimate.py."
        ),
        "q.max": cfg.q_max,
        "n.quad": cfg.n_quad,
        "lambda.grid.min": cfg.lambda_grid_min,
        "lambda.grid.max": cfg.lambda_grid_max,
        "lambda.grid.n": cfg.lambda_grid_n,
        "lambda.refine": cfg.lambda_refine,
        "em.n.starts": cfg.em_n_starts,
        "em.random.seed": cfg.em_random_seed,
        "diagnostics.resolvability.min_recall": cfg.resolvability_min_recall,
        "diagnostics.resolvability.background_quantile": cfg.resolvability_background_quantile,
        "screen.all_organs": cfg.all_organs_screen,
        "screen.all_organs.min_alt": cfg.all_organs_screen_min_alt if cfg.all_organs_screen else None,
        "layer.proportion.sensitivity.enabled": cfg.layer_proportion_sensitivity,
        "layer.proportion.sensitivity.steps": cfg.layer_proportion_sensitivity_steps or "-0.2,-0.1,0.0,0.1,0.2",
        "layer.proportion.sensitivity.layers": cfg.layer_proportion_sensitivity_layers or "all with nonzero proportion",
    } | ({"replicate.groups": cfg.replicate_groups} if cfg.replicate_groups else {})


UNIT_POSTERIOR_COLUMNS = [
    "dataset_label", "tree", "depth_filter", "unit", "is_control", "variant_id", "chrom", "pos", "ref", "alt",
    "P_E", "P_L1", "P_L2", "P_L3", "P_fixed", "obs_inferred_component_MAP", "obs_inferred_fixed_MAP",
    "obs_inferred_fixed_highconf_95",
]


def unit_posteriors_table(organ_calls_all: pd.DataFrame) -> pd.DataFrame:
    """Single-observation posteriors per depth tier, unit (organ or replicate group) and candidate variant (17)."""
    if organ_calls_all.empty:
        return pd.DataFrame(columns=UNIT_POSTERIOR_COLUMNS)
    out = organ_calls_all.rename(columns={"sample": "unit", "is_control_organ": "is_control"})
    return out[[c for c in UNIT_POSTERIOR_COLUMNS if c in out.columns]]


def run_estimate(cfg: EstimateConfig) -> dict[str, Path]:
    outdir = ensure_dir(cfg.outdir)
    design = build_model_design(cfg)
    metadata_path = outdir / "00_run_metadata.json"
    metadata_path.write_text(json.dumps(metadata_dict(cfg, design), indent=2, sort_keys=True))

    variants = read_variants(cfg.variants, cfg.dataset_label, tree_label=cfg.tree_label)
    root = read_root2tip(cfg.root2tip)
    genome = read_effective_genome(cfg.effective_genome_size, tree_label=cfg.tree_label)

    validation = input_validation_summary(variants, root, genome, cfg.dataset_label)
    validation_path = outdir / "01_input_validation_summary.csv"
    validation.to_csv(validation_path, index=False)

    screen_paths = {}
    if cfg.all_organs_screen:                         # 0.4.2: all-organs screen before the fit (vafsoma.screen)
        ctrl = {str(t): _control_organ_set(cfg.control_organs, str(t)) for t in variants["tree"].unique()}
        variants, screen_summary, screen_sites = all_organs_screen(variants, ctrl, min_alt=cfg.all_organs_screen_min_alt)
        screen_paths["all_organs_screen"] = outdir / "18_all_organs_screen.csv"
        screen_paths["all_organs_screened_sites"] = outdir / "18_all_organs_screened_sites.csv"
        screen_summary.to_csv(screen_paths["all_organs_screen"], index=False)
        screen_sites.to_csv(screen_paths["all_organs_screened_sites"], index=False)
        if variants.empty:
            raise ValueError("all-organs screen removed every variant")

    mu_grid = make_mu_grid(cfg, design)
    rho_grid = make_rho_grid(cfg)
    lambda_grid = parse_lambda_grid(cfg.lambda_grid, cfg.lambda_grid_min, cfg.lambda_grid_max, cfg.lambda_grid_n)

    groups = [
        ((str(tree), int(depth)), g.copy())
        for (tree, depth), g in variants.groupby(["tree", "depth_filter"], sort=True)
    ]
    if not groups:
        raise ValueError("No tree/depth groups found in variant input.")

    results = Parallel(n_jobs=cfg.n_jobs)(
        delayed(fit_tree_depth_task)(key, g, genome, root, cfg, design, mu_grid, rho_grid, lambda_grid)
        for key, g in groups
    )

    fit_rows = []
    calls_frames = []
    tier_rows = []
    organ_calls_frames = []
    unit_rows = []
    for fr, calls, tr, organ_calls, ur in results:
        fit_rows.extend(fr)
        calls_frames.append(calls)
        tier_rows.append(tr)
        organ_calls_frames.append(organ_calls)
        unit_rows.extend(ur)

    fit_df = pd.DataFrame(fit_rows)
    tier_df = pd.DataFrame(tier_rows).sort_values(["tree", "depth_filter"]).reset_index(drop=True)
    calls_all = pd.concat(calls_frames, ignore_index=True) if calls_frames else pd.DataFrame()
    organ_calls_all = pd.concat(organ_calls_frames, ignore_index=True) if organ_calls_frames else pd.DataFrame()
    unit_df = pd.DataFrame(unit_rows)

    fit_path = outdir / "02_fit_parameters_by_tree_depth.csv"
    tier_path = outdir / "03_fixed_counts_by_tree_depth.csv"
    fit_df.to_csv(fit_path, index=False)
    tier_df.to_csv(tier_path, index=False)
    unit_counts_path = outdir / "16_fixed_counts_by_unit_tier.csv"
    unit_df.to_csv(unit_counts_path, index=False)
    posteriors_path = outdir / "17_unit_posteriors_by_tier.csv.gz"
    unit_posteriors_table(organ_calls_all).to_csv(posteriors_path, index=False)

    window_scan = scan_windows(
        tier_df,
        dataset_label=cfg.dataset_label,
        min_start=cfg.min_window_start,
        min_cutoffs=cfg.min_window_cutoffs,
        max_cv=cfg.max_window_cv,
        max_adjacent_change=cfg.max_window_adjacent_change,
    )
    window_path = outdir / "04_dp_window_scan.csv"
    window_scan.to_csv(window_path, index=False)

    selected = select_windows(
        window_scan,
        tier_df,
        selected_window=parse_window(cfg.selected_dp_window),
        strategy=cfg.window_selection,
    )
    selected_summary, variant_summary = summarize_selected_windows(tier_df, selected, calls_all, cfg)

    selected_path = outdir / "05_selected_window_summary.csv"
    selected_summary.to_csv(selected_path, index=False)

    augmented = build_augmented_variant_calls(variants, variant_summary, selected, cfg)
    augmented_path = outdir / "06_variant_calls_augmented.csv"
    augmented.to_csv(augmented_path, index=False)

    variant_path = outdir / "07_variant_calls_selected_window.csv"
    if variant_summary.empty:
        variant_summary = pd.DataFrame(
            columns=[
                "dataset_label",
                "tree",
                "chrom",
                "pos",
                "ref",
                "alt",
                "n_dp_cutoffs",
                "fraction_MAP_fixed",
                "fraction_MAP_L1",
                "fraction_MAP_L2",
                "fraction_MAP_L3",
                "mean_P_E",
                "mean_P_L1",
                "mean_P_L2",
                "mean_P_L3",
                "mean_P_fixed",
                "stable_fixed_call",
                "stable_layer_call",
            ]
        )
    variant_summary.to_csv(variant_path, index=False)

    organ_calls_window = build_organ_variant_calls(variants, fit_df, selected, cfg)
    organ_calls_window_path = outdir / "08_organ_variant_calls_window.csv"
    organ_calls_window.to_csv(organ_calls_window_path, index=False)

    group_calls_path = None
    if cfg.replicate_groups:
        group_calls = summarize_group_calls_window(organ_calls_all, selected, cfg.stable_call_fraction)
        if not group_calls.empty:
            group_calls_path = outdir / "13_group_variant_calls_window.csv"
            group_calls.to_csv(group_calls_path, index=False)

    candidate_paths = {}
    candidates_by_tree = build_candidate_fixed_mutations(augmented, organ_calls_window)
    for tree, cdf in candidates_by_tree.items():
        p = outdir / f"09_candidate_fixed_mutations_{tree}.csv"
        cdf.to_csv(p, index=False)
        candidate_paths[tree] = p

    recurrence_sig_path = None
    recurrence_calls_paths: dict[str, Path] = {}
    if cfg.enable_recurrence_test:
        exp_cache_recurrence = ExponentialComponentCache(q_max=cfg.q_max, n_quad=cfg.n_quad, read_error=cfg.read_error)
        fit_by_key = {(str(r.tree), int(r.depth_filter)): r for r in fit_df.itertuples(index=False)}
        recurrence_by_tree: dict[str, RecurrenceReport] = {}
        for s in selected.itertuples(index=False):
            tree = str(s.tree)
            a, b = int(s.selected_window_start), int(s.selected_window_end)
            tv = variants[(variants["tree"] == tree) & (variants["depth_filter"].between(a, b))]
            if tv.empty:
                continue
            window_depths = sorted(tv["depth_filter"].unique())
            organs = sorted(tv["sample"].unique())
            control_set = _control_organ_set(cfg.control_organs, tree)
            if control_set:
                # See EstimateConfig.control_organs / vafsoma.recurrence.build_recurrence_report:
                # a control organ shows almost no alt-supporting reads at the candidate set BY
                # CONSTRUCTION, so it would appear as "always absent" in every organ_pattern and
                # only adds noise here, not signal.
                organs = [o for o in organs if o not in control_set]
            fit_row = fit_by_key.get((tree, window_depths[0])) if window_depths else None
            fit = _fit_result_from_fit_row(fit_row) if fit_row is not None else None
            report = build_recurrence_report(
                tree=tree,
                augmented=augmented,
                organ_calls_window=organ_calls_window,
                variants=variants,
                fit=fit,
                window_depths=window_depths,
                organs=organs,
                q_max=cfg.q_max,
                n_null_sites=cfg.recurrence_null_sites,
                presence_threshold=cfg.recurrence_presence_threshold,
                alpha=cfg.recurrence_alpha,
                seed=cfg.recurrence_seed,
                exp_cache=exp_cache_recurrence,
            )
            if report is not None:
                recurrence_by_tree[tree] = report

        if recurrence_by_tree:
            sig_frames = [r.significance_table for r in recurrence_by_tree.values()]
            sig_all = pd.concat(sig_frames, ignore_index=True)
            sig_all["subset"] = sig_all["subset"].apply(lambda s: ";".join(s))
            recurrence_sig_path = outdir / "11_recurrence_significance.csv"
            sig_all.to_csv(recurrence_sig_path, index=False)

            for tree, report in recurrence_by_tree.items():
                p = outdir / f"12_recurrence_boosted_calls_{tree}.csv"
                out_df = report.boosted_calls.copy()
                out_df["organ_pattern"] = out_df["organ_pattern"].apply(
                    lambda pat: ";".join(pat) if isinstance(pat, tuple) else ""
                )
                out_df.to_csv(p, index=False)
                recurrence_calls_paths[tree] = p

    sensitivity_path = None
    if cfg.layer_proportion_sensitivity:
        sensitivity = layer_proportion_sensitivity_sweep(
            variants, genome, root, selected, cfg, design, rho_grid, lambda_grid
        )
        sensitivity_path = outdir / "10_layer_proportion_sensitivity.csv"
        sensitivity.to_csv(sensitivity_path, index=False)

    if cfg.write_plots:
        write_count_plots(tier_df, selected, outdir)

    resolvability_path = resolvability_depth_path = fitcheck_path = fitcheck_summary_path = None
    if cfg.write_diagnostics:
        # 0.1.6: output-only diagnostics computed from the saved fit parameters (see vafsoma.diagnostics); 0.1.7: group-aware recall.
        group_maps = {}
        if cfg.replicate_groups:                      # 0.1.7: group-aware resolvability for replicate trees
            for tree_id, tv_all in variants.groupby("tree"):
                tree_id = str(tree_id)
                if tree_has_replicates(cfg.replicate_groups, tree_id):
                    smp = sorted(tv_all["sample"].astype(str).unique())
                    group_maps[tree_id] = sample_to_group(cfg.replicate_groups, tree_id, smp,
                                                          _control_organ_set(cfg.control_organs, tree_id))
        rec_df, rec_summary, chk_df = build_diagnostics(
            fit_df, variants, selected, q_max=cfg.q_max, n_quad=cfg.n_quad, read_error=cfg.read_error, fit_from_row=_fit_result_from_fit_row,
            min_recall=cfg.resolvability_min_recall, max_abs_residual=cfg.fit_check_max_abs_residual,
            group_maps=group_maps, background_quantile=cfg.resolvability_background_quantile,
        )
        resolvability_depth_path = outdir / "14_layer_resolvability_by_depth.csv"
        rec_df.to_csv(resolvability_depth_path, index=False)
        resolvability_path = outdir / "14_layer_resolvability.csv"
        rec_summary.to_csv(resolvability_path, index=False)
        fitcheck_path = outdir / "15_vaf_fit_check.csv"
        chk_df.to_csv(fitcheck_path, index=False)
        fitcheck_summary_path = outdir / "15_vaf_fit_check_summary.csv"
        fit_summary = summarize_fit_check(chk_df)
        fit_summary.to_csv(fitcheck_summary_path, index=False)
        # Carry the diagnostics into the window summary: counts of layers with `{layer}_resolvable` False should not be
        # interpreted.
        selected_summary = annotate_summary_with_diagnostics(selected_summary, rec_summary, fit_summary)
        selected_summary.to_csv(selected_path, index=False)

    outputs = {
        "metadata": metadata_path,
        "validation": validation_path,
        "fit_parameters": fit_path,
        "fixed_counts_by_tier": tier_path,
        "fixed_counts_by_unit_tier": unit_counts_path,
        "unit_posteriors_by_tier": posteriors_path,
        "dp_window_scan": window_path,
        "selected_window_summary": selected_path,
        "variant_calls_augmented": augmented_path,
        "variant_calls_selected_window": variant_path,
        "organ_variant_calls_window": organ_calls_window_path,
        "candidate_fixed_mutations": candidate_paths,
        "outdir": outdir,
    }
    if sensitivity_path is not None:
        outputs["layer_proportion_sensitivity"] = sensitivity_path
    if group_calls_path is not None:
        outputs["group_variant_calls"] = group_calls_path
    if recurrence_sig_path is not None:
        outputs["recurrence_significance"] = recurrence_sig_path
    if recurrence_calls_paths:
        outputs["recurrence_boosted_calls"] = recurrence_calls_paths
    if resolvability_path is not None:
        outputs["layer_resolvability"] = resolvability_path
        outputs["layer_resolvability_by_depth"] = resolvability_depth_path
        outputs["vaf_fit_check"] = fitcheck_path
        outputs["vaf_fit_check_summary"] = fitcheck_summary_path
    outputs.update(screen_paths)
    return outputs
