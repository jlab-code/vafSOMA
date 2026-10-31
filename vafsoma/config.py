"""vafSOMA run configuration (JSON).

A configuration describes one complete analysis: input tables, the layer model, depth-window selection, fitting
and output options. Relative paths are read relative to the configuration file (paths starting with ``vafSOMA_inputs/`` or
``vafSOMA_output/``: relative to the repository root, as in simSOMA). Results are written to
``<outdir>/<experiment>/``. ``vafsoma template run`` prints a complete template; keys starting with ``_`` are
comments and are ignored. Configuration files of vafSOMA <= 0.1.7 (TOML, ``vafsoma_run_config.toml``) are still
read and translated (transition support; to be removed in a later release).
"""
from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from typing import Any

from .estimate import EstimateConfig
from .replicates import read_replicate_groups

LAYERS = ("L1", "L2", "L3")

TEMPLATE: dict[str, Any] = {
    "_about": ("vafSOMA run configuration. Relative paths are relative to this file. Keys starting with '_' are "
               "comments. Results go to <outdir>/<experiment>/. See vafSOMA_docs/configuration.md."),
    "experiment": "my_analysis",
    "outdir": "vafSOMA_output",
    "input": {
        "_about": ("variants: one VAF table per minimum-depth cutoff (directory, .zip or single table with a "
                   "depth_filter column); effective_genome_size: table with tree, depth_filter, "
                   "effective_genome_size; root2tip: table with sample, root2tip_length (length or age of each organ, "
                   "measured from its divergence from the matched normal; used to choose the depth window and "
                   "passed to the outputs). tree: label used when the tables carry no tree column. "
                   "replicate_groups: table (tree, sample, group) or {tree: {sample: group}}. control_organs: "
                   "{tree: organ or [organs]} used as matched normal upstream (flagged in the outputs; not used "
                   "to choose the depth window)."),
        "variants": "vaf_tables.zip",
        "effective_genome_size": "effective_genome_size.tsv",
        "root2tip": "root2tip.tsv",
        "tree": None,
        "replicate_groups": None,
        "control_organs": {},
    },
    "model": {
        "_about": ("layer_specific_seq false: bulk organs with layer_proportions; true: layer-enriched sample with "
                   "active_layer (and its purity active_layer_proportion). vaf_phasing 'unphased' (collapsed "
                   "diploid; mu = c/2) or 'phased' (mu = c). Template layer proportions: mean of published leaf "
                   "compositions (as in the simSOMA layers template)."),
        "layer_specific_seq": False,
        "vaf_phasing": "unphased",
        "layer_proportions": {"L1": 0.13, "L2": 0.84, "L3": 0.03},
        "_about_organ_layer_proportions": ("0.4.1, optional: organ-specific layer proportions for bulk organs, a table "
                                           "(columns sample, L1, L2, L3, optional tree) or {sample: {L1, L2, L3}}; "
                                           "organs not listed use layer_proportions"),
        "organ_layer_proportions": None,
        "active_layer": None,
        "active_layer_proportion": None,
        "mu": {"source": "derived", "mode": "fixed", "manual": {"L1": 0.10, "L2": 0.36, "L3": 0.04},
               "grid": {"L1": None, "L2": None, "L3": None},
               "near_one_trigger": 0.999, "near_one_offsets": None},
        "layer_dispersion": {"mode": "fixed", "rho": 0.0, "rho_grid": None},
    },
    "observation": {
        "_about": ("0.4.0: observation model of the reads, with parameters calibrated for the data set. "
                   "read_error: symmetric per-read sequencing error e (all components, including the zero counts; 0 "
                   "reproduces 0.3.0). read_concentration: concentration k of the allele-fraction beta distribution; "
                   "sets the layer overdispersion rho = 1 / (k + 1) (model.layer_dispersion must then be fixed with rho "
                   "0 or the same value). With read_error > 0 the background also has an error-only part (q = 0) with "
                   "its own weight. preset, config_sha256, source_version: optional provenance of the calibration, recorded in "
                   "the outputs."),
        "read_error": 0.0,
        "read_concentration": None,
        "preset": None,
        "config_sha256": None,
        "source_version": None,
    },
    "depth_window": {
        "_about": "selected: fixed window 'start:end' (e.g. '28:36'); null selects a stable window automatically.",
        "selected": None,
        "selection": "auto",
        "min_start": 12,
        "min_cutoffs": 4,
        "max_cv": 0.15,
        "max_adjacent_change": 0.25,
    },
    "background": {"q_max": 0.5, "n_quad": 96, "lambda_grid": None, "lambda_grid_min": 1.0,
                   "lambda_grid_max": 500.0, "lambda_grid_n": 15, "lambda_refine": True},
    "em": {"n_starts": 16, "random_seed": 1, "max_iter": 1000, "tol": 1e-8},
    "calls": {"stable_call_fraction": 1.0},
    "diagnostics": {"_about": ("resolvability_background_quantile (0.4.2): a layer is also reported unresolvable when its "
                               "expected VAF lies below this quantile of the fitted background (null: recall only, as in "
                               "<= 0.4.1)."),
                    "enabled": True, "resolvability_min_recall": 0.5, "resolvability_background_quantile": 0.95,
                    "fit_check_max_abs_residual": 4.0},
    "screen": {"_about": ("0.4.2, opt-in. all_organs: remove, before the fit, candidates with >= min_alt alternative reads "
                          "in every non-control sample of a tree at its lowest depth tier. Valid only when no lineage "
                          "after the divergence from the control organ is shared by all sampled organs; needs "
                          "input.control_organs and >= 3 other samples per tree."),
               "all_organs": False, "min_alt": 1},
    "sensitivity": {"layer_proportions": False, "steps": None, "layers": None},
    "recurrence_test": {"enabled": False, "null_sites": 20000, "presence_threshold": 0.5, "alpha": 0.05, "seed": 0},
    "compute": {"n_jobs": 1, "plots": True},
}


class ConfigError(ValueError):
    pass


def template_text() -> str:
    return json.dumps(TEMPLATE, indent=2) + "\n"


def _strip_comments(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _strip_comments(v) for k, v in x.items() if not str(k).startswith("_")}
    return x


def _merge(base: dict, user: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in user.items():
        here = f"{path}.{k}" if path else k
        if k not in out:
            raise ConfigError(f"unknown configuration key: {here}")
        if isinstance(out[k], dict) and isinstance(v, dict) and k not in ("control_organs", "replicate_groups",
                                                                          "layer_proportions", "manual",
                                                                          "organ_layer_proportions"):
            out[k] = _merge(out[k], v, here)
        else:
            out[k] = v
    return out


def _from_legacy_toml(raw: dict) -> dict:
    """Translate a vafSOMA <= 0.1.7 TOML configuration (project/input/model/mu/dp_window/...)."""
    proj, inp, mod, mu = raw.get("project", {}), raw.get("input", {}), raw.get("model", {}), raw.get("mu", {})
    name = proj.get("name", "analysis")
    up = inp.get("upstream_dir", "projects/{project}/upstream").format(project=name)
    res = inp.get("results_dir", "results/{project}").format(project=name)
    dp, bg, em = raw.get("dp_window", {}), raw.get("background", {}), raw.get("em", {})
    return {
        "experiment": name, "outdir": str(Path(res).parent),
        "input": {"variants": f"{up}/{inp.get('vaf_dp_input', 'vaf_dp_input.zip')}",
                  "effective_genome_size": f"{up}/{inp.get('effective_genome_size_input', 'effective_genome_size_input.zip')}",
                  "root2tip": f"{up}/{inp.get('root2tip_input', 'root2tip_input.txt')}",
                  "control_organs": raw.get("control_organs", {})},
        "model": {"layer_specific_seq": bool(mod.get("layer_specific_seq", False)),
                  "vaf_phasing": mod.get("vaf_phasing", "unphased"),
                  "layer_proportions": mod.get("layer_proportions", {"L1": 0.20, "L2": 0.72, "L3": 0.08}),  # 0.1.x default
                  "active_layer": mod.get("active_layer") or None,
                  "active_layer_proportion": mod.get("active_layer_proportion") if mod.get("active_layer") else None,
                  "mu": {"source": mu.get("source", "derived"), "mode": mu.get("mode", "fixed"),
                         "manual": mu.get("manual", TEMPLATE["model"]["mu"]["manual"]),
                         "grid": {L: mu.get(f"grid_{L}") for L in LAYERS}}},
        "depth_window": {"selected": dp.get("selected") or None, "selection": dp.get("selection", "auto"),
                         "min_start": dp.get("min_window_start", 12), "min_cutoffs": dp.get("min_window_cutoffs", 4),
                         "max_cv": dp.get("max_window_cv", 0.15),
                         "max_adjacent_change": dp.get("max_window_adjacent_change", 0.25)},
        "background": {"q_max": bg.get("q_max", 0.5), "n_quad": bg.get("n_quad", 96),
                       "lambda_grid": bg.get("lambda_grid") or None, "lambda_grid_min": bg.get("lambda_grid_min", 1.0),
                       "lambda_grid_max": bg.get("lambda_grid_max", 500.0), "lambda_grid_n": bg.get("lambda_grid_n", 15)},
        "em": {"n_starts": em.get("n_starts", 16), "random_seed": em.get("random_seed", 1),
               "max_iter": em.get("max_iter", 1000), "tol": em.get("tol", 1e-8)},
        "calls": {"stable_call_fraction": raw.get("variant_calls", {}).get("stable_call_fraction", 1.0)},
        "compute": {"n_jobs": raw.get("compute", {}).get("n_jobs", 1), "plots": raw.get("compute", {}).get("plots", True)},
    }


def load_config(path: str | Path) -> tuple[dict, Path]:
    """Read a JSON (or legacy TOML) configuration; returns (complete config, base directory for relative paths)."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise ConfigError(f"configuration file not found: {path}")
    if path.suffix.lower() == ".toml":
        try:
            import tomllib
        except ModuleNotFoundError as exc:          # Python 3.10
            raise ConfigError("legacy TOML configurations need Python >= 3.11 (tomllib); convert to JSON") from exc
        with open(path, "rb") as fh:
            user = _from_legacy_toml(tomllib.load(fh))
        base = Path.cwd()                           # legacy configs were relative to the repository root
    else:
        try:
            user = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path}: invalid JSON ({exc})") from exc
        base = path.parent
    return _merge(_strip_comments(TEMPLATE), _drop_removed_keys(_strip_comments(user))), base


# Keys of vafSOMA <= 0.2.3 that belonged to rate estimation, removed in 0.3.0. They are ignored with a warning, so
# that older configuration files still run.
REMOVED_KEYS = (("bootstrap",), ("model", "ploidy_factor"))
# Keys renamed in 0.4.3 (old path -> new last key); the old name is still accepted.
RENAMED_KEYS = ((("observation", "plantsoma_obs_version"), "source_version"),)


def _drop_removed_keys(user: dict) -> dict:
    for path, new in RENAMED_KEYS:
        d = user
        for k in path[:-1]:
            d = d.get(k, {}) if isinstance(d, dict) else {}
        if isinstance(d, dict) and path[-1] in d:
            d.setdefault(new, d.pop(path[-1]))
    for path in REMOVED_KEYS:
        d = user
        for k in path[:-1]:
            d = d.get(k, {}) if isinstance(d, dict) else {}
        if isinstance(d, dict) and path[-1] in d:
            d.pop(path[-1])
            warnings.warn(f"configuration key {'.'.join(path)} was removed in vafSOMA 0.3.0 (vafSOMA no longer "
                          "estimates rates) and is ignored", stacklevel=3)
    return user


def validate(cfg: dict, base: Path) -> list[str]:
    """Return a list of problems (empty if the configuration is usable)."""
    err = []
    m = cfg["model"]
    if m["vaf_phasing"] not in ("phased", "unphased"):
        err.append("model.vaf_phasing must be 'phased' or 'unphased'")
    if m["layer_specific_seq"]:
        if m["active_layer"] not in LAYERS:
            err.append("model.active_layer must be L1, L2 or L3 when model.layer_specific_seq is true")
    else:
        if m["active_layer"]:
            err.append("model.active_layer can only be set when model.layer_specific_seq is true")
        lp = m["layer_proportions"] or {}
        if set(lp) - set(LAYERS) or not lp:
            err.append("model.layer_proportions must give L1, L2 and/or L3")
        elif abs(sum(float(v) for v in lp.values()) - 1) > 1e-6:
            err.append(f"model.layer_proportions must sum to 1 (got {sum(float(v) for v in lp.values()):.4f})")
    olp = m.get("organ_layer_proportions")
    if olp is not None:
        if m["layer_specific_seq"]:
            err.append("model.organ_layer_proportions can only be used with layer_specific_seq false (bulk organs)")
        elif isinstance(olp, str):
            if not _resolve(olp, base).exists():
                err.append(f"model.organ_layer_proportions not found: {_resolve(olp, base)}")
        elif not isinstance(olp, dict) or not olp:
            err.append("model.organ_layer_proportions must be a table path or {sample: {L1, L2, L3}}")
    if m["mu"]["source"] not in ("derived", "manual") or m["mu"]["mode"] not in ("fixed", "grid"):
        err.append("model.mu.source must be 'derived' or 'manual'; model.mu.mode 'fixed' or 'grid'")
    ob = cfg["observation"]
    try:
        if not (0.0 <= float(ob["read_error"]) < 0.5):
            err.append("observation.read_error must be in [0, 0.5)")
    except (TypeError, ValueError):
        err.append("observation.read_error must be a number")
    if ob["read_concentration"] is not None:
        try:
            k = float(ob["read_concentration"])
            if k <= 0:
                err.append("observation.read_concentration must be > 0")
            else:
                ld = m["layer_dispersion"]
                rho_k = 1.0 / (k + 1.0)
                if ld["mode"] != "fixed" or (float(ld["rho"] or 0.0) not in (0.0,) and abs(float(ld["rho"]) - rho_k) > 1e-12):
                    err.append("observation.read_concentration sets the layer rho; model.layer_dispersion must be "
                               "fixed with rho 0 or 1 / (read_concentration + 1)")
        except (TypeError, ValueError):
            err.append("observation.read_concentration must be a number or null")
    sel = cfg["depth_window"]["selected"]
    if sel is not None and not (isinstance(sel, str) and ":" in sel) and not (isinstance(sel, list) and len(sel) == 2):
        err.append("depth_window.selected must be null, 'start:end' or [start, end]")
    for key in ("variants", "effective_genome_size", "root2tip"):
        p = cfg["input"][key]
        if not p:
            err.append(f"input.{key} is required")
        elif not _resolve(p, base).exists():
            err.append(f"input.{key} not found: {_resolve(p, base)}")
    rg = cfg["input"]["replicate_groups"]
    if isinstance(rg, str) and not _resolve(rg, base).exists():
        err.append(f"input.replicate_groups not found: {_resolve(rg, base)}")
    return err


PROJECT_PREFIXES = ("vafSOMA_inputs", "vafSOMA_output")


def _project_root(base: Path) -> Path:
    """Repository root for paths starting with vafSOMA_inputs/ or vafSOMA_output/ (as in simSOMA)."""
    for parent in (base, *base.parents):
        if (parent / "vafSOMA_configs").is_dir() and (parent / "vafsoma").is_dir():
            return parent
    return base


def _resolve(p: str | Path, base: Path) -> Path:
    """Relative paths are relative to the configuration file; paths starting with vafSOMA_inputs/ or
    vafSOMA_output/ are relative to the repository root (same rule as simSOMA)."""
    p = Path(p).expanduser()
    if p.is_absolute():
        return p
    root = _project_root(base) if p.parts and p.parts[0] in PROJECT_PREFIXES else base
    return (root / p).resolve()


def _csv(x) -> str | None:
    if x is None or x == []:
        return None
    return ",".join(str(v) for v in x) if isinstance(x, (list, tuple)) else str(x)


def to_estimate_config(cfg: dict, base: Path, outdir: str | Path | None = None) -> EstimateConfig:
    """Translate a (validated) configuration into the estimator's EstimateConfig."""
    problems = validate(cfg, base)
    if problems:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(problems))
    inp, m, dw, bg = cfg["input"], cfg["model"], cfg["depth_window"], cfg["background"]
    out = Path(outdir) if outdir else _resolve(cfg["outdir"], base) / cfg["experiment"]
    rg = inp["replicate_groups"]
    if isinstance(rg, str):
        rg = read_replicate_groups(_resolve(rg, base))
    elif isinstance(rg, dict):
        rg = {str(t): {str(s): str(g) for s, g in v.items()} for t, v in rg.items()} or None
    sel = dw["selected"]
    if isinstance(sel, list):
        sel = f"{int(sel[0])}:{int(sel[1])}"
    lp = m["layer_proportions"] if not m["layer_specific_seq"] else None
    return EstimateConfig(
        variants=str(_resolve(inp["variants"], base)),
        effective_genome_size=str(_resolve(inp["effective_genome_size"], base)),
        root2tip=str(_resolve(inp["root2tip"], base)),
        dataset_label=cfg["experiment"], outdir=str(out), tree_label=inp["tree"],
        layer_specific_seq="yes" if m["layer_specific_seq"] else "no", vaf_phasing=m["vaf_phasing"],
        layer_proportions=",".join(f"{k}={v}" for k, v in lp.items()) if lp else None,
        organ_layer_proportions=(str(_resolve(m["organ_layer_proportions"], base))
                                 if isinstance(m.get("organ_layer_proportions"), str) else m.get("organ_layer_proportions")),
        active_layer=m["active_layer"], active_layer_proportion=m["active_layer_proportion"],
        mu_source=m["mu"]["source"], mu_mode=m["mu"]["mode"],
        mu_L1=m["mu"]["manual"].get("L1", 0.10), mu_L2=m["mu"]["manual"].get("L2", 0.36),
        mu_L3=m["mu"]["manual"].get("L3", 0.04),
        mu_L1_grid=_csv(m["mu"]["grid"].get("L1")), mu_L2_grid=_csv(m["mu"]["grid"].get("L2")),
        mu_L3_grid=_csv(m["mu"]["grid"].get("L3")),
        mu_near_one_trigger=m["mu"]["near_one_trigger"], mu_near_one_offsets=_csv(m["mu"]["near_one_offsets"]),
        layer_dispersion_mode=m["layer_dispersion"]["mode"],
        layer_rho=(1.0 / (float(cfg["observation"]["read_concentration"]) + 1.0)
                   if cfg["observation"]["read_concentration"] is not None else m["layer_dispersion"]["rho"]),
        read_error=float(cfg["observation"]["read_error"]),
        obs_preset=cfg["observation"]["preset"], obs_config_sha256=cfg["observation"]["config_sha256"],
        obs_version=cfg["observation"]["source_version"],
        layer_rho_grid=_csv(m["layer_dispersion"]["rho_grid"]),
        q_max=bg["q_max"], n_quad=bg["n_quad"], lambda_grid=_csv(bg["lambda_grid"]),
        lambda_grid_min=bg["lambda_grid_min"], lambda_grid_max=bg["lambda_grid_max"],
        lambda_grid_n=bg["lambda_grid_n"], lambda_refine=bool(bg["lambda_refine"]),
        selected_dp_window=sel, window_selection=dw["selection"],
        min_window_start=dw["min_start"], min_window_cutoffs=dw["min_cutoffs"], max_window_cv=dw["max_cv"],
        max_window_adjacent_change=dw["max_adjacent_change"],
        n_jobs=cfg["compute"]["n_jobs"],
        em_max_iter=cfg["em"]["max_iter"], em_tol=cfg["em"]["tol"], em_n_starts=cfg["em"]["n_starts"],
        em_random_seed=cfg["em"]["random_seed"], stable_call_fraction=cfg["calls"]["stable_call_fraction"],
        write_plots=bool(cfg["compute"]["plots"]), write_diagnostics=bool(cfg["diagnostics"]["enabled"]),
        resolvability_min_recall=cfg["diagnostics"]["resolvability_min_recall"],
        resolvability_background_quantile=(None if cfg["diagnostics"]["resolvability_background_quantile"] is None
                                           else float(cfg["diagnostics"]["resolvability_background_quantile"])),
        all_organs_screen=bool(cfg["screen"]["all_organs"]), all_organs_screen_min_alt=int(cfg["screen"]["min_alt"]),
        fit_check_max_abs_residual=cfg["diagnostics"]["fit_check_max_abs_residual"],
        control_organs={str(k): v for k, v in (inp["control_organs"] or {}).items()},
        replicate_groups=rg,
        layer_proportion_sensitivity=bool(cfg["sensitivity"]["layer_proportions"]),
        layer_proportion_sensitivity_steps=_csv(cfg["sensitivity"]["steps"]),
        layer_proportion_sensitivity_layers=_csv(cfg["sensitivity"]["layers"]),
        enable_recurrence_test=bool(cfg["recurrence_test"]["enabled"]),
        recurrence_null_sites=cfg["recurrence_test"]["null_sites"],
        recurrence_presence_threshold=cfg["recurrence_test"]["presence_threshold"],
        recurrence_alpha=cfg["recurrence_test"]["alpha"], recurrence_seed=cfg["recurrence_test"]["seed"],
    )
