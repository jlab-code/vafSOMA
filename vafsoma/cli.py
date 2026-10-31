"""vafSOMA command-line interface.

    vafsoma run CONFIG [--outdir DIR]     fit the model from a JSON configuration (results: <outdir>/<experiment>/)
    vafsoma check CONFIG                  validate a configuration and its input files, print the resolved settings
    vafsoma template run                  print a complete configuration template
    vafsoma version                       print versions
    vafsoma estimate --variants ... ...   flag-based interface of vafSOMA <= 0.1.7 (unchanged)
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from . import __version__
from .estimate import EstimateConfig, run_estimate
from .utils import parse_control_organs
from .replicates import read_replicate_groups


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vafsoma",
        description="vafSOMA: fixed somatic mutations and their layers from VAF spectra of layered plant tissue.",
    )
    parser.add_argument("--version", action="version", version=f"vafsoma {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="fit the model from a JSON configuration")
    r.add_argument("config", type=Path)
    r.add_argument("--outdir", default=None, help="write results here instead of <outdir>/<experiment>/")
    c = sub.add_parser("check", help="validate a configuration and its input files")
    c.add_argument("config", type=Path)
    tp = sub.add_parser("template", help="print a configuration template")
    tp.add_argument("kind", choices=["run"])
    sub.add_parser("version", help="print versions")

    est = sub.add_parser("estimate", help="flag-based interface of vafSOMA <= 0.1.7 (prefer `run`)")
    est.add_argument("--variants", required=True, help="Variant table, directory, or .zip of tables.")
    est.add_argument("--effective-genome-size", required=True, help="Effective genome-size table, directory, or .zip.")
    est.add_argument("--root2tip", required=True, help="Root-to-tip length table.")
    est.add_argument("--dataset-label", default="dataset", help="User-defined project/output label recorded in result tables.")
    est.add_argument("--outdir", required=True, help="Output directory.")

    est.add_argument("--layer-specific-seq", choices=["yes", "no"], default="no", help="Whether the input comes from layer-specific sequencing.")
    est.add_argument("--vaf-phasing", choices=["phased", "unphased"], default="unphased", help="Whether VAFs are haplotype-specific/phased or collapsed diploid/unphased.")
    est.add_argument("--layer-proportions", default=None, help="Bulk layer proportions, e.g. L1=0.20,L2=0.72,L3=0.08. Only valid with --layer-specific-seq no. Default: L1=0.20,L2=0.72,L3=0.08.")
    est.add_argument("--active-layer", choices=["L1", "L2", "L3"], default=None, help="Sequenced/active layer for layer-specific sequencing. Required with --layer-specific-seq yes and forbidden otherwise.")
    est.add_argument("--active-layer-proportion", type=float, default=None, help="Purity/contribution of active layer for layer-specific sequencing. Default 1.0. Only valid with --layer-specific-seq yes.")
    est.add_argument("--mu-source", choices=["derived", "manual"], default="derived", help="Derive mu from layer proportions and VAF phasing, or use manual --mu-L* values.")

    est.add_argument("--mu-mode", choices=["fixed", "grid"], default="fixed")
    est.add_argument("--mu-L1", type=float, default=0.10)
    est.add_argument("--mu-L2", type=float, default=0.36)
    est.add_argument("--mu-L3", type=float, default=0.04)
    est.add_argument("--mu-L1-grid", default=None, help="Comma-separated grid, default 0.08,0.10,0.12")
    est.add_argument("--mu-L2-grid", default=None, help="Comma-separated grid, default 0.32,0.36,0.40")
    est.add_argument("--mu-L3-grid", default=None, help="Comma-separated grid, default 0.03,0.04,0.05")
    est.add_argument(
        "--mu-near-one-trigger",
        type=float,
        default=0.999,
        help=(
            "A derived active-layer mu at/above this value (the phased single-layer sequencing "
            "boundary, mu -> 1.0) is automatically expanded into a softmax-blended grid "
            "approaching 1 from below, regardless of --mu-mode -- see --mu-near-one-offsets."
        ),
    )
    est.add_argument(
        "--mu-near-one-offsets",
        default=None,
        help="Comma-separated offsets below 1.0 for the near-boundary mu grid. Default 0.01,0.003,0.001,0.0003,0.0001,0.00003,0.00001",
    )

    est.add_argument(
        "--layer-dispersion-mode",
        choices=["fixed", "grid"],
        default="fixed",
        help="'fixed' with --layer-rho 0.0 (default) reproduces the original binomial layer model exactly.",
    )
    est.add_argument(
        "--layer-rho",
        type=float,
        default=0.0,
        help="Beta-binomial overdispersion for L1/L2/L3 in fixed mode (0.0 = binomial).",
    )
    est.add_argument(
        "--layer-rho-grid",
        default=None,
        help="Comma-separated overdispersion grid when --layer-dispersion-mode grid, default 0,0.005,0.01,0.02,0.05",
    )

    est.add_argument("--q-max", type=float, default=0.5, help="Upper bound for truncated exponential q.")
    est.add_argument("--n-quad", type=int, default=96, help="Gauss-Legendre quadrature nodes for exponential component.")
    est.add_argument("--lambda-grid", default=None, help="Optional comma-separated lambda grid.")
    est.add_argument("--lambda-grid-min", type=float, default=1.0)
    est.add_argument("--lambda-grid-max", type=float, default=500.0)
    est.add_argument(
        "--lambda-grid-n",
        type=int,
        default=15,
        help="Coarse lambda grid size (refined further unless --no-lambda-refine is set).",
    )
    est.add_argument(
        "--no-lambda-refine",
        action="store_true",
        help="Disable local continuous refinement of lambda_hat and use pure fixed-grid profiling (old behavior).",
    )

    est.add_argument("--selected-dp-window", default=None, help="Use fixed selected DP window, e.g. 28:36.")
    est.add_argument("--window-selection", choices=["auto", "common", "tree-specific"], default="auto")
    est.add_argument("--min-window-start", type=int, default=12)
    est.add_argument("--min-window-cutoffs", type=int, default=4)
    est.add_argument("--max-window-cv", type=float, default=0.15)
    est.add_argument("--max-window-adjacent-change", type=float, default=0.25)

    est.add_argument("--stable-call-fraction", type=float, default=1.0, help="Fraction of selected cutoffs required for stable selected-window calls.")
    est.add_argument("--n-jobs", type=int, default=1)
    est.add_argument("--em-max-iter", type=int, default=1000)
    est.add_argument("--em-tol", type=float, default=1e-8)
    est.add_argument("--em-n-starts", type=int, default=16, help="Number of random EM starts per lambda, in addition to deterministic starts.")
    est.add_argument("--em-random-seed", type=int, default=1)
    est.add_argument("--no-plots", action="store_true", help="Do not write the plots of fixed-mutation counts per depth tier.")
    est.add_argument("--no-diagnostics", action="store_true",
                     help="Do not write the post-fit diagnostics (14_layer_resolvability*, 15_vaf_fit_check*).")
    est.add_argument("--resolvability-min-recall", type=float, default=0.5,
                     help="Layers whose model-implied recall (mean over the selected window) is below this value are "
                          "reported as unresolvable in 14_layer_resolvability.csv.")
    est.add_argument("--resolvability-background-quantile", default="0.95",
                     help="0.4.2: a layer is also reported as unresolvable when its expected VAF lies below this "
                          "quantile of the fitted background ('none': recall only, as in <= 0.4.1).")
    est.add_argument("--all-organs-screen", action="store_true",
                     help="0.4.2: remove, before the fit, candidates with alternative reads in every non-control sample "
                          "of a tree (lowest depth tier). Needs --control-organs and >= 3 other samples per tree; valid "
                          "only when no lineage after the divergence from the control is shared by all sampled organs.")
    est.add_argument("--all-organs-screen-min-alt", type=int, default=1,
                     help="Minimum alternative reads per sample for the all-organs screen (default 1).")
    est.add_argument("--fit-check-max-abs-residual", type=float, default=4.0,
                     help="Pearson-residual threshold for flagging VAF bins in 15_vaf_fit_check.csv.")
    est.add_argument(
        "--replicate-groups",
        default=None,
        help=(
            "Optional table (tab/comma separated; columns tree, sample, group) grouping organs "
            "that derive from the same meristem, e.g. neighbouring leaves of one branch. Listed "
            "trees are fitted with the replicate-group model (one latent state per variant and "
            "group); unlisted samples are singleton groups. Without this option the base model "
            "runs unchanged."
        ),
    )
    est.add_argument(
        "--control-organs",
        default=None,
        help=(
            "Per-tree control/reference organ(s) used by the upstream variant caller "
            "(e.g. Mutect2 tumor-normal mode) as the baseline other organs were called "
            "against, e.g. treeA=leaf1,treeB=leaf3. Multiple control organs for one tree "
            "(e.g. a joint call with two normal samples) are '+'-joined, e.g. "
            "treeB=leaf3+leaf5. Excluded from the depth-window metric; their calls and counts are "
            "reported and flagged (is_control_organ)."
        ),
    )
    est.add_argument(
        "--layer-proportion-sensitivity",
        action="store_true",
        help=(
            "Run the layer-proportion sensitivity sweep (writes "
            "10_layer_proportion_sensitivity.csv). Off by default, since it independently "
            "refits each selected window at every perturbed layer_proportions value (see "
            "--layer-proportion-sensitivity-steps/-layers) -- roughly "
            "(n_active_layers * n_nonzero_steps + 1)x the EM refits of the main fit -- to "
            "check how stable the fixed-mutation counts are to the assumed, never-fit layer composition."
        ),
    )
    est.add_argument(
        "--layer-proportion-sensitivity-steps",
        default=None,
        help="Comma-separated relative steps for the sensitivity sweep. Default -0.2,-0.1,0.0,0.1,0.2.",
    )
    est.add_argument(
        "--layer-proportion-sensitivity-layers",
        default=None,
        help="Comma-separated layers to sweep, e.g. L1,L2. Default: every layer with a nonzero proportion.",
    )
    est.add_argument(
        "--enable-recurrence-test",
        action="store_true",
        help=(
            "Run the organ-sharing recurrence test (writes 11_recurrence_significance.csv and "
            "12_recurrence_boosted_calls_<tree>.csv). Off by default. Flags which non-trivial "
            "organ subsets recur across candidate loci significantly more than the fitted "
            "background-noise model would predict by chance, and promotes matching borderline "
            "per-variant calls to a new, additive calling status (called_fixed_shared_validated) "
            "alongside -- never replacing -- the existing calling_status. See vafsoma.recurrence "
            "module docstring for the full method."
        ),
    )
    est.add_argument(
        "--recurrence-null-sites",
        type=int,
        default=20000,
        help="Monte-Carlo background-only sites simulated per tree to calibrate the null organ-pattern rate.",
    )
    est.add_argument(
        "--recurrence-presence-threshold",
        type=float,
        default=0.5,
        help="Per-organ mean_P_fixed threshold above which an organ counts as 'present' in a variant's organ_pattern.",
    )
    est.add_argument(
        "--recurrence-alpha",
        type=float,
        default=0.05,
        help="Family-wise significance level for the recurrence test, Bonferroni-corrected across all non-trivial organ subsets.",
    )
    est.add_argument(
        "--recurrence-seed",
        type=int,
        default=0,
        help="Random seed for the recurrence test's null Monte-Carlo simulation.",
    )

    return parser


def config_from_args(args: argparse.Namespace) -> EstimateConfig:
    return EstimateConfig(
        variants=args.variants,
        effective_genome_size=args.effective_genome_size,
        root2tip=args.root2tip,
        dataset_label=args.dataset_label,
        outdir=args.outdir,
        layer_specific_seq=args.layer_specific_seq,
        vaf_phasing=args.vaf_phasing,
        layer_proportions=args.layer_proportions,
        active_layer=args.active_layer,
        active_layer_proportion=args.active_layer_proportion,
        mu_source=args.mu_source,
        mu_mode=args.mu_mode,
        mu_L1=args.mu_L1,
        mu_L2=args.mu_L2,
        mu_L3=args.mu_L3,
        mu_L1_grid=args.mu_L1_grid,
        mu_L2_grid=args.mu_L2_grid,
        mu_L3_grid=args.mu_L3_grid,
        mu_near_one_trigger=args.mu_near_one_trigger,
        mu_near_one_offsets=args.mu_near_one_offsets,
        layer_dispersion_mode=args.layer_dispersion_mode,
        layer_rho=args.layer_rho,
        layer_rho_grid=args.layer_rho_grid,
        q_max=args.q_max,
        n_quad=args.n_quad,
        lambda_grid=args.lambda_grid,
        lambda_grid_min=args.lambda_grid_min,
        lambda_grid_max=args.lambda_grid_max,
        lambda_grid_n=args.lambda_grid_n,
        lambda_refine=not args.no_lambda_refine,
        selected_dp_window=args.selected_dp_window,
        window_selection=args.window_selection,
        min_window_start=args.min_window_start,
        min_window_cutoffs=args.min_window_cutoffs,
        max_window_cv=args.max_window_cv,
        max_window_adjacent_change=args.max_window_adjacent_change,
        n_jobs=args.n_jobs,
        em_max_iter=args.em_max_iter,
        em_tol=args.em_tol,
        em_n_starts=args.em_n_starts,
        em_random_seed=args.em_random_seed,
        stable_call_fraction=args.stable_call_fraction,
        write_plots=not args.no_plots,
        write_diagnostics=not args.no_diagnostics,
        resolvability_min_recall=args.resolvability_min_recall,
        resolvability_background_quantile=(None if str(args.resolvability_background_quantile).lower() in ("none", "null", "")
                                           else float(args.resolvability_background_quantile)),
        all_organs_screen=args.all_organs_screen,
        all_organs_screen_min_alt=args.all_organs_screen_min_alt,
        fit_check_max_abs_residual=args.fit_check_max_abs_residual,
        control_organs=parse_control_organs(args.control_organs),
        replicate_groups=read_replicate_groups(args.replicate_groups) if args.replicate_groups else None,
        layer_proportion_sensitivity=args.layer_proportion_sensitivity,
        layer_proportion_sensitivity_steps=args.layer_proportion_sensitivity_steps,
        layer_proportion_sensitivity_layers=args.layer_proportion_sensitivity_layers,
        enable_recurrence_test=args.enable_recurrence_test,
        recurrence_null_sites=args.recurrence_null_sites,
        recurrence_presence_threshold=args.recurrence_presence_threshold,
        recurrence_alpha=args.recurrence_alpha,
        recurrence_seed=args.recurrence_seed,
    )


def _print_outputs(outputs: dict) -> None:
    for name, path in outputs.items():
        if name == "outdir":
            continue
        if isinstance(path, dict):          # per-tree outputs: tree -> path
            for tree, p in path.items():
                print(f"  {name}[{tree}]: {Path(p)}")
        else:
            print(f"  {name}: {Path(path)}")
    print(f"  outdir: {Path(outputs['outdir'])}")


def _versions() -> dict:
    import numpy, pandas, scipy
    return {"vafsoma": __version__, "python": platform.python_version(), "numpy": numpy.__version__,
            "pandas": pandas.__version__, "scipy": scipy.__version__}


def main(argv: list[str] | None = None) -> int:
    from . import config as vconf
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "version":
            print(json.dumps(_versions(), indent=2)); return 0
        if args.command == "template":
            print(vconf.template_text(), end=""); return 0
        if args.command in ("run", "check"):
            cfg, base = vconf.load_config(args.config)
            if args.command == "check":
                problems = vconf.validate(cfg, base)
                if problems:
                    print("vafsoma: configuration problems:\n  - " + "\n  - ".join(problems), file=sys.stderr)
                    return 2
                ec = vconf.to_estimate_config(cfg, base)
                print(f"configuration OK: {args.config}")
                print(f"  experiment: {cfg['experiment']}\n  results:    {ec.outdir}")
                for k in ("variants", "effective_genome_size", "root2tip"):
                    print(f"  {k}: {getattr(ec, k)}")
                lay = (f"layer-specific, active {ec.active_layer}" if ec.layer_specific_seq == "yes"
                       else f"bulk, layer proportions {ec.layer_proportions}")
                print(f"  model: {lay}; {ec.vaf_phasing}; replicate groups: {'yes' if ec.replicate_groups else 'no'}")
                return 0
            ec = vconf.to_estimate_config(cfg, base, outdir=args.outdir)
            Path(ec.outdir).mkdir(parents=True, exist_ok=True)
            (Path(ec.outdir) / "00_config_resolved.json").write_text(json.dumps(
                {"vafsoma_version": __version__, "config_file": str(Path(args.config).resolve()), "config": cfg},
                indent=2, default=str) + "\n", encoding="utf-8")
            outputs = run_estimate(ec)
            print("vafSOMA run completed"); _print_outputs(outputs); return 0
        if args.command == "estimate":
            outputs = run_estimate(config_from_args(args))
            print("vafSOMA estimate completed"); _print_outputs(outputs); return 0
    except vconf.ConfigError as exc:        # configuration errors: short message
        print(f"vafsoma: error: {exc}", file=sys.stderr)
        return 2
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
