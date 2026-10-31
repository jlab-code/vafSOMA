# Changelog

## 0.4.3 (2026-10-31)

Documentation and provenance cleanup. **Model, fits, posteriors, calls, diagnostics and counts are identical to 0.4.2.**

- Configuration key `observation.plantsoma_obs_version` renamed to `observation.source_version` (free-text provenance
  of the observation-model calibration); the old name is still accepted. The metadata key in `00_run_metadata.json`
  is now `observation.source_version`.
- Documentation, docstrings and test descriptions revised; no change to code paths.
- Tests: 1 new (old provenance key still accepted).

## 0.4.2 (2026-10-28)

- **Resolvability flag, background condition (default on).** A layer is also reported as unresolvable when its expected
  VAF lies below the 95% quantile of the fitted background allele fraction (truncated exponential; option
  `diagnostics.resolvability_background_quantile`, CLI `--resolvability-background-quantile`, `null`/`none` restores
  the recall rule of ≤ 0.4.1). New columns in `14_layer_resolvability.csv` (`background_quantile`,
  `mean_background_quantile_vaf`, `above_background`) and `14_layer_resolvability_by_depth.csv`
  (`background_quantile_vaf`). Why: with a light background and a small minor-layer contribution, the layer component
  can absorb background candidates and reach a high model-implied recall, so that the layer is reported resolvable
  with counts far above the truth. Fits, posteriors and calls are unchanged; only the flag (and what is reported as
  reportable in `05`) changes.
- **All-organs screen before the fit (opt-in).** `screen.all_organs` (CLI `--all-organs-screen`,
  `--all-organs-screen-min-alt`): candidates with alternative reads in every non-control sample of a tree at its lowest
  depth tier are removed from all tiers before fitting (new module `vafsoma.screen`; outputs `18_all_organs_screen.csv`,
  `18_all_organs_screened_sites.csv`; recorded in `00_run_metadata.json`). Needs control organs and ≥ 3 other samples
  per tree; valid only when no lineage after the divergence from the control is shared by all sampled organs. Why:
  such candidates are recurrent artefacts in these designs; removing them before the fit keeps them out of the fitted
  background, whereas removing them from the calls afterwards leaves a heavier background in the fit.
- Tests: 7 new (`tests/test_042.py`). Existing tests unchanged.
- Without the screen and with `resolvability_background_quantile: null`, outputs are those of 0.4.1 (plus the new
  metadata keys and columns).

## 0.4.1 (2026-10-27)

Organ-specific layer compositions. **Without the new option, models and outputs are those of 0.4.0** (all tests of
0.4.0 pass unchanged).

- New option `model.organ_layer_proportions` (Python API: `EstimateConfig(organ_layer_proportions=...)`): a table
  (columns `sample`, `L1`, `L2`, `L3`, optional `tree`) or `{sample: {L1, L2, L3}}`. The expected VAF of layer k in an
  organ becomes η a_k(organ); organs not listed use `layer_proportions`. Bulk organs only.
- Implementation: `MuTriple.organ_scale` carries the factors a_k(organ)/a_k; the per-organ model groups identical
  observations by organ as well, and every layer likelihood (fit, classification, per-organ calls) uses the organ's
  expected VAF. Replicate-group model: each replicate uses its organ's VAF (`build_units` keeps the organs,
  `ReplicateLikelihood(by_sample=True)`). Diagnostics: the model-implied recall is computed per organ and averaged with
  weights proportional to the organ's observations; the VAF fit check sums observed and expected counts over organs;
  the group recall simulates each replicate at its organ's VAF. The layer-proportion sensitivity sweep keeps the
  organ ratios.
- Outputs: `organ.layer.scale` in `00_run_metadata.json`; column `organ_layer_scale` (JSON) in `02` when the option is
  used. `read_params`, `layer_loglik` and the beta-binomial helpers accept arrays of μ.
- Why: when organ compositions vary (for example, fruit peels of different branches), a common composition can bias
  layer counts.
- Tests: 10 new (`tests/test_organ_composition_041.py`).

## 0.4.0 (2026-10-26)

The likelihood can use a read-level observation model (per-read sequencing error and allele-fraction
overdispersion). **With the defaults the model and all outputs are those of 0.3.0** (all 0.3.0 tests pass unchanged).

- New configuration section `observation`: `read_error` (symmetric per-read sequencing error e), `read_concentration`
  (allele-fraction overdispersion k; sets the fixed layer ρ = 1/(k + 1)), and optional provenance keys, all written to
  `00_run_metadata.json` (`observation.*`). Python API: `EstimateConfig(read_error=..., layer_rho=..., obs_preset=...,
  obs_config_sha256=..., obs_version=...)`.
- e enters every component: layer components use the read probability e + (1 − 2e)μ_k (beta-binomial with the
  moment-matched dispersion when ρ > 0; `model.read_params`), the background integrates e + (1 − 2e)q, and all zero-read
  probabilities (positivity conditioning; zero counts of replicates in the group model) include e.
- With e > 0 the background has an **error-only part** (q = 0: the organ or replicate group does not carry the variant
  and every alternative read is a sequencing error) with its own mixture weight, in both the per-organ and the
  replicate-group model. It is part of the model whenever e > 0 and vanishes at e = 0, where the model is the 0.3.0
  model. Why: replicates of other groups at a mutated site receive stray error reads next to zeros in their siblings;
  without this part only an exponential background with a very large λ can represent q = 0, and the replicate-group
  model then absorbs background into the layer components. Its weight is written as `pi_E_error` (`02`, only when
  e > 0) and its posterior as `P_E_error` (group calls, only when e > 0); it is part of E (`pi_E`, `P_E`) in all other
  outputs.
- Group-model diagnostics: the model-implied recall now simulates replicate reads with the fit's observation model
  (ρ and e). Before, ρ was ignored in that simulation (fix; no change for ρ = e = 0).
- Not changed: the organ-sharing recurrence test simulates its null without e.
- Tests: 20 new; two group-model tests adapted to the additional internal column.

## 0.3.0 (2026-10-23)

vafSOMA is now a caller only: it identifies layer-fixed somatic mutations, assigns their layer and reports counts and
diagnostics, but no longer estimates mutation rates. **Model, fits, posteriors, calls, diagnostics and the selected
depth window are identical to 0.2.3** for the same inputs and settings (checked on both example configurations:
outputs 02, 06–09, 13–15 byte-identical).

- Removed: the rate estimator (mean over organs of N/(2 G L)), the rate bootstrap (`vafsoma/bootstrap.py`), the
  background-subtracted and empirical-Bayes estimators, the rate columns of `03` and `05`, `--bootstrap-B`,
  `--bootstrap-seed`, `--ploidy-factor` and the configuration keys `bootstrap` and `model.ploidy_factor` (ignored with a
  warning in older configuration files).
- `03_rate_estimates_by_tree_depth.csv` is now `03_fixed_counts_by_tree_depth.csv`: G, counts summed over the
  non-control units (`sum_<count>`), unique-variant counts, the fit's background level per layer, `unit_type`,
  `fixation_criterion`, `n_units` and the window metric `fixed_call_density`.
- New `16_fixed_counts_by_unit_tier.csv`: counts per tree, depth tier and unit (organ or replicate group), with the
  unit's members, `root2tip_length`, `effective_genome_size`, `fixation_criterion` and `is_control`.
- New `17_unit_posteriors_by_tier.csv.gz`: the single-observation posteriors behind these counts, per tree, tier, unit
  and candidate variant.
- `05_selected_window_summary.csv`: per tree the selected window, the mean counts per tier (`mean_<count>`) and the
  diagnostics flags; `{layer}_rate_status` is now `{layer}_status`.
- Depth-window selection is unchanged. Its metric, called `rate_MAP_fixed` before, is now `fixed_call_density`
  (`04`: `mean_density`, `cv_density`, …); only this metric is scanned.
- `10_layer_proportion_sensitivity.csv` reports mean counts instead of the MAP rate.
- Diagnostic plots show posterior-weighted counts per layer and tier.
- The output tables are versioned as output specification 1 (`00_run_metadata.json`: `output.spec.version`).
- Tests: rate tests removed with the estimator; new checks that `16` and `17` agree with each other and with `03`, and
  that the removed configuration keys are ignored.

## 0.2.3 (2026-10-23)

- **Bootstrap intervals corrected.** The depth tiers of a window are nested subsets of the same sites and reads.
  Versions <= 0.2.2 resampled independently at each cutoff and averaged the draws across the window, which made the
  intervals far too narrow (the more cutoffs, the narrower). The bootstrap now resamples the candidate variants of
  each organ or replicate group once per replicate and applies the same draw to every cutoff. The fitted weights and
  background are held fixed, so the intervals do not include fitting uncertainty. Point estimates are unchanged.

## 0.2.2 (2026-10-22)

- Bootstrap intervals for the posterior-weighted rates in `05_selected_window_summary.csv`, per layer
  (`{layer}_posterior_weighted_rate_CI_lower/upper`) and pooled (`posterior_weighted_rate_CI_lower/upper`), with the
  same resampling scheme as the MAP intervals. The posterior-weighted rate is the recommended estimate.
- The rate summary now carries the diagnostics: `{layer}_model_implied_recall`, `{layer}_resolvable`,
  `{layer}_rate_status` (`reportable` / `unresolvable` / `no_diagnostics`) and `fit_warning`. Rates of unresolvable
  layers are still computed but marked.
- `vafsoma prepare` (added in 0.2.1) is removed: vafSOMA starts from VAF tables. The preprocessing from BAM files will
  be released separately once it has been validated. The input specification (`vafSOMA_docs/inputs.md`) now describes
  the required format, the rules that matter and how the paper's tables were produced with Mutect2.
- Model, fitting, posteriors, calls and the existing columns are unchanged: results are identical to 0.2.1.

## 0.2.1 (2026-10-16)

- Experimental command `vafsoma prepare` (preprocessing from BAM files through an optional external package); removed
  again in 0.2.2.
- Model, fitting, posteriors, calls and diagnostics are unchanged; results are identical to 0.2.0.

## 0.2.0 (2026-10-12)

Repository and interface restructured to match simSOMA. Model, fitting, posteriors, calls and
diagnostics are unchanged: results are identical to 0.1.7 for the same settings.

- Layout: the package moved from `src/vafsoma/` to `vafsoma/`; new folders `vafSOMA_configs/`, `vafSOMA_inputs/`
  (examples, templates), `vafSOMA_scripts/`, `vafSOMA_docs/`, `docs/`; `VERSION`, `CITATION.cff`, `LICENSE`,
  `requirements.txt` and `requirements.lock.txt` added.
- JSON run configurations (`vafsoma/config.py`): `vafsoma run CONFIG [--outdir DIR]`, `vafsoma check CONFIG`,
  `vafsoma template run`, `vafsoma version`. Relative paths are relative to the configuration file; paths starting
  with `vafSOMA_inputs/` or `vafSOMA_output/` are relative to the repository root (as in simSOMA). Results go to
  `<outdir>/<experiment>/` together with `00_config_resolved.json`. Unknown keys are an error.
- Configuration files of 0.1.x (TOML) are still read and translated (transition support).
- `vafsoma estimate` (flag-based) is unchanged. `vafsoma prepare-project`, the `projects/<name>/upstream|converted`
  convention and the wrappers `run_vafsoma`, `run_prepare_vafsoma`, `run_vafsoma_from_config` were removed: the
  preparation step only copied and renamed files, which the configuration now does by pointing at the tables directly.
- New option `input.tree` (`EstimateConfig.tree_label`): tree label for tables that carry none.
- Synthetic example data set made with simSOMA (`vafSOMA_inputs/examples/example_tree/`,
  `vafSOMA_scripts/make_example_data.py`) and two example configurations (per-leaf and replicate-group model).
- Template layer proportions L1 0.13 / L2 0.84 / L3 0.03 (mean of published leaf compositions, as in simSOMA). The
  default of `vafsoma estimate` stays L1 0.20 / L2 0.72 / L3 0.08.
- Tests: the prepare-project test is replaced by configuration/CLI tests, including equality of `vafsoma run` and the
  estimator called directly.

## 0.1.7 (2026-10-12)

Replicate-group mode only; the per-organ model, its calls and diagnostics are unchanged.

- Fix: group-level window calls (`13_group_variant_calls_window.csv`) computed the fraction of fixed / layer MAP calls
  over all depth cutoffs of the selected window instead of over the cutoffs at which the unit is present (as the
  per-organ and per-variant summaries do). Units that dropped out of higher cutoffs because of the site depth filter
  could therefore never be called stably fixed and were reported as `evaluated_unstable`. In simulated trees this cost
  up to half of the branch-fixed L2 mutations at low depth.
- Group-aware resolvability: for trees with replicate groups, the model-implied recall in
  `14_layer_resolvability*.csv` is evaluated per group (replicate vectors simulated at mu_k with the depth vectors of
  the observed units and classified by the fitted group mixture). 0.1.6 used the per-organ recall with the group fit's
  weights, which flagged resolvable layers (L1 at balanced compositions) as unresolvable.
- New test: `test_group_stability_uses_cutoffs_where_unit_is_present`.

## 0.1.6 (2026-10-07)

Output-only additions; model, fitting, posteriors and calls are identical to 0.1.5 (verified on simulated data:
identical calling status for every observation, numerical differences < 1e-13; all 0.1.5 tests pass).

- New module `vafsoma.diagnostics`.
- `14_layer_resolvability.csv` / `14_layer_resolvability_by_depth.csv`: model-implied recall per layer and a
  `resolvable` flag (option `--resolvability-min-recall`, default 0.5).
- `15_vaf_fit_check.csv` / `15_vaf_fit_check_summary.csv`: observed vs expected positive observations per VAF bin
  with Pearson residuals and a `fit_warning` (option `--fit-check-max-abs-residual`, default 4).
- `--no-diagnostics` switches both off.
- New tests: `tests/test_diagnostics.py`.
