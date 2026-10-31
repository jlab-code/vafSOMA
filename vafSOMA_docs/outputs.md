# Output tables

All files are written to `<outdir>/<experiment>/`. The tables are specified in "output specification 1"
(`00_run_metadata.json`, key `output.spec.version`); downstream analyses can rely on the files and columns listed here.

| File | Content |
|---|---|
| `00_config_resolved.json` | the complete configuration used (`vafsoma run`) |
| `00_run_metadata.json` | version, settings and model design |
| `01_input_validation_summary.csv` | per tree and depth tier: samples, sites, positive observations |
| `02_fit_parameters_by_tree_depth.csv` | fitted mixture per tree and tier (background λ, component weights, μ) |
| `03_fixed_counts_by_tree_depth.csv` | per tree and tier: G, fixed-mutation counts summed over organs, background level, window metric |
| `04_dp_window_scan.csv` | stability of the window metric over candidate depth windows |
| `05_selected_window_summary.csv` | per tree: selected window, mean counts per tier, diagnostics flags (below) |
| `06_variant_calls_augmented.csv` | per variant over the window: posterior probabilities and `calling_status` |
| `07_variant_calls_selected_window.csv` | per variant over the selected window: stable fixed call and layer |
| `08_organ_variant_calls_window.csv` | per variant and organ over the window: `organ_calling_status` |
| `09_candidate_fixed_mutations_<tree>.csv` | wide table of candidate fixed mutations with per-organ status |
| `10_layer_proportion_sensitivity.csv` | optional: counts under perturbed layer proportions |
| `11_recurrence_significance.csv`, `12_recurrence_boosted_calls_<tree>.csv` | optional: recurrence test |
| `13_group_variant_calls_window.csv` | replicate-group model: per variant and group, stability classification |
| `14_layer_resolvability.csv`, `14_layer_resolvability_by_depth.csv` | model-implied recall per layer (below) |
| `15_vaf_fit_check.csv`, `15_vaf_fit_check_summary.csv` | posterior-predictive VAF fit check (below) |
| `16_fixed_counts_by_unit_tier.csv` | per tree, tier and unit (organ or replicate group): fixed-mutation counts (below) |
| `17_unit_posteriors_by_tier.csv.gz` | per tree, tier, unit and candidate variant: single-observation posteriors (below) |
| `18_all_organs_screen.csv`, `18_all_organs_screened_sites.csv` | 0.4.2, only with `screen.all_organs`: per tree, the number of candidates removed before the fit, and the removed sites |
| `diagnostic_plots/` | posterior-weighted fixed-mutation counts per layer against the depth cutoff |

Units: without replicate groups the units are organs (`unit_type = organ`, `fixation_criterion = single_organ`); for a
tree with replicate groups they are the groups (`replicate_group`), with one latent state per variant and group.
Control organs (matched normals, `input.control_organs`) are written with `is_control = True`; they are excluded from
the sums in `03` and `05` and from the window metric.

## Counts

Counts come from the single-observation posteriors of each unit (`17`): an observation is a candidate variant with at
least one alternative read in that unit at that tier.

| Column | Meaning |
|---|---|
| `posterior_weighted_{layer}_count` | sum of P(fixed in layer) over the unit's observations (recommended) |
| `posterior_weighted_fixed_count` | sum of P(fixed in any layer) |
| `MAP_{layer}_count`, `MAP_fixed_count` | observations whose most probable state is the layer / any layer |
| `highconf95_fixed_count` | observations with P(fixed) ≥ 0.95 |

`16_fixed_counts_by_unit_tier.csv` has one row per tree, tier and unit with these counts and `unit`, `unit_type`,
`fixation_criterion`, `members` (the organs of the unit, `;`-separated), `is_control`, `root2tip_length` (the unit's
length from `input.root2tip`; for a replicate group the mean of its members) and `effective_genome_size` (G of the tier).
`03` gives the sums over the non-control units (`sum_<count>`), the number of units, G, the unique-variant counts and
the fit's background level per layer (`background_P_{layer}`: mean posterior of the layer among observations whose MAP
state is the background).

`17_unit_posteriors_by_tier.csv.gz` has one row per tree, tier, unit and observation: `variant_id`, `chrom`, `pos`,
`ref`, `alt`, `P_E`, `P_L1`, `P_L2`, `P_L3`, `P_fixed`, `obs_inferred_component_MAP`, `obs_inferred_fixed_MAP`,
`obs_inferred_fixed_highconf_95`, `is_control`. Summing its posteriors per unit and tier gives the counts of `16`.

## Depth window (`04`, `05`)

The depth tiers are nested filters of the same sites, so counts change gradually with the cutoff. vafSOMA chooses a
window of consecutive tiers in which the calls are stable. The window metric of a tier is the mean over the
non-control units of `MAP_fixed_count / (2 G L)` (`fixed_call_density` in `03`). A window passes if it starts at
`depth_window.min_start` or above, spans at least `min_cutoffs` tiers, the coefficient of variation of the metric is at
most `max_cv` and no adjacent change exceeds `max_adjacent_change`; the longest window passing in all trees is chosen
(fallback: all tiers). `depth_window.selected` fixes the window instead. The metric is a stability criterion only.

`05_selected_window_summary.csv` has one row per tree: the window, `unit_type`, `fixation_criterion`, `n_units` and,
for each count, `mean_<count>`: the mean over the window's tiers of the sum over non-control units. When the
diagnostics are written (default), it also carries `{layer}_model_implied_recall`, `{layer}_resolvable`,
`{layer}_status` (`reportable`, `unresolvable`, or `no_diagnostics` for an inactive layer) and the tree's
`fit_warning`.

## Layer resolvability (`14_*`)

For each tree, depth tier and layer, the model-implied recall: the probability that an observation of a variant truly
fixed in that layer is assigned to that layer by the fitted model, averaged over the depths of the tier's
observations. With replicate groups the recall is computed for the group model. A layer is reported as
`resolvable = False` when the mean over the selected window is below `diagnostics.resolvability_min_recall` (0.5).
Counts and calls of unresolvable layers should not be interpreted: most of their fixed variants are absorbed by the
background or by another layer, and their posterior mass comes mostly from background candidates.

0.4.2: a layer is also reported as `resolvable = False` when its expected VAF `mu` lies below the
`diagnostics.resolvability_background_quantile` (0.95) quantile of the fitted background allele fraction (truncated
exponential with rate λ), averaged over the selected window (columns `background_quantile`,
`mean_background_quantile_vaf`, `above_background`; per tier in `14_layer_resolvability_by_depth.csv`). The recall can
miss this case: with a light background and a small layer contribution, the layer component can take over much of the
background, so its own recall is high while its count is mostly background (simulations: L1 counts 60-400 times the
truth at L1 contributions 0.05-0.09). Set the quantile to `null` for the recall rule alone (≤ 0.4.1).

## All-organs screen (`18_*`, 0.4.2, opt-in)

With `screen.all_organs`, a candidate with at least `screen.min_alt` (1) alternative reads in every non-control sample
of a tree at its lowest depth tier is removed from all tiers before the fit. In a design where no lineage after the
divergence from the control organ is shared by all sampled organs, such candidates cannot be mutations; removing them
before the fit keeps them out of the fitted background, which otherwise becomes heavier and absorbs true fixed
mutations. The screen needs `input.control_organs` and at least three other samples per tree, and it removes true
mutations that are fixed in every sampled organ, so it is off by default. `18_all_organs_screen.csv`: tree, depth tier
used, number of non-control samples, `min_alt`, candidates at that tier, number and fraction removed;
`18_all_organs_screened_sites.csv`: the removed sites.

## VAF fit check (`15_*`)

Observed versus expected numbers of positive observations per VAF bin, with Pearson residuals. A bin is flagged when
its expected count is at least 10 and |residual| ≥ `diagnostics.fit_check_max_abs_residual` (4);
`fit_warning = True` when flagged bins occur in at least half of the selected tiers. A warning typically indicates a
heavier-tailed background than the model assumes, which can inflate the count of the lowest-VAF resolvable layer.

## Changes in 0.3.0

vafSOMA ≤ 0.2.3 also computed mutation rates (`03_rate_estimates_by_tree_depth.csv`, rate columns and bootstrap
intervals in `05`). From 0.3.0 on vafSOMA is a caller: these columns are gone, `03` was renamed
`03_fixed_counts_by_tree_depth.csv`, and `16` and `17` were added. Calls, posteriors, diagnostics and the selected
window are identical to 0.2.3 for the same inputs and settings.
