# Run configuration

`vafsoma run CONFIG` reads one JSON configuration describing a complete analysis. `vafsoma template run` prints a
template with every key and its default (also `vafSOMA_configs/master_config_template.json`). Keys starting with `_`
are comments. Only keys that differ from the defaults need to be given; unknown keys are an error.

Paths: relative paths are relative to the configuration file; paths starting with `vafSOMA_inputs/` or
`vafSOMA_output/` are relative to the repository root (as in simSOMA). Results are written to
`<outdir>/<experiment>/`; `vafsoma run CONFIG --outdir DIR` overrides this. The resolved configuration is saved as
`00_config_resolved.json` next to the results.

| Section | Key | Default | Meaning |
|---|---|---|---|
| | `experiment` | `my_analysis` | name of the analysis and of the results folder |
| | `outdir` | `vafSOMA_output` | parent folder for results |
| `input` | `variants`, `effective_genome_size`, `root2tip` | | input tables (`inputs.md`) |
| | `tree` | `null` | tree label when the tables carry none |
| | `replicate_groups` | `null` | table path or `{tree: {sample: group}}` |
| | `control_organs` | `{}` | `{tree: organ or [organs]}` used as matched normal upstream (flagged; excluded from the window metric) |
| `model` | `layer_specific_seq` | `false` | `false`: bulk organs; `true`: layer-enriched sample |
| | `vaf_phasing` | `unphased` | `unphased` (collapsed diploid, μ = c/2) or `phased` (μ = c) |
| | `layer_proportions` | L1 0.13, L2 0.84, L3 0.03 | assumed layer contributions c_k of bulk organs |
| | `organ_layer_proportions` | null | 0.4.1: organ-specific layer contributions (table path or `{sample: {L1, L2, L3}}`); organs not listed use `layer_proportions` |
| | `active_layer`, `active_layer_proportion` | `null` | layer and purity of a layer-enriched sample |
| | `mu.source`, `mu.mode` | `derived`, `fixed` | derive μ from c_k, or give `mu.manual`; `grid` profiles over `mu.grid` |
| | `layer_dispersion` | binomial | `mode` `fixed`/`grid`, beta-binomial `rho` / `rho_grid` |
| `observation` | `read_error` | 0 | symmetric per-read sequencing error e in every component, zero counts included (0.4.0; 0 = 0.3.0 model) |
| | `read_concentration` | `null` | allele-fraction overdispersion of the reads; sets the fixed layer `rho` = 1/(k+1) |
| | `preset`, `config_sha256`, `source_version` | `null` | optional provenance of the calibration (free text), written to `00_run_metadata.json` |
| `depth_window` | `selected` | `null` | fixed window `"28:36"`; `null` selects a stable window |
| | `selection`, `min_start`, `min_cutoffs`, `max_cv`, `max_adjacent_change` | | automatic window selection |
| `background` | `q_max`, `n_quad`, `lambda_*` | | truncated-exponential background and its λ grid |
| `em` | `n_starts`, `random_seed`, `max_iter`, `tol` | 16, 1, 1000, 1e-8 | EM fitting |
| `calls` | `stable_call_fraction` | 1.0 | fraction of window cutoffs required for a stable call |
| `diagnostics` | `enabled`, `resolvability_min_recall`, `resolvability_background_quantile`, `fit_check_max_abs_residual` | true, 0.5, 0.95, 4 | post-fit diagnostics; 0.4.2: a layer below the background's 0.95 quantile is unresolvable (`null`: recall only) |
| `screen` | `all_organs`, `min_alt` | false, 1 | 0.4.2: all-organs screen before the fit (needs `control_organs` and ≥ 3 other samples per tree; see outputs.md) |
| `sensitivity` | `layer_proportions`, `steps`, `layers` | false | layer-proportion sensitivity sweep |
| `recurrence_test` | `enabled`, … | false | organ-sharing recurrence test |
| `compute` | `n_jobs`, `plots` | 1, true | parallel jobs; plots of counts per depth tier |

Notation: the configuration keys keep their historical names; in the paper, the layer contributions c_k are written
a_k and the expected layer VAFs μ_k (`mu`) are written φ_k.

The template layer proportions are the mean of published leaf compositions, as in the simSOMA layered template.
Adapt them to the tissue; their effect can be checked with the sensitivity sweep.

## Observation model (0.4.0)

The read model: a site with allele fraction v reports the
alternative allele with probability p = v(1 − e) + (1 − v)e, and v is overdispersed (beta distribution with
intra-class correlation ρ = 1/(k + 1)). vafSOMA uses this in every component: the layer components are beta-binomial
with mean e + (1 − 2e)μ_k and the matching dispersion (moment-matched; exact when e = 0), the background integrates the
read probability e + (1 − 2e)q over the truncated exponential, and the probability of zero reads (the positivity
conditioning and the zero counts of sibling replicates) includes the error. e and k are calibrated for the data set
beforehand rather than estimated by vafSOMA (the calibration used in the paper is described in its Supplementary Text,
Section S7); for the apricot data of the paper, e = 3 × 10⁻⁴ and k = 140:

```json
"observation": {"read_error": 3e-4, "read_concentration": 140}
```

With e > 0 the background also has an error-only part (q = 0) with its own weight, fitted like the other weights:
organs or replicate groups that do not carry a variant still show stray error reads at its site (next to zeros in
sibling replicates), and without this part the background rate is driven up to mimic a point mass at zero (false calls
at high depth). Calibration therefore enters only through e and k; the structure is the same for every data set.

With the defaults (e = 0, no concentration) the model is the 0.3.0 model and the outputs are
unchanged.

## Configurations of vafSOMA 0.2.x

The keys `bootstrap` and `model.ploidy_factor` belonged to the rate estimation removed in 0.3.0. They are ignored with
a warning, so that older configuration files still run.

## Configurations of vafSOMA ≤ 0.1.7

TOML configurations (`vafsoma_run_config.toml`) are still read and translated, with paths relative to the current
directory as before. This transition support will be removed in a later release. The flag-based
`vafsoma estimate` command is unchanged.
