# vafSOMA

**vafSOMA** identifies somatic mutations fixed in one cell layer from variant allele-frequency (VAF) spectra of
layered plant tissue. Bulk samples mix cells of the shoot apical meristem layers (L1, L2, L3), so a
mutation fixed in one layer appears at a VAF set by that layer's contribution to the sample. vafSOMA fits a mixture of
a low-VAF background and one component per layer to the read counts of every depth tier, calls fixed mutations and
their layer from the posterior, reports the calls and the fixed-mutation counts per organ and depth tier, and flags
layers the data cannot resolve. Organs that derive from the same meristem (for example neighbouring leaves of one branch) can be analysed
jointly as replicate groups.

This repository accompanies the paper:

> **vafSOMA: Detecting cell-layer-fixed somatic mutations in plants**

vafSOMA shares its repository layout and configuration format with simSOMA (simulation of somatic VAF spectra,
https://github.com/jlab-code/simSOMA).

## Status

Version 0.4.3. vafSOMA is a caller: from 0.3.0 on it no longer computes mutation rates; instead it writes the counts,
posteriors, genome sizes and lengths from which rates or other summaries can be computed (`vafSOMA_docs/outputs.md`).
0.4.0 added a read-level observation model (per-read sequencing error and allele-fraction overdispersion,
configuration section `observation`). It is off by default, so fits and calls equal those of 0.3.0 unless it is set;
for real data, calibrate its two parameters for the data set (`vafSOMA_docs/configuration.md`).
0.4.1 added organ-specific layer compositions (`model.organ_layer_proportions`). 0.4.2 added a background condition to
the resolvability flag (a layer whose expected VAF lies below the fitted background's 95% quantile is flagged; on by
default) and an optional screen before the fit that removes sites with alternative reads in every non-control organ
(`screen.all_organs`; valid only when no lineage after the divergence from the matched normal is shared by all
sampled organs). 0.4.3 is a documentation release with results identical to 0.4.2. Changes are listed in
`CHANGELOG.md`.

A step-by-step tutorial is in a separate repository: https://github.com/jlab-code/vafSOMA-tutorial

## Install and command line

Install the tagged release with the exact library versions vafSOMA was tested with (`requirements.lock.txt`), so that
the same commands give the same results on every computer:

```bash
git clone --branch v0.4.3 https://github.com/jlab-code/vafSOMA.git
cd vafSOMA
conda create -y -n vafsoma043 python=3.13
conda activate vafsoma043
pip install -r requirements.lock.txt      # exact library versions vafSOMA was tested with
pip install .
vafsoma version                           # vafsoma 0.4.3; numpy 2.5.3, pandas 3.0.5, scipy 1.18.1
```

Without conda, a Python 3.13 virtual environment works the same way (`python3.13 -m venv .venv && source
.venv/bin/activate`, then the two `pip install` lines). Report the output of `vafsoma version` with your results; each
run also records the vafSOMA version in `00_config_resolved.json`. Activate the environment in every new terminal before
using vafSOMA (`conda activate vafsoma043`); its name contains the version, so later versions can be installed next to
it. To try vafSOMA in an existing environment, `pip install .` alone is enough (Python >= 3.10, recent numpy, pandas,
scipy, matplotlib and joblib).

```bash
conda activate vafsoma043                                   # once per terminal
vafsoma template run > my_analysis.json                     # complete configuration template
vafsoma check my_analysis.json                              # validate configuration and inputs
vafsoma run my_analysis.json                                # fit and call; results in <outdir>/<experiment>/
vafsoma run vafSOMA_configs/example_01_per_leaf.json        # synthetic example tree (a few minutes)
```

The flag-based interface of earlier versions (`vafsoma estimate --variants ... --outdir ...`) is still available.

## Inputs

vafSOMA starts from candidate somatic variants called beforehand, for example with GATK Mutect2 in multi-sample mode
against a matched normal. Per tree it needs:

- one VAF table per minimum-depth cutoff: all candidate sites at which every organ reaches the cutoff, one row per site
  and organ with reference and alternative read counts, **including organs without alternative reads**;
- the effective (callable) genome size at each cutoff;
- the root-to-tip length or age of each organ, measured from the divergence with the matched normal;
- optionally, replicate groups and the organs used as matched normal.

The format, the rules that matter and how we produced the tables with Mutect2 are described in
`vafSOMA_docs/inputs.md`.

## Configuration

A JSON file describes one analysis:

```json
{
  "experiment": "treeA_leaves",
  "outdir": "vafSOMA_output",
  "input": {"variants": "vafSOMA_inputs/treeA/vaf_tables.zip",
            "effective_genome_size": "vafSOMA_inputs/treeA/effective_genome_size.tsv",
            "root2tip": "vafSOMA_inputs/treeA/root2tip.tsv",
            "replicate_groups": "vafSOMA_inputs/treeA/replicate_groups.tsv"},
  "model": {"layer_specific_seq": false, "vaf_phasing": "unphased",
            "layer_proportions": {"L1": 0.13, "L2": 0.84, "L3": 0.03}}
}
```

Only keys that differ from the template need to be given. Relative paths are relative to the configuration file;
paths starting with `vafSOMA_inputs/` or `vafSOMA_output/` are relative to the repository root. All keys:
`vafSOMA_docs/configuration.md`.

## Outputs

- Calls: per variant (`06`, `07`), per variant and organ (`08`, `09`) and per variant and replicate group (`13`) over
  the selected depth window.
- Counts of fixed mutations per layer: posterior-weighted (sum of the layer's posterior probabilities) and MAP counts,
  per depth tier (`03`), per tier and organ or replicate group (`16`), and averaged over the selected window (`05`);
  the single-observation posteriors behind them (`17`).
- Diagnostics: the model-implied recall of each layer (`14_layer_resolvability.csv`; layers below 0.5, or, from 0.4.2,
  with an expected VAF below the fitted background's 95% quantile, are flagged as unresolvable) and a posterior-predictive VAF fit check (`15_vaf_fit_check_summary.csv`). The window summary carries
  `{layer}_status`: counts and calls of layers marked `unresolvable` should not be interpreted, because most of their
  posterior mass comes from background candidates.

`16` and `17` also carry the effective genome size of each tier, the root-to-tip length of each organ or group (as
given in the input) and the fixation criterion (single organ or replicate group). Description of all tables:
`vafSOMA_docs/outputs.md`.

## Example

`vafSOMA_inputs/examples/example_tree/` is a synthetic data set simulated with simSOMA: three branch tips with three
leaves each (formed about one week apart), bulk leaf samples, mean depth 60×, with the simulated truth for comparison.

```bash
vafsoma run vafSOMA_configs/example_01_per_leaf.json          # every leaf separately
vafsoma run vafSOMA_configs/example_02_replicate_groups.json  # leaves of a tip as replicates of one meristem
```

At this layer composition (L1 0.13) L1 and L3 are flagged as unresolvable and L2 is resolved. The VAF fit check
warns because the simulated background is gamma-shaped (shape 2, as estimated from layer-enriched apricot data)
rather than exponential. Calls in a flagged layer are not meaningful: in the replicate-group run the L3 component
(expected VAF 0.015) absorbs artefact sites shared by all leaves of a tip and reports them as fixed L3. Each example takes a few minutes on two cores.

## Repository layout

```text
vafSOMA/
├── vafsoma/               # package: model, fitting, calls, diagnostics, configuration and command-line interface
├── vafSOMA_configs/       # example configurations and the configuration template
├── vafSOMA_inputs/        # example data (synthetic, simSOMA) and input templates
├── vafSOMA_scripts/       # environment setup, run wrappers, example-data generator
├── vafSOMA_docs/          # input, configuration and output reference
├── docs/                  # pointer to the tutorial (vafSOMA-tutorial)
├── tests/                 # unit and smoke tests
├── paper/                 # scripts reproducing the paper's analyses (added with the paper's publication)
├── pyproject.toml
├── requirements.txt
├── requirements.lock.txt
├── VERSION
├── CITATION.cff
└── LICENSE
```

Runtime outputs are written to `vafSOMA_output/`, which is ignored by Git.

## Tests

```bash
pip install ".[test]"
pytest tests
```

## Citation

If you use vafSOMA, please cite the accompanying paper and the software repository. Citation metadata are provided in
`CITATION.cff`.

## License

vafSOMA is released under the MIT License. See `LICENSE`.
