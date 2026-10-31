# Input data

vafSOMA starts from candidate somatic variants that have already been called from the sequencing reads. For each plant
(tree) it needs:

| Input | Configuration key | Required |
|---|---|---|
| Variant tables: read counts of every organ at every candidate site, one table per minimum-depth cutoff | `input.variants` | yes |
| Effective genome size per depth cutoff | `input.effective_genome_size` | yes |
| Root-to-tip length (or age) of each organ | `input.root2tip` | yes |
| Replicate groups | `input.replicate_groups` | no |
| Organs used as matched normal by the caller | `input.control_organs` | no |
| Layer composition of the organs | `model.layer_proportions` (configuration) | yes |
| Organ-specific layer composition | `model.organ_layer_proportions` (table or configuration; 0.4.1) | no |

Tables can be tab- or comma-separated, optionally gzipped. Common alternative column names are recognised (see
`vafsoma/io.py`, `*_ALIASES`). Templates are in `vafSOMA_inputs/templates/`; a complete example data set is in
`vafSOMA_inputs/examples/example_tree/`.

## 1. Variant tables (`input.variants`)

### Content

One table per **depth tier**, i.e. per minimum-depth cutoff d. The table for cutoff d contains every candidate site at
which **every organ of the tree has a depth of at least d**, normals included, with **one row per site and organ**:

| Column | Meaning |
|---|---|
| `sample` | organ name; must match the names in the root-to-tip and replicate-group tables |
| `chrom`, `pos`, `ref`, `alt` | the variant (biallelic single-nucleotide variants) |
| `ref_depth`, `alt_depth` | reads supporting the reference and the alternative allele in this organ |
| `DP` | total depth in this organ; used as the number of binomial trials (optional; default `ref_depth + alt_depth`) |
| `VAF` | `alt_depth / DP` (optional; computed if missing) |
| `tree` | tree label (optional; see below) |
| `depth_filter` | the cutoff d (optional if it is in the file name) |

```text
sample    chrom  pos     ref  alt  ref_depth  alt_depth  DP  VAF
B1_leaf1  chr1   10222   A    T    38         0          38  0.0
B1_leaf2  chr1   10222   A    T    55         0          55  0.0
B1_leaf3  chr1   10222   A    T    47         3          50  0.06
```

Rules that matter:

- **Every organ at every site, zeros included.** An organ without alternative reads at a site must still have a row
  (`alt_depth = 0` and its depth). The model uses these rows: a mutation fixed in a branch meristem should appear in
  every organ of that branch, and the replicate-group model needs the depths of organs in which it was not seen.
  Callers that report only the samples with a variant (e.g. tumour–normal pairs analysed separately) do not provide
  this; a joint multi-sample call does.
- **Candidates, not a VAF-filtered set.** Include all candidates that passed the caller's filters, without an allele-
  frequency cutoff. vafSOMA models the low-VAF background of errors and artefacts explicitly and needs to see it.
  An observation counts as positive if `alt_depth > 0`; the model is conditioned on at least one alternative read.
- **Same site set across organs within a tier.** A site enters the table for cutoff d only if all organs reach d.
  Consecutive cutoffs (e.g. 20, 21, …, 40) let vafSOMA pick a window of tiers over which the estimates are stable.
- **Germline variants removed.** Mutations shared with the matched normal should have been removed by the caller.

### Files

Give the tables as a directory, a `.zip` archive, or a single table with a `depth_filter` column. The tree is taken
from a `tree` column, else from `input.tree` in the configuration, else from a `treeNN` token in the file name. The
cutoff is taken from a `depth_filter` column or from a `dp_NN` token in the file name:

```text
treeA_dp_20_vaf.txt
treeA_dp_21_vaf.txt
...
```

Several trees can be analysed together; each tree is fitted separately.

## 2. Effective genome size (`input.effective_genome_size`)

For each cutoff d, the number of reference positions at which a candidate could have been called at that depth. vafSOMA
uses it to choose the depth window and writes it with the counts (`03`, `16`), so that counts can be expressed per
callable site.

```text
tree    depth_filter  effective_genome_size
treeA   20            201345221
treeA   21            199876002
```

Compute it with the same rule as the tiers: the number of haploid reference positions (A, C, G or T; not N) at which
**every organ** has a depth of at least d, counting only reads that pass the caller's read filters (for Mutect2:
mapping quality ≥ 20; base quality ≥ 10). For example, from `samtools depth -a -Q 20 -q 10` over all BAM files of the
tree, count the positions with an A/C/G/T reference base where the minimum depth across organs is ≥ d. The `tree`
column can be omitted for a single tree.

## 3. Root-to-tip lengths (`input.root2tip`)

```text
sample    root2tip_length
B1_leaf1  10.2
B1_leaf2  10.2
```

The developmental time (or path length) to each organ, e.g. in years or metres. Measure it **from the point at which
the organ's lineage diverged from the matched normal**: mutations that arose before this point are shared with the
normal and removed as germline, so the counts refer to the path after the divergence. vafSOMA uses the lengths to
choose the depth window (window metric `MAP_fixed_count / (2 G L)`, averaged over organs or replicate groups) and
writes them with the counts (`16`, column `root2tip_length`; for a replicate group the mean of its members). A length
is needed for every sample, including control organs (their value is not used).

## 4. Replicate groups (`input.replicate_groups`, optional)

Organs that derive from the same meristem and were formed close in time, for example two or three neighbouring leaves
of one branch tip, can be declared as a replicate group. Listed trees are fitted with the replicate-group model: one
latent state per variant and group, so that a mutation is called fixed in the group only if the reads of all members
support it. Unlisted organs are singleton groups. Counts are then reported per group, which targets mutations fixed in
the branch meristem rather than in a single organ.

```text
tree    sample    group
treeA   B1_leaf1  B1
treeA   B1_leaf2  B1
treeA   B1_leaf3  B1
```

or, inside the configuration, `{"treeA": {"B1_leaf1": "B1", "B1_leaf2": "B1", ...}}`.

## 5. Control organs (`input.control_organs`, optional)

If the caller used one or more organs as matched normal and they appear in the tables, give them per tree:
`{"treeA": "B9_leaf1"}` or `{"treeB": ["B9_leaf1", "B9_leaf2"]}`. These organs are excluded from the window metric and
from the count sums in `03` and `05`; their calls and counts are still reported, flagged (`is_control_organ` in `08`,
`is_control` in `16` and `17`).

## 6. Layer composition (configuration)

`model.layer_proportions` gives each layer's share of the organ's DNA, a_k (e.g. `{"L1": 0.13, "L2": 0.84, "L3":
0.03}` for a typical leaf). It sets the expected VAF of a heterozygous mutation fixed in layer k, φ_k = a_k / 2. Take
it from anatomy or cell-type data for the organ; check its effect with `sensitivity.layer_proportions: true`. For a
layer-enriched sample (e.g. fruit flesh for L2), set `model.layer_specific_seq: true`, `active_layer` and its purity
`active_layer_proportion`. See `configuration.md`.

### Organ-specific layer composition (0.4.1, optional)

If organs differ in composition (for example peels of different branches), give each organ its own layer
proportions with `model.organ_layer_proportions`: either a table (tab- or comma-separated; columns `sample`, `L1`, `L2`,
`L3`, optionally `tree`) or a configuration entry `{"B1_peel": {"L1": 0.72, "L2": 0.25, "L3": 0.03}, ...}`. The expected
VAF of layer k in that organ is η a_k(organ); organs not listed use `layer_proportions`. Only the modelled layers matter
(a layer with proportion 0 in `layer_proportions` stays unmodelled). The mixture weights, background and calls are still
fitted jointly over all organs. The scale factors a_k(organ)/a_k are written to `00_run_metadata.json`
(`organ.layer.scale`) and to `02_fit_parameters_by_tree_depth.csv` (`organ_layer_scale`). A composition can be estimated
from the data in two steps: fit with the common composition, then take, per organ, twice the pooled VAF of the confident
calls of the major layer as its share, and refit (see the paper's Materials and Methods). For layer-enriched samples
(`layer_specific_seq: true`) the option is not available.

## How we produced the tables (Mutect2)

For the vafSOMA paper, the tables were produced from aligned, duplicate-marked reads (one BAM file per organ) with GATK
Mutect2 (4.6.2.0):

1. **Joint calling.** All organs of a plant in one Mutect2 run (multi-sample mode), with one or more organs from an
   early-diverging part of the plant as matched normal (`-normal`). Emission thresholds were set to zero
   (`--tumor-lod-to-emit 0 --initial-tumor-lod 0`) so that weakly supported, low-VAF candidates reach the filter. No
   panel of normals or germline resource.
2. **Filtering.** FilterMutectCalls with `--threshold-strategy OPTIMAL_F_SCORE --f-score-beta 1` and a minimum median
   mapping quality of 20; no allele-fraction cutoff. Sites with `FILTER=PASS` that are biallelic SNVs were kept
   (`bcftools view -f PASS -m2 -M2 -v snps`).
3. **Tables.** For each cutoff d, the sites at which every sample's FORMAT/DP is ≥ d; `ref_depth` and `alt_depth`
   from FORMAT/AD, `DP` from FORMAT/DP. Because the call is joint, AD and DP are reported for every organ at every
   site.
4. **Genome size.** As in section 2.

Other callers can be used if their output follows the rules above, in particular read counts for every organ at every
candidate site. vafSOMA has been tested with Mutect2 output.

## Example

`vafSOMA_inputs/examples/example_tree/` is a complete synthetic data set made with simSOMA
(`vafSOMA_scripts/make_example_data.py`): three branch tips with three leaves each, bulk leaf samples, mean depth 60×,
six depth tiers. `truth_layer_carriers.csv.gz` holds the simulated per-layer carrier fractions for checking results;
the `site_id` column of the tables links each site to it (not needed for real data).
