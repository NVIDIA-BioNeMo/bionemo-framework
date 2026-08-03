# Evidence and proxy limits

## Essentiality

Essentiality is condition- and host-dependent. Only direct perturbation in the same phage and relevant host/condition, supported by an orthogonal line of evidence, can justify hard preservation. Record assay coverage and blind spots. Conservation, expression, structural prediction, transferred knockout evidence, and database annotation are useful soft evidence; none makes an untested gene dispensable. Preserve uncertain genes softly or route them to experimental review.

## Synteny

Decide whether the goal needs conserved organization or a deliberate break from the reference. Define:

- unit: gene IDs, homolog groups, modules, or boundaries;
- order versus adjacency, strand/orientation, circular rotation and reverse-complement equivalence;
- overlapping genes, duplicated homologs, tolerated insertions/deletions;
- positive preservation target or negative-break target;
- partial-credit distance and evidence-based pass/fail thresholds.

Do not inherit a numeric synteny threshold from a different phage without recalibration.

## Viability and bootability enrichment

No single computational proxy proves a design will boot. Combine mechanistically relevant integrity constraints, calibrated likelihood or predictive scores, and final QC; describe them as enrichment.

- **CheckV:** optional genome-quality signal. Calibrate category/score behavior for the target genome size, topology, and collection before gating.
- **gLM likelihood:** may support an RL objective or final filter. Use the model's intended input format, define scored positions, and audit outcome-correlated control-token, metadata, order/batch/rank, and padding confounds. Use applicable prior or run-specific evidence to establish within-protocol ranking or calibrated decisions. Derive any cutoff from relevant positive and negative outcomes, report class overlap and matched-control transport, and validate it held out or prospectively; do not set it from positives alone. Revalidate after material model/input/scoring/deployment changes, and do not transfer cutoffs or scorer superiority without evidence. In an unpublished PhiX174 comparison, an SFT trained with cluster-disjoint splits led the AUROC point estimate but not base Evo 2 conclusively, base had higher average precision, and neither absolute score scale transferred to broad natural controls.
- **Host-range models, including GenoPHI:** require an appropriate phenotype matrix, versioned host and phage assemblies, deployment-matched holdouts, and compatible target setting. Record interaction rows lost to unavailable genomes; assay/batch harmonization; duplicate/conflicting labels; class balance; leakage control; calibration; OOD and lineage-stratified performance. Compare per-dataset and same-taxon pooled training rather than assuming more matrices improve generalization. A whole-genome model may learn receptor, defense, counter-defense, and other lifecycle-correlated features, but its score does not prove complete-genome viability or productive lysis. Map an increasing score to a saturating reward with the calibrated negative/baseline operating point at 0 and the accepted threshold at 1; scores above the threshold remain 1. Retain the hard threshold and independent phenotype validation.
- **PhageHostLearn:** applies to its narrow Klebsiella receptor-binding-protein/K-locus setting. Do not generalize it to arbitrary hosts or whole-phage viability.

## Lifecycle-wide host range

Productive host range is not synonymous with adsorption. For the target strain and assay, cover:
access/receptor binding and genome entry; restriction-modification, CRISPR-Cas, BREX, DISARM,
abortive-infection and other defenses plus phage counter-defense; early takeover, replication and host
dependencies; structural assembly and packaging; and family-appropriate lysis and progeny production.
For therapy, also apply the design-relevant [EMA-derived intended-use
guardrails](../../bionemo-phage-design/references/design-scope-and-viability.md#apply-intended-use-therapeutic-guardrails),
including strictly lytic behavior, detrimental-cargo exclusions, and unacceptable
transduction/off-target risks. Put each applicable axis in the lifecycle table with evidence,
genes/modules or host factors, online proxy, final QC, controls and experimental validation. Marking
an axis not applicable requires a target-specific reason.

Treat virion methylation and other DNA modification as a physical production-host-dependent state.
Sequence motif avoidance, encoded modification enzymes and anti-restriction mechanisms can be scored,
but a sequence-only model cannot establish the epigenetic state of the produced particle.

## Whole-genome integrity and viable references

Use a portfolio rather than a single universal viability score: complete topology/termini and
packaging-compatible length; intact essential/key genes and modules; regulatory signals, overlaps,
copy number, order/orientation and evidence-appropriate synteny; GC, codon/oligonucleotide and
restriction-motif envelopes; calibrated genome-model/QC scores; and coverage-aware identity to a
versioned set of viable relatives. Similarity may provide lower and upper guardrails against
implausible novelty and copying, but does not freeze the reference backbone. Report clustering,
identity/coverage semantics, topology handling and training-data overlap. Unknown genes remain
unknown, not dispensable.

## Directional change

Name the biological direction relative to the reference and construct partial credit along that direction while preserving viability. Examples of dimensions—not universal objectives—include module retention/replacement, sequence distance bands, packaging limits, host phenotype, or avoidance of a disfavored feature. Research target-specific positive and negative thresholds and measure tradeoffs against protected traits.

## Diversity

Default online reward is reciprocal membership in a 99%-identity cluster. Record sequence coverage and circular canonicalization assumptions. If replacing it, show how the alternative prevents collapse and how final post-QC clustering measures the same diversity concept.
