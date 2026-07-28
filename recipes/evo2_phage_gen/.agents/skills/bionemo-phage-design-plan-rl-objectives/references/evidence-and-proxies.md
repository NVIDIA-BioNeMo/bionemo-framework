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
- **GenoPHI:** requires an appropriate phenotype training matrix and compatible target setting. Absence of that matrix blocks a defensible target-specific score.
- **PhageHostLearn:** applies to its narrow Klebsiella receptor-binding-protein/K-locus setting. Do not generalize it to arbitrary hosts or whole-phage viability.

## Directional change

Name the biological direction relative to the reference and construct partial credit along that direction while preserving viability. Examples of dimensions—not universal objectives—include module retention/replacement, sequence distance bands, packaging limits, host phenotype, or avoidance of a disfavored feature. Research target-specific positive and negative thresholds and measure tradeoffs against protected traits.

## Diversity

Default online reward is reciprocal membership in a 99%-identity cluster. Record sequence coverage and circular canonicalization assumptions. If replacing it, show how the alternative prevents collapse and how final post-QC clustering measures the same diversity concept.
