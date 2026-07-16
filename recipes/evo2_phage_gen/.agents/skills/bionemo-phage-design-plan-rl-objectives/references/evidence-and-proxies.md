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
- **Evo2 likelihood:** use as a viability or bootability-enrichment proxy only after fitting a threshold on base Evo2 7B-1M scores from a relevant labeled success/failure set. The checked-in PhiX174 proof-of-concept cohort contains 302 designs with 16 viable outcomes; rescore that cohort with the exact public deployment checkpoint, then fit and evaluate the operating threshold with held-out or nested validation that accounts for class imbalance and the intended precision/recall tradeoff. Never fit and report on the same designs. Do not call the threshold a bootability guarantee or transfer it untested across models, phages, genome lengths, or tasks.
- **GenoPHI:** requires an appropriate phenotype training matrix and compatible target setting. Absence of that matrix blocks a defensible target-specific score.
- **PhageHostLearn:** applies to its narrow Klebsiella receptor-binding-protein/K-locus setting. Do not generalize it to arbitrary hosts or whole-phage viability.

## Directional change

Name the biological direction relative to the reference and construct partial credit along that direction while preserving viability. Examples of dimensions—not universal objectives—include module retention/replacement, sequence distance bands, packaging limits, host phenotype, or avoidance of a disfavored feature. Research target-specific positive and negative thresholds and measure tradeoffs against protected traits.

## Diversity

Default online reward is reciprocal membership in a 99%-identity cluster. Record sequence coverage and circular canonicalization assumptions. If replacing it, show how the alternative prevents collapse and how final post-QC clustering measures the same diversity concept.
