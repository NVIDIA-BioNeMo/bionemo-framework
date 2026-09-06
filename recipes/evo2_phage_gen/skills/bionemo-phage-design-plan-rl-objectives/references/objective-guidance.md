# Objective guidance

Choose rewards and hard filters from the target, intended use, available evidence, and the complete productive-infection lifecycle. Prefer target-specific measurements, then calibrated transfer evidence. A host-range or sequence model is one signal and does not establish viability or productive lysis.

For learned or stacked scorers used online, validate ranking and calibration in the high-score tail the policy will exploit, not only mean cross-validation performance. Generate every stacked input out of fold under deployment-matched groups; keep a blend only for stable incremental gain, and retain model/seed disagreement as uncertainty or out-of-domain evidence rather than averaging it away.

Apply the ML-training playbook when a learned or model-based RL scorer is explicitly requested or independently justified. A baseline without one, including PhiX, does not prohibit adding a scorer for host range, bootability, or another supported objective; plan it as a new reward component with its own endpoint, evidence, validation, and integration. Do not introduce one solely as an unsolicited response to weak non-model rewards. When developing or improving such a scorer, follow current ML best practices. Evidence-dependent options may include stronger local validation and baselines, diverse model families, feature engineering and tuning, relevant external data, simple blends, out-of-fold residual or stacked models, fold-safe pseudo-labeling, full-data fitting after design freeze, seed ensembles, or post-processing. This is an open-ended menu, not a required sequence: retain steps only for stable deployment-matched gains, especially in the policy-selected tail.

Carry these phage-specific constraints into the portfolio:

- Derive the packaging-length envelope from exact FASTA lengths of same-capsid congeners, usually at genus level; rounded kilobase metadata can erase a narrow distribution. Shape it asymmetrically where biology requires: capsid capacity is a hard upper constraint, while supported undersize can package (T7: about 85–103% of unit length).
- Protect the adsorption/receptor-binding module explicitly: it is a primary determinant of adsorption specificity and often host range, but does not establish productive infection; it is also commonly hypervariable and sorted into an unprotected variable tier. Normalize identity to a lower full-credit target than for core genes so specificity domains can diverge without losing the module.
- Treat DIAMOND, GenoPHI's 4,243 protein-family hashes, and PhageHostLearn's ESM2 RBP embeddings as blind to ribosome-binding sites, start-codon choice, codon usage, and restriction-site content; add nucleotide-level terms for required properties.
- Score the non-coding leader/promoter that bootstraps infection and transcription terminators that set structural-protein stoichiometry; gene-level objectives cannot protect them.
- Treat `Mu-like` packaging as a therapeutic exclusion: it signals a transposable temperate phage carrying host DNA at its packaged ends. Record no-host-derived-DNA evidence as a distinct composition/final-QC requirement that gene and protein screens do not satisfy.
- For a functionally swappable cross-species or module slot, score family presence, copy number, and synteny by mechanism or purpose—a curated family HMM/profile, annotation-derived module identity, or domain-level evidence—not by best hit to the reference; retain reference-homology scoring for reference-preserved slots. Virion-assembly order can persist without detectable nucleotide or protein homology ([Hatfull, 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2706577/)); in a local 27-genome alignment around the K1 phage K1PH164C1 reported by [Ferriol-González et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11448410/), its 651-aa depolymerase cross-detected only itself and one 93%-identity relative under 30%-identity/50%-bidirectional-coverage thresholds, while a tail fiber hit all 27 at 62–69% identity but only 31–35% query coverage, leaving novel depolymerases and all tail-fiber matches architecturally invisible under a 50% whole-protein coverage gate.

For each reward, record:

- the measured quantity and direction;
- a meaningful random/baseline anchor at 0 and supported target at 1;
- partial-credit behavior and valid range;
- the observations/denominator needed to compute it;
- the molecular sequence form and coordinate system used by topology-sensitive measurements;
- missing, empty, no-hit, too-short, missing-gene, invalid, and tool-failure behavior;
- biologically relevant controls known or expected high-scoring and known or expected low-scoring, with provenance and expected ordering;
- baseline, boundary, counterexample, missing, and failure controls; and
- its relationship to final hard QC and experimental validation.

Keep component scores visible. Test for duplicated evidence, scale dominance, opposing directions, sparse support, and combinations that favor the wrong endpoint; use counterfactuals to ask whether a reward improves by deleting evidence, running to the generation cap, or relocating rather than preserving a feature. A required safety or viability check remains a final exclusion or INDETERMINATE result even when an online proxy is useful for shaping.

Derive each hard gate with its own control contract:

- For a non-directional viability property, use viable references as the default naturalness prior; the wild type and chosen viable controls must pass, and the gate comes from their viable range rather than an RL full-credit plateau or brittle narrow band.
- For a hard gate meant to certify an explicitly directional change, require the wild type and known negatives to fail and known positives to pass in a confusion matrix; the natural range is a baseline, not a ceiling, while independent viability, safety, and non-directional naturalness gates remain.
- For intended-use safety, require suitable safe controls to pass and hazards to fail. If ground-truth controls are absent or the model cannot reproduce them, mark the gate unvalidated rather than tuning the threshold around the result.

Write the resolved human-readable record to `artifacts/RL_SCORE_DEFINITIONS.md` in the selected result root. Include exact zero/full-credit states and both-sided partial-credit behavior when applicable, plus concise biological rationale and evidence citations. Use the **Current PhiX174 GDPO score definitions** section in `examples/README.md` as a worked format only. An agent creates or refreshes this artifact when objectives are planned or implemented; it is not a stage of the fixed E2E script.
