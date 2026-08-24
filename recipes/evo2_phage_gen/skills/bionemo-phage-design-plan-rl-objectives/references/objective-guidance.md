# Objective guidance

Choose rewards and hard filters from the target, intended use, available evidence, and the complete productive-infection lifecycle. Prefer target-specific measurements, then calibrated transfer evidence. A host-range or sequence model is one signal and does not establish viability or productive lysis.

For learned or stacked scorers used online, validate ranking and calibration in the high-score tail the policy will exploit, not only mean cross-validation performance. Generate every stacked input out of fold under deployment-matched groups; keep a blend only for stable incremental gain, and retain model/seed disagreement as uncertainty or out-of-domain evidence rather than averaging it away.

Apply the ML-training playbook when a learned or model-based RL scorer is explicitly requested or independently justified. A baseline without one, including PhiX, does not prohibit adding a scorer for host range, bootability, or another supported objective; plan it as a new reward component with its own endpoint, evidence, validation, and integration. Do not introduce one solely as an unsolicited response to weak non-model rewards. When developing or improving such a scorer, follow current ML best practices. Evidence-dependent options may include stronger local validation and baselines, diverse model families, feature engineering and tuning, relevant external data, simple blends, out-of-fold residual or stacked models, fold-safe pseudo-labeling, full-data fitting after design freeze, seed ensembles, or post-processing. This is an open-ended menu, not a required sequence: retain steps only for stable deployment-matched gains, especially in the policy-selected tail.

For each reward, record:

- the measured quantity and direction;
- a meaningful random/baseline anchor at 0 and supported target at 1;
- partial-credit behavior and valid range;
- the observations/denominator needed to compute it;
- missing, empty, no-hit, too-short, missing-gene, invalid, and tool-failure behavior;
- biologically relevant controls known or expected high-scoring and known or expected low-scoring, with provenance and expected ordering;
- baseline, boundary, counterexample, missing, and failure controls; and
- its relationship to final hard QC and experimental validation.

Keep component scores visible. Test for duplicated evidence, scale dominance, opposing directions, sparse support, and combinations that favor the wrong endpoint. A required safety or viability check remains a final exclusion or INDETERMINATE result even when an online proxy is useful for shaping.

Write the resolved human-readable record to `artifacts/RL_SCORE_DEFINITIONS.md` in the selected result root. Include exact zero/full-credit states and both-sided partial-credit behavior when applicable, plus concise biological rationale and evidence citations. Use the **Current PhiX174 GDPO score definitions** section in `examples/README.md` as a worked format only. An agent creates or refreshes this artifact when objectives are planned or implemented; it is not a stage of the fixed E2E script.
