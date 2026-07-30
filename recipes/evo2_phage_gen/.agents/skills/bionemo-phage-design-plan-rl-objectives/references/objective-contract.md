# RL objective contract

The proposed source of truth is `rl/runs/<attempt>/artifacts/RL_OBJECTIVES.yaml`. Resolve it into each implementation/RL attempt; after approval the controller may promote the exact hash-addressed copy to rl/RL_OBJECTIVES.yaml. Never edit a shared recipe default.

## Required sections

```yaml
schema_version: 1
intent:
  reference_phage: {id: null, sequence_sha256: null, topology: null}
  host: {domain: null, taxon: null}
  desired_change: null
  protected_traits: []
lineage:
  sft_project: null
  sft_stage_name: null
  sft_stage_type: null
  sft_run_id: null
  sft_checkpoint: {iteration: null, content_sha256: null}
  prompt_manifest:
    path: null
    sha256: null
    reference_genome_sha256: null
    derivation: null
    tokenizer: null
    tokenizer_version: null
    prompt_lengths: []
    seed: null
    procedure_version: null
validation_generation:
  manifest_path: null
  manifest_sha256: null
  prompt_ids: []
  prompt_length_strata: []
  seeds: []
  sampling: {}
  sample_count: null
  filter_profile_sha256: null
goal_trace:
  viability: []
  bootability_enrichment: []
  directional_change: []
  diversity: []
evidence: []
online_objectives: []
adversarial_analysis:
  per_objective: []
  portfolio: []
hard_qc: {}
checkpoint_selection: {}
calibration_and_ablations: []
unresolved_decisions: []
```

Each evidence row records claim, objective/filter IDs, tier, citation/artifact, population/context, assumption, uncertainty, threshold derivation, validation, and owner/status.

Each online objective records stable ID, goal trace, function/version, biological meaning/proxy status, baseline mapped to 0, target mapped to 1, monotonic clipped formula, units/direction, fail-closed invalid behavior, weight/range/calibration/ablation, matching final filters, and known mismatch.

Each per-objective adversarial row records a concrete shortcut candidate, affected numerator/denominator/support/default or proxy, expected score, biological failure, detection fixture, and mitigation. Include empty/missing/tool-failure, deletion/truncation/duplication, denominator shrinkage, threshold-edge, canonicalization, and support-manipulation cases when applicable. Portfolio rows record the component vector and aggregate for reference, baseline/random, desired, and adversarial designs; correlated/double-counted terms, weight/scale dominance, conflicts, OR dominance, A+B-to-C failure modes, sensitivity/ablation result, and any added guardrail. Do not approve a portfolio when an unintended design ranks with or above the desired target without a recorded resolution.

## Logic

Represent hard QC recursively:

```yaml
hard_qc:
  op: all
  children:
    - {filter: genome_integrity}
    - op: any
      children: [{filter: route_a}, {filter: route_b}]
```

An all branch normally keeps distinct GDPO objectives. An any branch normally has one aggregate online objective max(route_a, route_b) and final hard any. Always log child scores, pass flags, and dominance. Use another aggregation only with calibration.

## Threshold and validation selection

For every threshold, record positive and negative consequences. Prefer same-phage outcomes, then related comparable phages, natural/reference and baseline-SFT distributions, then clearly uncertain transferred evidence. Hold calibration data out of checkpoint selection/final claims.

Pin validation generation before training. Checkpoint comparisons count only when prompt manifest, prompt IDs/length strata, seeds, sampling parameters, sample count, canonicalization, filter/tool versions, and denominator are compatible. Otherwise use predeclared stratified/reweighted analysis or mark the event non-comparable. Record uncertainty and a practical minimum change.

Before training, name one primary validation metric, tie breakers, cadence, patience, rebound extension, collapse rules, and filter profile/version. Changing them creates a new contract version and decision-log entry.
