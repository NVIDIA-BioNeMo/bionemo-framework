# RL objective contract

The proposed source of truth is `rl/runs/<attempt>/artifacts/RL_OBJECTIVES.yaml`. Resolve it into each implementation/RL attempt; after approval the controller may promote the exact hash-addressed copy to rl/RL_OBJECTIVES.yaml. Never edit a shared recipe default.

## Required sections

```yaml
schema_version: 3
project_mode: adapted-design
intent:
  reference_phage: {id: null, sequence_sha256: null, topology: null}
  host: {domain: null, taxon: null, strain: null}
  original_hosts: []
  desired_change: null
  desired_host_range: {gain: [], retain: [], lose: [], avoid: []}
  protected_traits: []
  intended_use: {category: null, rationale: null}
design_scope:
  design_spec_path: null
  design_spec_sha256: null
  output_unit: complete-genome
  mutable_scope: whole-genome
  fixed_regions: []
  scope_reduction: {status: none, decision_record: null, approved_by: null}
viability_contract:
  viable_reference_set: {manifest: null, sha256: null}
  lifecycle_coverage: []
  complete_genome_checks: []
  production_host_and_dna_modification: []
host_model:
  model_id: null
  interaction_datasets: []
  sequence_mapping_manifest: null
  split_and_leakage_contract: null
  pooling_ablations: []
  calibration: null
  operating_threshold: null
therapeutic_quality:
  ema_source: null
  ema_status_verified: null
  applicability: []
  online_objectives: []
  hard_qc: []
  experimental_validation: []
  reward_support: []
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
  complete_genome_viability: []
  lifecycle_productive_infection: []
  host_range_and_directional_change: []
  therapeutic_quality: []
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

Schema 3 adds intended-use applicability, EMA-derived therapeutic-quality objectives, and reward
support to schema 2's explicit design scope, lifecycle coverage, viable references, and host-model
lineage. Preserve older artifacts as historical evidence; a new adapted project must resolve a
schema-3 contract rather than interpreting missing fields as approval or not-applicable.

When the request does not clearly state another intended use, resolve adapted design provisionally as
therapeutic, record that assumption and its rationale, and expose it for revision. Do not use a null
category to bypass the applicability review or its default online components.

Before approval, inspect the requested endpoint for replication within eukaryotic cells. Reject that
endpoint declaratively and record the rejection in the attempt; do not encode it as a reward, soft
penalty, or unresolved trade-off. A non-replicative eukaryotic entry or host-range proposal is not
automatically prohibited, but its intended use, evidence, case-specific safety review, and remaining
validation must be explicit.

Each evidence row records claim, objective/filter IDs, tier, citation/artifact, population/context, assumption, uncertainty, threshold derivation, validation, and owner/status.

Every lifecycle-coverage row records axis, applicability/rationale, target-strain and production-host
context, evidence, implicated genes/modules/motifs or host factors, measurement and controls, failure
mode, uncertainty, online objective, final filter, experimental validation, and status. The contract
is incomplete while an applicable axis is silently absent.

Each online objective records a stable ID, the user goal it serves, function/version, biological meaning or proxy status, baseline mapped to 0, target mapped to 1, monotonic clipped formula, units/direction, treatment of invalid inputs, weight/range/calibration/ablation, matching final filters, and known mismatches.

For therapeutic intended use, base `therapeutic_quality.applicability` on the design-relevant parts of
the EMA draft: [phage seed
lots](../../bionemo-phage-design/references/ema-2025-draft-phage-therapy-quality-guideline.md#phage-seed-lots),
[genome characterisation](../../bionemo-phage-design/references/ema-2025-draft-phage-therapy-quality-guideline.md#genome-characterisation),
[host range](../../bionemo-phage-design/references/ema-2025-draft-phage-therapy-quality-guideline.md#host-range),
[potency](../../bionemo-phage-design/references/ema-2025-draft-phage-therapy-quality-guideline.md#potency),
and [transducing
capacity](../../bionemo-phage-design/references/ema-2025-draft-phage-therapy-quality-guideline.md#transducing-capacity).
Each row records source section, applicability and rationale, design-time evidence, online component or
no-proxy reason, hard filter, experimental endpoint, and uncertainty. Adapted therapeutic work
includes every defensibly measurable applicable item in `online_objectives`. The recipe's current
PhiX174 replication profile also includes applicable design-relevant safety items by default while
keeping filters 1–6, 8, and 9 enabled and filter 7 disabled. Record historical and active component-set
identities separately and never directly compare aggregates built from different component sets.
Explicitly non-therapeutic adapted work records why therapy-specific rows are not applicable.

Each `reward_support` row records the component's independent denominator, reference and baseline-SFT
distribution, nonzero/partial/full-credit rates, missingness, expected gradient support, and recovery
action. A sparse or fixed-zero component triggers runtime/proxy diagnosis and, when scientifically
valid, recalibrated partial credit, proposal-distribution work, or a preapproved staged schedule. It
does not authorize dropping the component, weakening final hard QC, or zeroing the entire portfolio
behind a sequential gate.

For an increasing host-model score `s`, the default bounded form is
`clip((s - baseline) / (target_threshold - baseline), 0, 1)`. Both anchors must come from held-out,
deployment-relevant calibration and `target_threshold` must exceed `baseline`; values above the
accepted threshold stay at 1. Another saturating transform is allowed only with equivalent documented
anchors and tests.

Each per-objective counterexample row records a concrete shortcut, the affected numerator, denominator, observation count, default or proxy, expected score, biological failure, detection fixture, and correction. Include empty or missing inputs, tool failure, deletion, truncation, duplication, too few observations, threshold edges, alternate canonical forms, and selective inputs when applicable. Combined-objective rows record every component and the total for reference, baseline/random, desired, and counterexample designs; correlated or double-counted terms, weight or scale dominance, conflicts, one OR branch dominating, combinations that favor the wrong design, sensitivity or ablation results, and any added constraint. Do not approve a set of objectives when an unintended design ranks with or above the desired target without a recorded resolution.

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

For every threshold, record positive and negative consequences. During planning, distinguish a
user-supplied operational threshold from an evidence-calibrated proposal. A user-supplied threshold
may set the approved operational decision boundary when its source, rationale, comparator, model and
population identity, uncertainty, applicability limits, and validation are recorded; it does not
establish viability, bootability, productive infection, or therapeutic suitability. Prefer same-phage
outcomes, then related comparable phages, natural/reference and baseline-SFT distributions, then
clearly uncertain transferred evidence. Hold calibration data out of checkpoint selection/final
claims.

Pin validation generation before training. Checkpoint comparisons count only when prompt manifest, prompt IDs/length strata, seeds, sampling parameters, sample count, canonicalization, filter/tool versions, and denominator are compatible. Otherwise use predeclared stratified/reweighted analysis or mark the event non-comparable. Record uncertainty and a practical minimum change.

Before training, name one primary validation metric, tie breakers, cadence, patience, rebound extension, collapse rules, and filter profile/version. Changing them creates a new contract version and decision-log entry.
