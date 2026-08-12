# NeMo-RL reward runtime contract

Runtime APIs can change. Inspect the environment actually selected for execution and pin:

- repository/package identity, version, commit, and source-tree hash;
- policy and KL-reference model identities;
- reward entry point and import path;
- callable signature, positional/keyword order, and sequence/token representation;
- batch dimensions, dtype, device, distributed reduction, and output cardinality;
- logger names and checkpoint/resume state needed by rewards.

The currently pinned recipe path expects positional reward output shaped `Tensor[B, K]` and assigns columns as `reward1`, `reward2`, and so on. Treat that only as a versioned fact to verify, not a universal API. Keep a stable mapping from each column to the objective-contract ID in the resolved run config and telemetry.

Fail before training when any checked name, order, shape, dtype, reduction, or component count differs. Do not truncate, pad, reorder, average, or rename components merely to satisfy the runtime. Update the adapter and tests deliberately, or select a compatible environment.

Apply the central [external-tool filtering and scoring
policy](../../bionemo-phage-design-adapt-execution/references/resource-and-oom-policy.md#external-tool-filtering-and-scoring).
Cache expensive safety features by sequence plus asset/policy/tool/parser hashes rather than launching
an unbounded process per rollout. Preserve deterministic row mapping and independent component records.
An accelerated implementation is optional and must demonstrate control-panel and numerical parity plus
a useful deployment-scale speedup; otherwise retain the validated path.

## Smoke evidence

For one tiny deterministic batch, retain:

- input IDs or sequence hashes;
- each raw feature, normalized component, aggregate, and hard-pass value;
- tensor shape/dtype/device before and after the runtime boundary;
- column-to-objective mapping;
- expected versus observed values;
- policy, SFT/KL reference, code, config, and environment hashes.

The smoke passes only if offline direct invocation and the runtime-observed reward vector match within the declared numerical tolerance.
