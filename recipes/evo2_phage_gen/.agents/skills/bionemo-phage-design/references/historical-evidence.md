# Historical GDPO evidence snapshots

This reference keeps two generations of Microviridae/PhiX174-like GDPO evidence separate: a later operator-reported SFT+RL rerun and an earlier checksum-backed snapshot. Neither is a target for new runs, a portable launch config, or evidence of wet-lab bootability or viability.

## Later operator-reported SFT+RL rerun

The later summary was pushed on 2026-08-10 in README commit [`55efb7c2dbe799dfc8b7c67d9517186309c76499`](https://github.com/NVIDIA-BioNeMo/bionemo-recipes/commit/55efb7c2dbe799dfc8b7c67d9517186309c76499). The operator then clarified in the working tree that the new rollout used step `430` and restored the prior step-190 comparison row. The commit is an immutable report, but the raw run manifests are not present in the fetched branch, so these values are not independently checksum-verified here.

| Reported observation                      | Reported result                                                                                                                                                                | Evidence status                                                                                                 |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| SFT split and selection                   | 14,266 train, 100 validation, and 100 test genomes after the reported 99%-identity leakage check; step 5,600 selected with validation loss `0.750670` and test loss `0.798180` | Pushed README report; raw split and metric manifests absent                                                     |
| RL selection                              | GDPO ran to a 500-step ceiling; step `430` selected                                                                                                                            | Pushed README report plus operator checkpoint-label clarification                                               |
| Latest target-profile offline Arc rollout | `610/1000` (`61.00%`) with filters 1–6, 8, and 9 enabled and architecture-removal filter 7 disabled                                                                            | Pushed README report and waterfall; raw rollout manifest absent                                                 |
| Latest all-filter diagnostic branch       | `22/1000` (`2.20%`) with filter 7 also enabled                                                                                                                                 | Pushed README report; raw diagnostic manifest absent                                                            |
| Publication-era no-RL baseline            | `15/110000` (approximately `0.014%`)                                                                                                                                           | Operator-reported comparison only; its screening pipeline is not directly comparable to the offline Arc profile |

The reported target-profile waterfall is:

```text
1,000 generated
  → 998 valid nucleotide
  → 994 length/GC
  → 992 nucleotide/ORF
  → 957 protein/CheckV/GA gates
  → 815 tropism
  → 815 representatives at 99% identity
  → 815 AAI
  → 698 required genes
  → 610 synteny/total-gene final passes
```

Do not merge these later reported values with the checksum-backed snapshot below or interpret the cross-pipeline publication comparison as a controlled RL enrichment estimate. Replace the qualification above only when the actual resolved configs, manifests, checkpoint identity, metrics, and source hashes are archived.

## Checksum-backed step-190 snapshot

### Snapshot provenance

Captured 2026-07-16 at repository revision `99673b047a196352afcbb35e7aa4200127af2616`. Line locators apply only when the source checksum matches.

| Source ID | Repository-relative source                                                                                                                                                                    | Status                 | SHA-256                                                            |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- | ------------------------------------------------------------------ |
| R1        | `tmp_RUNLOG.md`                                                                                                                                                                               | local, ignored         | `2963454e53e13f1318d05d75052879a8e78bb194f1120f6673557fcdf3f8315c` |
| R2        | `tmp_e2evalpassrate.md`                                                                                                                                                                       | local, ignored         | `da592a223ddde5c113b3e04d4c5fdcb0c85b8c8dd3b1267ad4b2b3d7be71ca25` |
| C1        | [`configs/gdpo_phage_megatron.yaml`](https://github.com/NVIDIA-BioNeMo/bionemo-recipes/blob/99673b047a196352afcbb35e7aa4200127af2616/recipes/evo2_phage_gen/configs/gdpo_phage_megatron.yaml) | tracked                | `d625367e8d2d41502e5702590ffc3bbecf1143efe95da1bf5432215026d642a7` |
| C2        | [`configs/grpo_phage_megatron.yaml`](https://github.com/NVIDIA-BioNeMo/bionemo-recipes/blob/99673b047a196352afcbb35e7aa4200127af2616/recipes/evo2_phage_gen/configs/grpo_phage_megatron.yaml) | tracked inherited base | `e8ba285ff1f0a286e833f3f58a394542273eb97e5242553bce767a962e6ba32b` |

Run IDs, employee-specific telemetry namespaces, local absolute paths, and transient process identifiers were removed. The checksums retain a verification path for an operator who possesses R1/R2; the excerpts below are the portable record for everyone else.

## Direct empirical observations

| Observation                                  | Recorded result                                                                                                                       | Source locator and sanitized excerpt                                                                                                                                                                                     |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Step-190 fixed 96-design validation          | `50/96` raw full-QC (`52.08%`); `48/96` after the configured 99%-identity cluster deduplication (`50%`)                               | R1 lines 10199–10206: “samples 96 ... full-QC pass raw/deduplicated 50/48.” C1 lines 147–158 records `min_seq_id: 0.99`, `coverage: 0.0`, `cov_mode: 0`; do not silently substitute the final-rollout coverage contract. |
| Step-190 offline 1,000-design Arc evaluation | Architecture Removal disabled: `358/1000` (`35.80%`); corresponding Full branch with Architecture Removal enabled: `5/1000` (`0.50%`) | R2 lines 76–82 defines the two branches; line 141 records `1k Full Arc Final=5/1000` and `1k No-Architecture Final=358/1000`. These are offline results, not the 96-design validation.                                   |
| Same-shape TP2 96-request smoke memory       | Train generation peak about `68.3 GB/GPU`; end-validation generation about `70.9 GB/GPU` and completed without OOM                    | R1 lines 9948–9953. This was a smoke, not a memory measurement from the final run.                                                                                                                                       |
| Observation horizon                          | The monitor reached and recorded step `250`                                                                                           | R1 lines 10253–10266: “Latest completed train step ... 250” and “Monitor target satisfied.” This is the later evidence-collection point, not a rule to select step 250.                                                  |

## Configuration facts, not outcomes

C1 lines 17–64 records the step ceiling, validation cadence, KL, batches, learning rate, and completion setting. C2 lines 27–51 records Evo2 7B, bf16, the historical sequence setting, and TP2/PP1/CP1. These values describe the checked-in historical configuration; they are not context defaults and do not prove fitness, throughput, checkpoint quality, or hardware portability.

## Interpretation guardrails

- Keep the 96-design online validation and the offline 1,000-design Arc branches separate in every report.
- Step 190 is the documented selected checkpoint for this case, not a generic early-stop step. The run was observed through step 250 under a configured 500-step ceiling so later comparable evidence could inform selection.
- Never substitute an unsupported unique-cluster count for either offline result, extrapolate either as a universal rollout yield, or treat the smoke-memory values as final-run telemetry.
- For a new run, archive the actual immutable manifests, resolved config, checkpoint hash, and metric artifacts in its result directory; this historical snapshot is contextual evidence only.
