# Historical 6 kb GDPO evidence snapshot

This checked-in snapshot makes a small set of historical planning facts auditable without the ignored operational logs. Use it only for the recorded Microviridae/PhiX174-like GDPO case; it is neither a target for new runs nor a portable launch config.

## Snapshot provenance

Captured 2026-07-16 at repository revision `99673b047a196352afcbb35e7aa4200127af2616`. Line locators apply only when the source checksum matches.

| Source ID | Repository-relative source | Status | SHA-256 |
| --- | --- | --- | --- |
| R1 | `tmp_RUNLOG.md` | local, ignored | `2963454e53e13f1318d05d75052879a8e78bb194f1120f6673557fcdf3f8315c` |
| R2 | `tmp_e2evalpassrate.md` | local, ignored | `da592a223ddde5c113b3e04d4c5fdcb0c85b8c8dd3b1267ad4b2b3d7be71ca25` |
| C1 | [`configs/gdpo_phage_megatron.yaml`](../../../../configs/gdpo_phage_megatron.yaml) | tracked | `d625367e8d2d41502e5702590ffc3bbecf1143efe95da1bf5432215026d642a7` |
| C2 | [`configs/grpo_phage_megatron.yaml`](../../../../configs/grpo_phage_megatron.yaml) | tracked inherited base | `e8ba285ff1f0a286e833f3f58a394542273eb97e5242553bce767a962e6ba32b` |

Run IDs, employee-specific telemetry namespaces, local absolute paths, and transient process identifiers were removed. The checksums retain a verification path for an operator who possesses R1/R2; the excerpts below are the portable record for everyone else.

## Direct empirical observations

| Observation | Recorded result | Source locator and sanitized excerpt |
| --- | --- | --- |
| Step-190 fixed 96-design validation | `50/96` raw full-QC (`52.08%`); `48/96` after the configured 99%-identity cluster deduplication (`50%`) | R1 lines 10199–10206: “samples 96 ... full-QC pass raw/deduplicated 50/48.” C1 lines 147–158 records `min_seq_id: 0.99`, `coverage: 0.0`, `cov_mode: 0`; do not silently substitute the final-rollout coverage contract. |
| Step-190 offline 1,000-design Arc evaluation | Architecture Removal disabled: `358/1000` (`35.80%`); corresponding Full branch with Architecture Removal enabled: `5/1000` (`0.50%`) | R2 lines 76–82 defines the two branches; line 141 records `1k Full Arc Final=5/1000` and `1k No-Architecture Final=358/1000`. These are offline results, not the 96-design validation. |
| Same-shape TP2 96-request smoke memory | Train generation peak about `68.3 GB/GPU`; end-validation generation about `70.9 GB/GPU` and completed without OOM | R1 lines 9948–9953. This was a smoke, not a memory measurement from the final run. |
| Observation horizon | The monitor reached and recorded step `250` | R1 lines 10253–10266: “Latest completed train step ... 250” and “Monitor target satisfied.” This is the later evidence-collection point, not a rule to select step 250. |

## Configuration facts, not outcomes

C1 lines 17–24 sets a 500-step ceiling, validation every 10 steps, and 96 validation samples. C1 lines 40–64 records KL `0.001`, train MBS/GBS `1/96`, generation/prompt batch `96`, LR `1e-6`, and `max_new_tokens=5989`. C2 lines 27–51 records Evo2 7B, bf16, maximum total sequence length 10,240, and TP2/PP1/CP1. These values describe the checked-in historical configuration; they do not prove fitness, throughput, checkpoint quality, or hardware portability.

## Interpretation guardrails

- Keep the 96-design online validation and the offline 1,000-design Arc branches separate in every report.
- Step 190 is the documented selected checkpoint for this case, not a generic early-stop step. The run was observed through step 250 under a configured 500-step ceiling so later comparable evidence could inform selection.
- Never relabel either offline result as `385`, extrapolate it as a universal rollout yield, or treat the smoke-memory values as final-run telemetry.
- For a new run, archive the actual immutable manifests, resolved config, checkpoint hash, and metric artifacts in its result directory; this historical snapshot is contextual evidence only.
