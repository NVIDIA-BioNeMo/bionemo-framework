# Phage-design validation snapshot

> Human audit record, not agent instructions. Agents should use ../SKILL.md and only the references it names. This asset is not linked from SKILL.md, so normal skill loading does not consume it.

## Snapshot

- Recorded: 2026-07-17 UTC
- Base revision: 54250070db2d407b3ee2f2e65e0c8202e0904590
- Branch: jstjohn/evo2_phage_gen
- Evaluated changes: recipe-local .agents/** and README.md
- Runtime scope: skills, documentation, literature assets, and standard-library utilities; no SFT, RL, inference, GPU, scheduler, or cloud job was launched

## Results

| Surface | Result |
| --- | --- |
| Skill structure | PASS — all 10 controller/subskill directories passed the skill-creator validator |
| Behavioral eval schema | PASS — 10 portable eval files, 26 globally unique exact-field cases |
| Portable eval runner | PASS — 41/41 unit tests across Codex and Claude adapters |
| Harness dry runs | PASS — Codex and Claude each produced 26/26 zero-cost plans |
| Live Claude behavioral eval | PASS — effective current suite 26/26: 22 unaffected clean-sweep passes plus four passing current-definition reruns |
| Claude plugin bridge | PASS — Claude Code 2.1.211 validates .agents as local plugin evo2-phage-gen |
| Codex plugin bridge | PASS — the official validator accepts .agents/.codex-plugin/plugin.json and its ./skills/ bundle |
| Claude workspace isolation | PASS — Git-index allowlist, explicit answer/audit/generated-path exclusions, outward-symlink rejection, required-path checks, and a per-file content manifest |
| Context size | PASS — SKILL.md entrypoints remain 331–667 words; detail is routed to references |
| Markdown links | PASS — 41 Markdown files and 174 relative links, 0 broken |
| Literature utility | PASS — 25/25 unit tests |
| Literature manifests | PASS — both paper bundles verified with zero errors and reconstructed byte-identically offline |
| Official workbook | PASS — media-1.xlsx SHA-256 3cd26d4cca8bc1273a863c4b2304e755635fe0c7bed46308f54029b88f063fc9 |
| Workbook extraction | PASS — 302 rows, 33 columns, 302 unique IDs; repeated extraction produced the same TSV hash |
| Recipe README | PASS — 211-line, results-first human guide with Codex and Claude launch examples |
| Result isolation | PASS — results/ is gitignored from the recipe root |
| Historical evidence | PASS — the checked-in sanitized snapshot separates empirical outcomes from configuration facts and retains source hashes/locators |
| Portability | PASS — runtime skills and README contain no employee-only W&B destination, local user path, disallowed spreadsheet source, or unsupported historical count |
| Policy ownership | PASS — active recipe instructions add no custom biological policy; each harness retains its normal policy behavior |

## Reproducible behavioral evals

Each owning skill has evals/evals.json with the BioNeMo Agent Toolkit-compatible fields id, prompt, expected_output, assertions, expected_skill, and expected_script. Scientific-discovery cases grade source properties and evidence quality rather than a fixed paper identity, so a stronger future source can pass. Exact historical-number cases remain intentionally fixed.

The standard-library runner follows the [OpenAI skill-eval loop](https://developers.openai.com/blog/eval-skills): prompt, preserved trace/artifacts, deterministic checks, then a fresh structured rubric grade. Harness-specific execution and provenance stay outside the portable eval JSON.

From recipes/evo2_phage_gen:

~~~bash
# Validate shape, ownership, required fields, and globally unique IDs.
python .agents/skills/bionemo-phage-design/scripts/run_skill_evals.py --validate

# Discover case IDs before spending model calls.
python .agents/skills/bionemo-phage-design/scripts/run_skill_evals.py --list

# Plan all Codex calls without invoking a model.
python .agents/skills/bionemo-phage-design/scripts/run_skill_evals.py \
  --dry-run --all \
  --results-dir results/skill-evals-codex-dry-$(date -u +%Y%m%dT%H%M%SZ)

# Run one Codex generation and independent structured grade.
python .agents/skills/bionemo-phage-design/scripts/run_skill_evals.py \
  --run \
  --case bionemo-phage-design-implement-rl-objectives-001-denominator-gaming \
  --results-dir results/skill-evals-codex-$(date -u +%Y%m%dT%H%M%SZ)

# Plan all Claude calls through the local plugin.
python .agents/skills/bionemo-phage-design/scripts/run_skill_evals.py \
  --harness claude --dry-run --all \
  --results-dir results/skill-evals-claude-dry-$(date -u +%Y%m%dT%H%M%SZ)

# After approving external transfer, run one Claude case.
python .agents/skills/bionemo-phage-design/scripts/run_skill_evals.py \
  --harness claude --run --allow-external-skill-upload \
  --case bionemo-phage-design-implement-rl-objectives-001-denominator-gaming \
  --results-dir results/skill-evals-claude-$(date -u +%Y%m%dT%H%M%SZ)
~~~

Bare --run refuses to execute every case; add --all intentionally. Result roots are reserved before execution and never reused.

For live Claude runs, the runner stages current working-tree bytes only for Git-indexed paths. It excludes every eval definition, VALIDATION.md, results directory at any depth, tmp path component, generated package metadata, runner audit-test directory, common caches/data, and outward or unresolved symlink. Required plugin and selected SKILL.md paths must survive staging. Provenance records every staged file/hash, every exclusion, and a whole-manifest hash. Dirty tracked edits are included; untracked files are absent.

Generation has read/search/skill tools but no Bash/Edit/Write, loads no user/project/local settings, disables CLAUDE.md and auto-memory, and has no session persistence. Grading uses a second process without the target plugin or tools. The external-transfer flag acknowledges that Claude may receive the prompt, skill text, and any staged recipe file it reads.

Neither adapter pins a model by default. Each CLI resolves its own default and the runner records requested overrides, observed models, executable/version, setting isolation, and reported cost. Optional --model and --grader-model flags support controlled comparisons without putting a model choice in portable eval JSON.

## Clean live Claude campaign

An isolation audit invalidated all earlier Claude campaigns: the original staging approach could expose ignored run-history files, and an intermediate Git-allowlist version still admitted tracked eval-audit tests. Those campaigns are not evidence for this snapshot.

The final clean initial sweep produced 24 passes and two genuine failures:

1. A stop-before-code RL objective audit omitted planned runtime-boundary checks and a deterministic integration smoke.
2. A research packet omitted two axes from its objective-portfolio completeness review.

The owning skills were tightened without changing the eval assertions, static regressions were added, and both failed cases passed in fresh sanitized workspaces. The unchanged sibling case for each modified skill was also rerun and passed against the final skill bytes. The effective result is therefore 26/26: 22 unaffected initial passes plus four current-definition rerun passes. The two initial failures and two old-definition sibling passes remain in their immutable roots as superseded evidence; they are not silently relabeled.

| Campaign | Raw result | Reported cost (USD) |
| --- | ---: | ---: |
| final-isolated-v2-a | 9 pass | 4.53628650 |
| final-isolated-v2-b | 8 pass, 1 superseded fail | 4.17702750 |
| final-isolated-v2-c | 7 pass, 1 superseded fail | 3.68514100 |
| rerun-joint | 1 pass | 0.63338275 |
| rerun-research | 1 pass | 0.87150925 |
| rerun-implement-sibling | 1 pass | 0.53751350 |
| rerun-research-sibling | 1 pass | 0.86046550 |
| Total execution evidence | 28 pass, 2 superseded fail | 15.30132600 |

The 26 accepted current-definition case executions account for USD 12.71906975; the four retained superseded executions account for USD 2.58225625.

- Full-sweep roots: results/skill-evals-claude-final-isolated-v2-{a,b,c}-20260717/
- Current-definition reruns: results/skill-evals-claude-final-isolated-v2-rerun-{joint,research,implement-sibling,research-sibling}-20260717/
- Claude Code: 2.1.211 (Claude Code)
- Requested generation/grader models: none
- Observed generation models: claude-opus-4-8[1m], claude-opus-4-8, and the reported claude-haiku-4-5-20251001 helper
- Observed grading models: claude-opus-4-8[1m] and the reported claude-haiku-4-5-20251001 helper
- Reported-cost processes: 60

The three initial batches share sanitized workspace hash c782ae4fde649a350ed239766aac4455c93dec0fb29c09da13aaad32b7be54e8. The four post-fix reruns share hash 3f53cd2f8c68071b179efe4c45d6189868c7fb78a6c41c14cae7ddebd231071d. Each manifest contains 157 regular tracked files, excludes 15 tracked paths and three outward symlinks, and records answer_keys_excluded=true plus untracked_paths_excluded=true.

A structured-input audit found no source-checkout path, eval definition, expected output/assertion, VALIDATION.md, audit-test, tmp/generated-metadata path, or path escape. Three controller cases attempted a results glob inside their sanitized workspace; all returned no files. One Bash request was rejected before execution because Bash was not enabled. No excluded artifact was read or returned.

Run the offline runner tests with:

~~~bash
python .agents/skills/bionemo-phage-design/scripts/tests/test_run_skill_evals.py
~~~

## Scientific forward checks

| Scenario | Outcome |
| --- | --- |
| Slurm intake with cross-project SFT reuse and full-genome OOM | PASS — explicit reuse approval, complete SFT/prompt lineage, traceable scripts, due-gated polling, preserved effective token batch, and no silent genome truncation |
| Jumbo-phage collection discovery | PASS — clean-context evaluation found a qualifying current primary paper, exact versioned repository payload, and more than 10,000 verified biological records without relying on a hard-coded identity |
| Jumbo-phage essentiality discovery | PASS — clean-context evaluation surfaced decision-relevant primary genome-wide evidence and preserved assay/transfer limitations |
| RL score implementation audit | PASS after fix — exploit analysis, portfolio effects, telemetry, runtime contract, online/final-QC alignment, and deterministic smoke are all retained even when implementation must stop before code |

The target identifiers are intentionally omitted here so this record cannot answer future discovery cases.

## Historical-number cross-check

The [sanitized historical evidence snapshot](../references/historical-evidence.md) retains source locators and SHA-256 values. The ignored source logs are optional checksum-verifiable corroboration, not a fresh-clone dependency.

| Artifact/profile | Result |
| --- | --- |
| Step-190 96-design validation | 50/96 raw full-QC (52.08%); 48/96 after 99%-cluster deduplication (50%) |
| Step-190 offline 1,000-design Arc, architecture removal disabled | 358/1000 (35.80%) |
| Corresponding full branch, architecture removal enabled | 5/1000 (0.50%) |
| Same-shape TP2 96-request smoke | About 68.3 GB/GPU during train generation and 70.9 GB/GPU during validation; not final-run memory telemetry |
| Observation horizon | Monitor recorded step 250 under a configured 500-step ceiling; selected checkpoint and observation stop remain distinct |

The 96-design validation and 1,000-design offline Arc results are different evaluations and must remain separately labeled.

## Limitations

- No training, inference, external-QC pipeline, scheduler/cloud launch, or hardware-fit test was run.
- The suite ran live through Claude. Codex received schema, unit, validator, and 26-case dry-run coverage, not paid/live generation and grading.
- Broader recipe tests were not run because recipe source/config behavior was not changed.
- Public-source discovery reflects the state observed on 2026-07-17 and should be rerun as literature and APIs change.
- This is a validation snapshot, not workflow authority. Changes were not committed at capture time.

## Minimal recheck

1. Run eval validation, the 41 runner unit tests, and both 26-case dry runs.
2. Validate all 10 skills and both plugin manifests.
3. Run the 25 literature tests, manifest check, and byte-identical offline reconstruction.
4. Recheck relative links, forbidden portable-content patterns, historical snapshot hashes, git diff whitespace, scoped status, and core.worktree.
5. For changed behavioral contracts, run affected live cases in a fresh immutable root and audit the sanitized manifest plus structured tool inputs.
