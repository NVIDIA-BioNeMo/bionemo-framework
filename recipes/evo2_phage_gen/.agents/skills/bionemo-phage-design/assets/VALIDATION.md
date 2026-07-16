# Phage-design validation snapshot

> Human audit record, not agent instructions. Agents should use ../SKILL.md and only the references it names. This file is deliberately under assets/ and is not linked from SKILL.md, so normal skill loading does not consume it.

## Snapshot

- Recorded: 2026-07-16 UTC
- Checks run: 2026-07-15 through 2026-07-16 UTC
- Base revision: 99673b047a196352afcbb35e7aa4200127af2616
- Branch: jstjohn/evo2_phage_gen
- Evaluated changes: `.agents/**` and `README.md` in the recipe
- Runtime scope: documentation, skill contracts, literature assets, and two standard-library utilities; no SFT, RL, inference, or GPU job was launched

## Results

| Surface | Result |
| --- | --- |
| Skill structure | PASS — all 10 skill directories passed the skill-creator validator |
| Behavioral eval schema | PASS — 10 BioNeMo-compatible eval files, 27 globally unique exact-field cases |
| Portable eval runner | PASS — 25/25 unit tests across Codex and Claude adapters, a real disposable Claude generation/grading case passed end to end, and both adapters produced 27/27 zero-cost dry-run plans |
| Claude plugin bridge | PASS — installed Claude Code 2.1.211 validates `.agents` as local plugin `evo2-phage-gen`; namespaced non-interactive invocation, structured grading, requested/observed model provenance, cost capture, and disabled CLAUDE.md/auto-memory were verified without sending recipe skills |
| Context size | PASS — SKILL.md entrypoints remained compact at 362–717 words; detail is routed to references |
| Markdown links | PASS — 41 Markdown files checked, 0 broken relative links |
| Historical evidence portability | PASS — checked-in sanitized snapshot separates empirical results from config facts and preserves locators plus source SHA-256 values |
| Literature utility | PASS — 25/25 unit tests |
| Literature manifests | PASS — both checked-in paper bundles verified with 0 errors |
| Offline reproducibility | PASS — offline synchronization reported changed=false for both bundles |
| Official workbook | PASS — media-1.xlsx SHA-256 3cd26d4cca8bc1273a863c4b2304e755635fe0c7bed46308f54029b88f063fc9 |
| Workbook extraction | PASS — 302 rows, 33 columns, 302 unique IDs; repeated check produced identical TSV hash |
| Recipe README | PASS — 210 lines, results-first human guide, and explicit-worktree git diff --check returned clean |
| Result isolation | PASS — `results/` is gitignored from the recipe root |
| Portability scan | PASS — no employee-only W&B destination, local user path, disallowed spreadsheet source, or erroneous 385 claim in runtime skill text/README |
| Blind-answer leakage | PASS — target paper and repository identifiers were absent from the collection, research, and objective-planning runtime skills |
| Independent review | PASS — follow-up review found no remaining Critical or Important findings; prior I1–I4 and M1–M2 were resolved |

## Reproducible behavioral evals

Each owning skill now has `evals/evals.json` using the BioNeMo Agent Toolkit-compatible fields `id`, `prompt`, `expected_output`, `assertions`, `expected_skill`, and `expected_script`. The 10 files contain 27 cases. Scientific discovery cases grade source properties, evidence quality, corpus size, experimental relevance, and transfer limits rather than fixed titles or identifiers, so a stronger future source can pass. Exact historical-number cases remain separate and intentionally fixed.

The standard-library runner follows the [OpenAI skill-eval loop](https://developers.openai.com/blog/eval-skills): prompt, preserved trace/artifacts, deterministic checks, then a fresh structured rubric grade. It does not add harness-specific fields to the portable eval JSON.

From `recipes/evo2_phage_gen`:

~~~bash
# JSON shape, ownership, required fields, and globally unique IDs
python .agents/skills/bionemo-phage-design/scripts/run_skill_evals.py --validate

# Discover IDs before spending model calls
python .agents/skills/bionemo-phage-design/scripts/run_skill_evals.py --list

# Materialize every Codex command and hash without invoking a model
python .agents/skills/bionemo-phage-design/scripts/run_skill_evals.py \
  --dry-run --all \
  --results-dir results/skill-evals-dry-run-$(date -u +%Y%m%dT%H%M%SZ)

# Run one Codex generation plus an independent structured grading pass
python .agents/skills/bionemo-phage-design/scripts/run_skill_evals.py \
  --run \
  --case bionemo-phage-design-implement-rl-objectives-001-denominator-gaming \
  --results-dir results/skill-evals-$(date -u +%Y%m%dT%H%M%SZ)

# Plan all Claude commands using the same portable cases
python .agents/skills/bionemo-phage-design/scripts/run_skill_evals.py \
  --harness claude --dry-run --all \
  --results-dir results/skill-evals-claude-dry-run-$(date -u +%Y%m%dT%H%M%SZ)

# After approving transfer of these skills/prompts to Anthropic, run one Claude case
python .agents/skills/bionemo-phage-design/scripts/run_skill_evals.py \
  --harness claude --run --allow-external-skill-upload \
  --case bionemo-phage-design-implement-rl-objectives-001-denominator-gaming \
  --results-dir results/skill-evals-claude-$(date -u +%Y%m%dT%H%M%SZ)

# Intentionally run either whole suite by adding --all; bare --run refuses
~~~

The runner reserves an immutable result root and prepares every selected case before launching any. Codex uses ephemeral JSON sessions and a fresh schema-constrained grader. Claude loads the same `.agents` tree through the validated local plugin, explicitly invokes the expected namespaced skill, exposes no Bash/Edit/Write tools, loads no user/project/local settings, disables `CLAUDE.md` and auto-memory with recorded environment overrides, uses no-session persistence, and grades in a second process with the target plugin omitted and all skills/tools disabled. Live Claude execution requires `--allow-external-skill-upload` because it transmits recipe-local skill text and prompts to Anthropic.

Neither adapter pins a model by default: each CLI resolves the defaults available to its process. Codex retains its normal config and records its hash. Claude user/project/local settings are intentionally excluded, so its unpinned resolution comes from account, managed, environment, or built-in defaults; a model selected only in those excluded settings is not loaded. Optional `--model` and `--grader-model` overrides support controlled comparisons; do not put a model choice in portable eval JSON. Preflight provenance records requested overrides and setting isolation, while per-case trace summaries/status and root `run-status.json` record the models actually reported by each CLI and Claude's reported cost, including completed failures and skips; unavailable cost is null rather than zero. In the tested environment, Claude's isolated default primary model resolved to `claude-opus-4-8[1m]`; schema-constrained calls also reported a small `claude-haiku-4-5-20251001` helper usage, demonstrating why observed provenance is retained rather than inferred.

Root `run-provenance.json` records runner/harness identity, requested models, permission/setting isolation, the actual Git top-level revision/dirty state, schema/eval hashes, and deterministic instruction-file hashes. Each case preserves `case.json`, `run-plan.json`, raw traces/stderr, `answer.md`, exact-assertion `grade.json`, generation/grading trace summaries, hashes, and status. `--max-budget-usd` is an optional Claude ceiling per subprocess. Use `--sandbox workspace-write` only for a disposable Codex checkout and an eval that genuinely requires mutation. Only the runner may classify a nonzero process as `SKIP`, using allowlisted stderr or structured error evidence; a grader cannot emit skip, and finding no qualifying scientific result is a failure.

Run the runner unit tests without network or a model-service account:

~~~bash
python .agents/skills/bionemo-phage-design/scripts/tests/test_run_skill_evals.py
~~~

## Forward evaluations

| Scenario | Outcome |
| --- | --- |
| Slurm intake with cross-project SFT reuse and full-genome OOM | PASS — required explicit reuse approval, full SFT/prompt lineage, attempt-scoped scripts, due-gated polling, preserved effective token batch, and no silent genome truncation |
| Jumbo-phage collection discovery | PASS — a fresh-context evaluator resolved the intended paper to the exact versioned archive and biological FASTA, rejected a script-only look-alike, and preserved the 10,000/15,000 collection gates |
| Jumbo-phage essentiality discovery | PASS for discoverability — an identifier-free bounded query surfaced the intended primary genome-wide study in the first batched search call. Two delegated attempts were interrupted for latency before returning, so this does not validate harness latency |
| Final contract review | PASS — SFT/RL stopping, provenance, public checkpoints, prokaryotic scope, execution environments, and README terminology were mutually consistent |

The blind target identifiers are intentionally omitted here so this record cannot contaminate future forward tests.

## Historical-number cross-check

These values are preserved in the checked-in [sanitized historical evidence snapshot](../references/historical-evidence.md), which includes source locators and SHA-256 values. The ignored source logs are optional checksum-verifiable corroboration, not a fresh-clone dependency:

| Artifact/profile | Result |
| --- | --- |
| Step-190 96-design validation | 50/96 raw full-QC (52.08%); 48/96 after 99%-cluster deduplication (50%) |
| Step-190 offline 1,000-design Arc, architecture removal disabled | 358/1000 (35.80%) |
| Corresponding full branch, architecture removal enabled | 5/1000 (0.50%) |
| Same-shape TP2 96-request smoke | About 68.3 GB/GPU during train generation and 70.9 GB/GPU during validation; not final-run memory telemetry |
| Observation horizon | Monitor recorded step 250 under a configured 500-step ceiling; selected checkpoint and observation stop remain distinct |

The 96-design validation and 1,000-design offline Arc results are different evaluations and must remain separately labeled.

## Limitations

- No expensive training, inference, external-QC pipeline, cluster submission, cloud launch, or hardware-fit test was run.
- The complete 27-case suite was not run through paid/live Codex or Claude generation and grading. Claude live recipe-skill execution is pending explicit approval to transmit recipe-local skill text and eval prompts to Anthropic. All cases passed schema validation and immutable-root command dry-run; unit/negative tests passed, and a disposable real-Claude case passed generation plus independent structured grading end to end.
- The broader recipe test suite was not run because recipe source/config behavior was not changed.
- Web discovery checks reflect the public state observed on 2026-07-15 and should be rerun when sources or APIs change.
- This is a snapshot, not an authority for workflow, thresholds, or current command syntax.
- Changes were not committed at the time of validation.

## Minimal recheck

After changing the suite:

1. Run `run_skill_evals.py --validate`, its unit tests, and all-case Codex plus Claude `--dry-run` plans; with external-transfer approval, run selected real cases in each affected harness.
2. Run the harness skill validator on each of the 10 skill directories.
3. Run `scripts/tests/test_sync_literature_assets.py` and `scripts/sync_literature_assets.py check --json`.
4. Recheck relative Markdown links, forbidden portable-content patterns, blind-target leakage, README diff whitespace, and scoped git status.
5. Verify the historical snapshot's tracked-config checksums; when ignored source logs are present, verify their checksums too.
