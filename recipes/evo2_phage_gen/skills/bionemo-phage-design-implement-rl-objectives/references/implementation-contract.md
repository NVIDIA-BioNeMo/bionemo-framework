# Implementation and test contract

## Repository placement

Locate the recipe that owns the RL entry point. Put reusable functions in its `src/` package and tests in the corresponding `tests/` tree, following neighboring names and fixtures. Preserve existing repository-authorized symlink boundaries, but do not add ad hoc cross-recipe imports or symlinks. For intentionally identical copies, edit the authoritative source, update `SOURCE_TO_DESTINATION_MAP` when the mapping changes, and run `python ci/scripts/check_copied_files.py --fix`.

Generated artifacts belong to the result attempt:

- `resolved_config.yaml`
- `design_spec.yaml` or an immutable path/hash reference to the approved project design spec
- `source_state.json` with commit/diff or content hashes
- `metrics/objective_contract.json`
- `artifacts/SCORE_AUDIT.md` with individual shortcuts, joint counterexamples, mitigations, and fixture expectations
- `logs/tests-red.log`, `logs/tests-green.log`, and `logs/rl-smoke.log`
- `OUTPUTS.yaml` with paths and hashes

## Adversarial score audit

Brainstorm before writing tests or code. Trace every objective from raw model output through parsing, matching/alignment, support and denominator construction, missing/error defaults, clipping, aggregation, hard pass, and final filter. For each stage, construct the easiest output that maximizes score while violating intent. Include empty or omitted elements that default to zero, deletion/truncation that reduces counted obligations, tiny matching subsets that inflate identity, duplicate elements, ambiguous/canonicalized forms, tool failures, non-finite values, and just-inside threshold cases when applicable. Missing or insufficient evidence is not a favorable measured value.

Audit the portfolio separately from its components. Score reference, baseline/random, desired-target, and adversarial designs through the exact aggregate. Check correlated or duplicated terms, weights/scales, opposing directions, discontinuities, OR-branch dominance, and whether A+B drives toward C when the user wants D. Expected objective tension is acceptable; an unintended design that ties or beats the desired design is a contract failure. Add a discriminating reward, hard filter, support gate, calibration, or revised weights and repeat sensitivity/ablation analysis. If tests cannot express the desired ordering, return to objective design rather than coding the scalar.

## Test matrix

Scale extra verification to the objective's biological and integration risk, but keep the following
contract-critical matrix and integration gates. Proportionality may reduce redundant cases; it does
not remove adversarial, boundary, missing-input, online/final-alignment, installed-runtime, or RL-smoke
coverage.

For each scalar reward or filter:

1. baseline/chance maps exactly to `0`;
2. target maps exactly to `1`;
3. representative partial values are in order and in range;
4. below/above-domain inputs clip as declared;
5. NaN, infinity, malformed, absent, and tool-failure inputs receive zero credit and a recorded reason;
6. threshold equality and immediate neighbors match the declared comparator;
7. batch and scalar evaluation agree.

Add invariant/property tests when the domain is combinatorial. For sequences, include empty, truncated, duplicated, circularly rotated, reverse-complemented, ambiguous-base, and over-limit cases as applicable. For synteny, exercise strand, circular origin, overlaps, duplicated homolog groups, insertions/deletions, and both preservation and deliberate-break directions. For `any`, verify every winning branch, no branch, ties, and dominance telemetry; for `all`, verify each isolated failure.

For each enabled therapeutic-quality component, add reference, representative baseline-SFT,
partial-credit or near-pass, target/pass, isolated hard-violation, missing-tool, and insufficient-support
fixtures as applicable. Verify that components are scored independently, sparse support does not zero
unrelated rewards, and final hard exclusions remain unchanged by online shaping.

## Integration gates

- Run an offline fixture through the same callable used online.
- Run final hard QC over the fixture and compare every aligned primitive.
- Measure each enabled component's reference and baseline-SFT support before training; treat an
  unexplained missing or fixed-zero distribution as a preflight fault, not permission to omit it.
- Validate names, order, dtype, device, shape, and reduction against the installed RL runtime.
- Run a tiny deterministic RL smoke that exercises reward calculation, logging, checkpoint writing, and restart metadata.
- Run focused tests plus the nearest relevant suite and formatter/linter configured by the recipe.

Record every command, exit status, package/runtime versions, source hash, config hash, and output hash. RED must fail because behavior is absent—not because a fixture, import, or environment is broken. GREEN must be clean before declaring the objective implemented.

## Review questions

- Does each emitted value retain the stable ID and high-level goal trace?
- Can missing external-tool results, no observations, or too few observations accidentally pass or receive favorable partial credit?
- Can deletion, truncation, duplication, alternate canonical forms, or a tiny matching subset make a component score well for the wrong reason?
- Does the combined score rank desired fixtures above reference, baseline/random, and counterexamples, and do ablations reveal double counting or dominance?
- Are online and final algorithms identical where promised?
- Is every approved operational threshold tied to the exact population and model without being promoted to viability or bootability proof?
- Does the implementation preserve OR children rather than only the max?
- Could sequence canonicalization merge biologically distinct cases or split equivalent circular cases?
- Does the runtime still generate and evaluate complete genomes under the approved mutable scope, and can every scope reduction be traced to an explicit decision?
- Does every applicable, measurable intended-use therapeutic objective remain visible and trainable
  without weakening its final hard or experimental endpoint?
