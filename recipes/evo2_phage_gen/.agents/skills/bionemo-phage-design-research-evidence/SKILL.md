---
name: bionemo-phage-design-research-evidence
description: Use when a phage-design decision needs literature, database, dataset, gene-essentiality, synteny, viability, bootability, host, threshold, or model evidence that is not already pinned locally.
---

# Research Phage Evidence

Produce a compact, auditable evidence packet for a stated decision. Prefer primary sources and distinguish observation, author claim, and inference.

## Establish the question

Write the decision, target phage/taxon and host, evidence needed, acceptable transfer distance, and stopping condition in a research attempt under `research/runs/<attempt>/`. Read the project contract from `../bionemo-phage-design/references/project-contract.md` and keep all writes in this attempt.

For new RL objectives or filters, build a portfolio coverage checklist: viability preservation, bootability proxies, gene essentiality, synteny and its target-specific definition, topology/packaging, desired and undesired directional changes with meaningful positive/negative thresholds, host-specific predictive models, diversity, and final QC. Before delivery, give each axis a decision-table row or mark it unresolved/not applicable with a reason, and state material interactions across the portfolio. Do not assume that a score useful for one phage or design goal transfers to another.

## Search efficiently

1. Discover installed specialist life-science skills first; prefer suitable bioRxiv, NCBI, PMC, repository, or research-router capabilities. Optional examples are OpenAI Life Science Research skills in Codex and Anthropic's [bio-research plugin](https://github.com/anthropics/knowledge-work-plugins) in Claude. Neither is required; if none fit, warn once and use ordinary web search and public APIs.
2. Inspect `../bionemo-phage-design/assets/literature/**/MANIFEST.json`; use checked-in papers when relevant and cite their exact version.
3. Follow [search-protocol.md](references/search-protocol.md). Search from primary paper to Data Availability to repository concept/version to compact metadata to exact file/checksum. Use only a capped byte range or prefix to validate a candidate payload before a justified download stage.
4. Search up to three independent lanes. Stop when stable identifiers and decision-relevant coverage are obtained, or after two successive refinements add no evidence. Expand the budget only with a recorded reason.
5. Triangulate biologically material thresholds. Prefer direct evidence in the target phage and condition, then close relatives, then calibrated transfer evidence; label lower tiers as uncertain.

Do not full-download a dataset merely to identify it. Do not treat search snippets, repository landing pages, predictive scores, expression, conservation, or transposon depletion as universal gene essentiality without stating their limits.

## Deliver the attempt

Write `artifacts/EVIDENCE.md`, `artifacts/SOURCES.yaml`, `artifacts/SEARCH_LOG.jsonl`, and, when relevant, `artifacts/DATASET_CANDIDATES.yaml`. Include citations, exact versions, access dates, licenses, query/API parameters, identifiers, checksums supplied by the source, payload validation, contradictory findings, evidence tier, transfer limits, and unresolved decisions. Finish the standard `OUTPUTS.yaml`, `SUMMARY.md`, and `RUNLOG.md`.

In a read-only or planning-only session, do not write files, but still include a compact planned-artifacts block naming every applicable path above and the exact queries, sources, and run-log fields that would populate it. Read-only execution does not waive the artifact contract.

Recommend a threshold or objective only with its evidence and uncertainty. When evidence is insufficient, propose a calibration experiment rather than a false universal cutoff.
