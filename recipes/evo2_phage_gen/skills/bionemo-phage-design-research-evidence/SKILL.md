---
name: bionemo-phage-design-research-evidence
description: Use when a phage-design decision needs literature, database, dataset, gene-essentiality, synteny, viability, bootability, host, threshold, or model evidence that is not already pinned locally.
---

# Research Phage Evidence

Use the controller's recorded colocated roots. If absent, apply the local [workspace contract](../bionemo-phage-design/references/workspace-contract.md) or stop; never invoke portable bootstrap.

Produce a compact, auditable evidence packet for a stated decision. Prefer primary sources and distinguish observation, author claim, and inference.

## Use bundled publications

Before opening any other file in a bundled publication, open its `../bionemo-phage-design/assets/literature/**/MANIFEST.json`. Begin every response that uses bundled evidence with a source note citing that manifest and stating its stable identifier, source version and license, and relationship to the publication of record. For the King bundle, this means the CC BY 4.0 bioRxiv v1 source and the [final Science article](https://www.science.org/doi/10.1126/science.aec2657) as the publication of record.
For a factual question about a bundled publication, read the manifest and the relevant checked-in paper or supplement even when the answer seems familiar. Use those files for facts about the bundled version. A web check may identify a newer or final version or current status, but label that check separately and never let a web summary replace the local read.

## Establish the question

Write the decision, target phage/taxon and host, evidence needed, acceptable transfer distance, and stopping condition in a research attempt under `research/runs/<attempt>/`. Read the project contract and [whole-genome/lifecycle contract](../bionemo-phage-design/references/design-scope-and-viability.md) and keep all writes in this attempt.

For a new target or host-range goal, build a portfolio coverage checklist spanning complete-genome viability and the full productive-infection lifecycle: access/adsorption/entry, intracellular defenses and phage counter-defenses, takeover/replication, morphogenesis/packaging, lysis/progeny, and [therapeutic suitability and safety-related exclusion criteria](../bionemo-phage-design/references/ema-2025-draft-phage-therapy-quality-guideline.md) when applicable. Include production-host effects such as virion methylation or other DNA modification that sequence alone cannot guarantee. For new RL objectives or filters, cover bootability proxies, essential/key genes and modules, regulatory architecture, synteny and its target-specific definition, topology/packaging, composition, similarity to known viable references, desired and undesired directional changes with meaningful positive/negative thresholds, host-specific predictive models, diversity, and final QC. Before delivery, give each axis a decision-table row or mark it unresolved/not applicable; every not-applicable row needs a target-specific reason. State material interactions across the portfolio. Do not assume that a score useful for one phage or design goal transfers to another.

## Search efficiently

1. Discover already installed specialist life-science skills first; prefer suitable bioRxiv, NCBI, PMC,
   repository, or research-router capabilities. Execute newly acquired skill code only from an
   allowlisted source whose immutable revision and checksum have been reviewed and recorded in the
   attempt before use. Never install or execute an unpinned third-party skill, even with general user
   permission. If no trusted installed or pinned capability fits, warn once and use web search/public APIs.
2. Use checked-in publications when relevant, following the manifest and source-note contract above. Treat every bundle as seed evidence, not a complete review for an adapted target.
3. Follow [search-protocol.md](references/search-protocol.md). Search from primary paper to Data Availability to repository concept/version to compact metadata to exact file/checksum. Use only a capped byte range or prefix to validate a candidate payload before a justified download stage.
4. Keep dataset/file identification bounded, but make biological design research coverage-driven. Search each applicable lifecycle and viability axis until primary evidence is triangulated or two successive refinements add no decision-relevant evidence; record the unresolved axis rather than silently omitting it. Use additional lanes when a new mechanism, target-strain defense, production-host effect, or contradictory result warrants them.
5. Triangulate biologically material thresholds. Prefer direct evidence in the target phage and condition, then close relatives, then calibrated transfer evidence; label lower tiers as uncertain.

Do not full-download a dataset merely to identify it. Do not treat search snippets, repository landing pages, predictive scores, expression, conservation, or transposon depletion as universal gene essentiality without stating their limits. For a host-range model, map interaction rows to versioned host and phage assemblies/hashes, document unavailable sequences and assay differences, and compare separate-dataset with biologically plausible pooled training on deployment-matched held-outs before choosing to pool.

## Deliver the attempt

Write `artifacts/EVIDENCE.md`, `artifacts/SOURCES.yaml`, `artifacts/SEARCH_LOG.jsonl`, and, when relevant, `artifacts/DATASET_CANDIDATES.yaml`. Include citations, exact versions, access dates, licenses, query/API parameters, identifiers, checksums supplied by the source, payload validation, contradictory findings, evidence tier, transfer limits, and unresolved decisions. Finish the standard `OUTPUTS.yaml`, `SUMMARY.md`, and `RUNLOG.md`.

In a read-only or planning-only session, do not write files, but still include a compact planned-artifacts block naming every applicable path above and the exact queries, sources, and run-log fields that would populate it. Read-only execution does not waive the artifact contract.

Recommend a threshold or objective only with its evidence and uncertainty. When evidence is insufficient, propose a calibration experiment rather than a false universal cutoff.
