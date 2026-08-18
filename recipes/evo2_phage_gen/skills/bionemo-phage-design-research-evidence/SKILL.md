---
name: bionemo-phage-design-research-evidence
description: Use when a phage-design decision needs literature, database, dataset, gene-essentiality, synteny, viability, bootability, host, threshold, or model evidence.
metadata:
  author: NVIDIA <bionemofeedback@nvidia.com>
---

# Research Phage Evidence

Work inside the recipe and result roots selected by the controller. Prefer primary sources and distinguish measured results, author claims, and your inference.

Use the available background sources as needed. The public King phage-generation paper and supplement describe the published experiment, the Black preprint discusses design efficiency and novelty, and the local EMA transcription records a historical draft phage-therapy quality guideline. If an answer is neither common knowledge nor already supported by this skill or its auxiliary notes, consult the relevant full paper, supplement, or guideline rather than guessing. Distinguish historical versions from later publications and current guidance.

For a new target, host, or objective, gather enough evidence to cover complete-genome viability and the productive-infection lifecycle: adsorption and entry, defenses and counter-defenses, replication, morphogenesis and packaging, lysis and progeny, production-host effects, and intended-use safety. Also consider essential/key genes, regulatory architecture, synteny, topology, composition, similarity to viable references, host models, diversity, and final experimental validation. Mark unsupported or non-applicable axes instead of inventing evidence.

Use available life-science skills, public databases, and web sources as useful, following the concise [research notes](references/research-notes.md). Do not treat a predictive score, conservation, expression, or transposon depletion as universal essentiality.

For project work, write a concise `artifacts/EVIDENCE.md` with citations, relevant source versions, methods/results, contradictions, transfer limits, and unresolved decisions, then note the decision in `RUNLOG.md`. For a read-only question, simply answer with the evidence needed for that question.
