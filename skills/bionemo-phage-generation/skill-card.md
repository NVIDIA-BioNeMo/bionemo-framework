## Description

Coordinates an auditable, whole-genome BioNeMo workflow for evidence-grounded phage candidate generation, training, objective design, and safety screening.

This skill is ready for commercial/non-commercial use.

## Owner

NVIDIA

### License/Terms of Use

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt)

## Use Case

For research and engineering teams that need a reproducible whole-genome, or targeted sub-genome phage-design workflow: establish a compatible workspace, plan from the target, host, intended use, and evidence, assemble licensed inputs and leakage-controlled training data, train and select checkpoints, design and calibrate objectives, and generate and screen auditable candidates.

### Deployment Geography for Use

Global

## Requirements / Dependencies

**Requires API Key or External Credential:** Optional

**Credential Type(s):** API key, Cloud Credentials

A Python-capable agent and a compatible BioNeMo Recipes phage-generation package at recipe version 2.5 or later. Compute, storage, models, databases, and command-line tools depend on the approved project and selected implementation. Credentials are optional for supported telemetry or artifact publication; local records remain authoritative when those services are not used or unavailable. Never place secrets in prompts, logs, or outputs.

## Known Risks and Mitigations

### EMA draft context

EMA/CHMP/BWP/1/2024 remains a draft quality guideline for phage therapy medicinal products, with its consultation closed. It is a quality-oriented reference for design screening, not a complete biosafety, clinical safety, efficacy, manufacturing, regulatory compliance, or authorization standard.

- Characterize the complete genome and any modifications; establish strictly lytic behavior and demonstrate absence of lysogeny.
- Screen or risk-assess toxin, virulence, antimicrobial-resistance, lysogeny-associated, and other detrimental genetic factors.
- Justify intended and off-target host range using productive-infection evidence across relevant strains and growth forms, not adsorption alone.
- Evaluate propagation, lysis, progeny release, and potency experimentally.
- Assess generalized transduction risk independently, including for strictly lytic phages.

Sequence analysis cannot establish production-cell-bank quality, impurities, sterility, pyrogenicity, formulation, process validation, or stability.

### CTXφ illustrative failure route

CTXφ illustrates a known indirect route by which a phage can contribute to human disease: this filamentous phage carries the `ctxA` and `ctxB` cholera-toxin genes and can integrate into the *Vibrio cholerae* chromosome through lysogenic conversion, allowing the toxigenic genotype to be inherited by bacterial descendants. This is not direct phage infection of, or replication within, eukaryotic cells. The case illustrates why toxin-cargo and lysogeny safeguards are important.

### Workflow failure modes and safeguards

| Risk                                                                                                           | Mitigation                                                                                                                                                                                                                                                          |
| -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Harmful cargo, lysogeny, or transduction could make a candidate unsafe.                                        | Use separate whole-sequence screens, strictly lytic evidence, hard exclusions, and experimental lifecycle and transduction endpoints.                                                                                                                               |
| An objective increases replication within eukaryotic cells, or work silently narrows from whole-genome design. | Reject the prohibited replication endpoint declaratively; default to complete-genome scope and require explicit approval for material reductions. This is not a blanket ban on non-replicative host-range or entry work, which still requires case-specific review. |
| Adsorption or a single model score is mistaken for productive infection, viability, or potency.                | Cover the full lifecycle, calibrate models for the target domain, apply versioned final QC, and require phenotypic validation.                                                                                                                                      |
| Reward gaming, missing evidence, or unavailable tools produces a misleading pass.                              | Use adversarial fixtures, bounded independent objectives, support telemetry, online/final alignment checks, and `INDETERMINATE` outcomes that block PASS when required evidence or dependencies are missing.                                                        |
| Data leakage, stale calibration, out-of-domain predictors, or mixed lineage biases results.                    | Use cluster-held-out splits, documented data and model versions, uncertainty records, and comparable validation gates.                                                                                                                                              |
| Computational outputs are overstated as biological, therapeutic, clinical, or regulatory conclusions.          | State explicit non-claims and retain wet-lab validation plus expert, biosafety, clinical, and regulatory review.                                                                                                                                                    |

**Execution safeguards:** Preserve existing work, prevent duplicate or destructive launches, retain full-genome context rather than concealing resource failures, protect credentials, and verify publication scope and licensing before transfer.

## Reference(s)

- [Portable skill source](./SKILL.md)
- [EMA draft guideline record](https://www.ema.europa.eu/en/quality-aspects-phage-therapy-medicinal-products)
- [Evo 2 phage-design publication in Science](https://www.science.org/doi/10.1126/science.aec2657)
- [Public CC BY bioRxiv v1 source with linked supplement and data](https://www.biorxiv.org/content/10.1101/2025.09.12.675911v1.full)
- [Waldor and Mekalanos CTXφ primary study](https://pubmed.ncbi.nlm.nih.gov/8658163/)
- [CTXφ genomics review](https://pubmed.ncbi.nlm.nih.gov/31272871/)

## Skill Output

**Output Type(s):** Markdown guidance, Project artifacts, Model checkpoints, Screened sequence sets, Validation reports

**Output Format:** Repository files and environment-specific model/data artifacts

**Output Parameters:** Target, host, intended use, approved scope, evidence, selected model or checkpoint, objective/QC policy, and execution environment

**Other Properties Related to Output:** Documented and reproducible where supported; outputs are not biological or regulatory determinations.

## Evaluation Tasks

Three declarative top-level cases cover bootstrap, preservation of an incompatible checkout with local work, and controller handoff. Fifty-two recipe-package cases cover scientific scope, evidence use, data leakage, training, objective behavior, durable monitoring, resource-aware execution, and safety screening.

## Evaluation Results

On 2026-08-18, 428 recipe and recipe-local skill tests and 3 repository-level skill tests passed; six recipe tests were skipped. These are behavioral and software-package checks, not execution of every declarative case, biological validation, agent red-teaming, network or product security assessment, or clinical testing.

## Skill Version(s)

Working-tree snapshot; card updated 2026-08-18.

Initial draft rendered with NVIDIA Trustworthy-AI `skill-card-generator` revision `8717620a622922550c1ffc7b1debdac1195bbfd5` on 2026-08-11. This Markdown card is human-owned after generation.

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

(For Release on NVIDIA Platforms Only)

Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns through the [NVIDIA vulnerability disclosure program](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail).
