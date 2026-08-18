# Whole-genome design and lifecycle viability

## Keep whole genomes as the default

Treat each generated candidate as a complete, coherent phage genome and keep the whole genome in
the mutable design space by default. Similarity to a reference, preservation of synteny, a dominant
host-range objective, a named receptor-binding protein, or omission of a diversification metric does
not authorize a fixed backbone, locus-only edit, tail-fiber-only edit, or another restricted search
space.

Record the intended use, complete-genome output, whole-genome mutable scope, protected traits, and any approved scope reduction in the project summary and runlog.

Treat a restricted region, module, edit count, or fixed backbone as a biologically material scope
reduction. In interactive mode, show the whole-genome alternative, expected benefit, lifecycle blind
spots, and lost discovery potential, then obtain explicit user approval before adopting it. In batch
mode, proceed only when a durable user decision already authorizes the reduction; otherwise leave it
blocked. If the user explicitly requests a regional edit, honor it while still evaluating complete
genome integrity and downstream lifecycle effects.

Use preservation requirements as measured rewards, filters, or review criteria rather than as hidden
edit masks. Target-similarity conditioning steers a model; it does not reduce the design scope.

Reject a requested objective or workflow whose endpoint is increased phage replication within
eukaryotic cells; treat it as prohibited rather than a tunable penalty. This is not a blanket ban on
non-replicative eukaryotic entry or host-range work. Route that work through case-specific intended-use,
evidence, and safety review before planning it.

Keep sequence eligibility source-neutral. Source records establish identity; a source label alone is not evidence of
biological eligibility. RefSeq, GenBank, phage databases, metagenomic collections, local
experiments, and model-generated candidates may all proceed when the applicable sequence, host, and
safety evidence is present. Generated candidates inherit the approved target replication-host evidence
from the recorded project scope; never fail them merely because they are generated. Missing evidence may remain
indeterminate for a decision that actually needs it, but an origin label alone is neither a pass nor a
failure.

## Analyze host range across the infection lifecycle

For an adapted host-range goal, create a coverage table rather than equating host range with
adsorption. Review the target phage family, target strain, production/rebooting host, assay conditions,
and intended therapeutic setting across these axes:

- access, adsorption, receptor recognition, capsule or biofilm barriers, and genome delivery;
- restriction-modification, CRISPR-Cas, BREX, DISARM, abortive infection, superinfection exclusion,
  and other target-strain defenses plus plausible phage counter-defenses;
- early takeover, transcription, translation, replication, nucleotide metabolism, and host-factor
  dependencies;
- morphogenesis, structural compatibility, genome packaging, termini/topology, and size limits;
- holin/endolysin/spanin or family-appropriate lysis, progeny production, burst and latent-period
  behavior, and productive infection rather than adsorption alone;
- for therapeutic work, lytic lifestyle, off-target host range, transduction risk, and exclusion of
  lysogeny, toxin, virulence, antimicrobial-resistance, and other harmful cargo.

## Apply intended-use therapeutic guardrails

First record the intended use. Unless the user clearly states another use, provisionally classify an
adapted design as therapeutic and expose that assumption for revision. For a therapeutic design,
use the cleaned local [EMA draft guideline on quality aspects of phage therapy medicinal
products](ema-2025-draft-phage-therapy-quality-guideline.md) as the detailed quality reference and
verify its current status. Translate only design-relevant expectations into separate, traceable
checks:

- characterize the complete genome, topology, GC content, open reading frames, taxonomy, and closest
  relatives; verify the identity and integrity of intended modifications;
- require evidence of strictly lytic behavior and screen or risk-assess antimicrobial-resistance
  determinants, toxins, virulence factors, temperate/lysogeny-related modules, and other detrimental
  genetic factors;
- justify desired and off-target host range with productive-infection evidence across relevant
  strains and growth forms, including biofilm only when claimed;
- assess generalised-transduction risk independently, because a strictly lytic phenotype and a clean
  cargo screen do not establish its absence; and
- verify potency-relevant propagation, lysis, and progeny release experimentally rather than
  inferring [therapeutic suitability](ema-2025-draft-phage-therapy-quality-guideline.md) from
  adsorption or sequence scores alone.

For adapted-design work with therapeutic intended use, emit a separate online RL component for each
applicable design-relevant check that has a defensible measurable proxy. If an item cannot be scored
from generated sequence or available models, record why and retain it as final hard QC or experimental
validation rather than pretending the RL score covers it. The current PhiX174 case-study replication
adds applicable safety components to its existing customized filter profile by default; preserve the
historical component set and report added components separately rather than comparing changed
aggregates directly.

Map each applicable item to an online objective, design-time hard QC, experimental characterization,
or unresolved work; do not collapse them into one learned “safety” score or one multiplicative online
gate. Measure components on independent denominators and calibrate them on the reference plus baseline
SFT generations. Use monotonic partial credit where biologically defensible. When an enabled component
is missing or fixed at zero, diagnose its runtime, support, and proxy; do not silently remove the
criterion, weaken final QC, or abandon the therapeutic goal merely to restore aggregate reward.

Product-manufacturing controls such as purity, sterility, residual host material, formulation, and
container closure are not sequence-design constraints; route them separately only when the project
includes production or product development. The EMA draft is quality guidance, not a complete
biological-design, biosafety, clinical-safety, or efficacy standard.

For an explicitly non-therapeutic project, perform a brief intended-use and applicability review.
Keep the lifecycle, viability, and biosafety checks justified by that use, and mark therapy-specific
items not applicable with a reason rather than applying or dropping the therapeutic set wholesale.

For every applicable axis, record evidence, implicated genes/modules/motifs or host factors, candidate
measurements, positive and negative controls, failure mode, uncertainty, and whether it belongs in an
online reward, final hard QC, experimental validation, or unresolved research. Mark an axis not
applicable only with a phage- and host-specific reason. Do not turn this list into universal required
genes or thresholds.

Treat DNA modification separately from nucleotide sequence. Review the target strain's restriction
motifs and modification-dependent defenses, phage-encoded base modification or anti-restriction
mechanisms, and the methylation/modification state conferred by the production host. Sequence motifs
may be designable; a sequence-only model cannot guarantee the physical epigenetic state of a virion.

## Use a portfolio of viability evidence

Combine target-specific signals instead of treating one proxy as viability proof. Consider:

- complete genome/topology/termini, length and packaging compatibility;
- key-gene and module presence, intact reading frames, copy number, overlaps, regulatory elements,
  order/orientation, and synteny at an evidence-appropriate resolution;
- GC, codon and oligonucleotide composition, homopolymers, restriction motifs, and other relevant
  sequence-composition envelopes relative to the reference and viable relatives;
- coverage-aware similarity to a versioned set of known viable phages, with explicit lower/upper
  bounds when needed to avoid both implausible novelty and copying a natural solution;
- calibrated genome-model likelihood, structure/function predictions, host-range predictions, and
  independent whole-genome QC;
- annotations of unknown genes and uncertain functions, which remain unknown rather than dispensable.

Use similarity to known viable sequences as an auxiliary signal, not a hidden requirement to copy one
reference. Report the reference set, clustering, identity and coverage semantics, topology handling,
training-data overlap, and why the allowed band supports the goal. Preserve multiple nondominated
solutions when evidence does not justify one scalar trade-off.

## Calibrate host-range models without score chasing

Before training or reusing a host-range model, map every interaction row to identified versions of the host and phage
assemblies. Report missing assemblies, conflicting labels, assay/batch differences,
duplicates, class balance, and the population removed by sequence availability. Do not silently impute
genome features or claim that the filtered matrix represents the original matrix.

Do not assume that pooling interaction matrices improves the deployment model. Harmonize labels and
assays, prevent study/sequence leakage, and compare preregistered separate-dataset, same-taxon pooled,
and other biologically plausible models on deployment-matched held-out phages and hosts. Select pooling
only when it improves calibrated performance or coverage beyond uncertainty without creating a
shortcut; retain dataset indicators and per-dataset metrics.

Choose the operating threshold on held-out, deployment-relevant positives and negatives according to
the experimental false-positive/false-negative trade-off. Record calibration, uncertainty, OOD and
similarity diagnostics, and performance by host and phage lineage. For an increasing model score `s`,
a default anti-score-chasing reward is:

`reward = clip((s - baseline) / (target_threshold - baseline), 0, 1)`

Require `target_threshold > baseline`. Set the calibrated negative/baseline anchor to 0 and the
accepted operating threshold to 1, so higher scores remain 1. Use another monotonic saturating
transform only with the same documented anchors. Keep the hard final threshold and independent
phenotypic validation; do not reward confidence above the point needed for the decision.

Specify the desired host-range vector: gain on the target host, retention or loss on original hosts,
and avoidance of disfavored hosts. A strong whole-genome predictor may integrate several lifecycle
mechanisms, but it does not waive the lifecycle coverage table, viability portfolio, or model-specific
gaming and applicability checks.

## Starting evidence, not a substitute for target research

Use current versions of broad host-range reviews such as [Phage host range: determinants, dynamics
and applications](https://pubmed.ncbi.nlm.nih.gov/42026225/), defense-system reviews such as
[Bacterial defense systems against phages](https://pmc.ncbi.nlm.nih.gov/articles/PMC11676413/), and
therapy-quality guidance such as the cleaned local [EMA draft guideline on quality aspects of phage
therapy medicinal products](ema-2025-draft-phage-therapy-quality-guideline.md) to seed the
coverage table. For a whole-genome interaction model, [GenoPHI](https://github.com/Noonanav/GenoPHI)
and its [versioned preprint](https://www.biorxiv.org/content/10.1101/2025.11.15.688630v2) illustrate
relevant model and dataset questions. These sources do not replace target-phage, target-strain,
production-host, and assay-specific primary evidence; resolve current versions and search each
unresolved axis.
