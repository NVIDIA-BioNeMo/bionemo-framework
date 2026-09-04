---
name: bionemo-phage-design-calibrate-rl-sampling
description: Use after selecting an Evo 2 phage SFT checkpoint and defining RL objectives to calibrate prompt serialization, temperature, prefix-length distribution, and fixed validation sampling.
metadata:
  author: NVIDIA <bionemofeedback@nvidia.com>
---

# Calibrate Phage RL Sampling

Work inside the recipe and result roots selected by the controller. Calibrate the selected SFT checkpoint rather than copying paper or earlier-run settings.

Reconstruct the actual SFT prompt serialization and tokenization, including conditioning, orientation, wrappers, BOS/EOS, padding/masking, and continuation boundary. Use only cues the SFT model saw.

Sweep a reasonable range of temperatures and prefix lengths with paired seeds and enough samples to compare uncertainty. Materialize the selected top-k/top-p, completion length, and prompt mixture in the commands; when both filters are nonzero, Evo 2 applies temperature, then top-k, then top-p. Treat every prompt base as fixed rather than designed: consume the objective plan's exclusions for genes or regions intended to change and reject circular prompts—including origin-wrapping intervals—that overlap any intended-to-change bases. For a confirmed circular genome, compare alternate rotations and prefer anchors with lower overlap-aware CDS occupancy and fewer annotated regulatory, origin, promoter, or terminator features. Record both union CDS coverage and summed CDS occupancy; the latter may biologically exceed 100% when genes overlap. Absence of an annotation is not proof that an interval lacks an unannotated regulatory element. If neutral anchors never produce a required ORF that should be preserved, calibrate a minimal start-seeded stratum, document its fixed bases, and retain an unseeded stratum; never seed a locus intended to change.

Use verified termini class, not linearity: headful/pac genomes may rotate; fixed-end genomes may not. For fixed-end dsDNA, use the objective plan's molecular form and reconstruct biological termini rather than treating a collapsed deposit join as the origin. Then calibrate both strand-equivalent terminal prompts—the forward left terminus and reverse-complement left terminus (the right terminus read backwards)—never an internal rotation; packaged-DTR prompts begin in the repeat and its reverse complement.

Set the decoder and calibration ceilings from the reward curve, not its full-credit boundary. When overlength outputs receive declining partial credit, let generation reach meaningfully into that region while keeping the full prompt-plus-completion length within the model context; retain the separate hard acceptance interval.

Report prompt bases and fraction of the genome fixed. Start with the shortest workable prompt consistent with the selected SFT serialization and measured generation quality; do not scale prompt length linearly with genome size by default, because even a sub-percent prompt can be substantial on a small genome. Retain a longer prompt only with a model- and design-specific rationale supported by calibration evidence. Distribute eligible starts or strand orientations according to termini class and interleave strata so each global rollout batch is representative. When packed dynamic decode supports heterogeneous prompts, pass the full ordered orientation-by-length mixture through one native call without partitioning by length; fixed-shape backends may still require same-length batches. Keep validation independent but comparably stratified. When step metrics oscillate, stratify by prompt position, length, and composition before attributing the pattern to the policy.

Score raw and cluster-deduplicated hard-QC yield, target/lifecycle evidence, complete-genome integrity, copying, diversity, and every enabled objective. For circular prompt mixtures, run the reference rotations through the exact reward/filter environment and require equivalent intrinsic outcomes for metrics declared rotation-invariant; record intentional origin-dependent exceptions. Do not compare per-row cluster-deduplicated representative flags across equivalent rotations: they are set-relative, so instead verify the expected representative count and set-level yield. Audit secondary metrics separately: whole-sequence language-model likelihood depends on the chosen linear origin and must not globally rank mixed-origin designs without an origin-normalized method. A successful wrapper is insufficient: require every configured external measurement used for selection to be available, distinguish a biological no-hit from scorer failure, and diagnose unexplained missing or fixed-zero components rather than dropping them. Use positive and failure controls to confirm each score is measurable.

Run calibration with the same sequence-safety policy, asset manifest, host domain, and confirmed,
versioned host evidence as online RL. Reject selection when an enabled external or safety objective
is neither measured consistently nor exactly documented as biologically inapplicable; explicit
inapplicability remains `INDETERMINATE`, at zero affected-class credit, and hard-QC-ineligible.
Missing configuration, malformed reasons, unexplained `NOT_RUN`, tool failure, or
status/availability disagreement fails closed.

Choose a robust quality-diversity plateau rather than a noisy maximum. Prefer temperature 1.0 when it is practically equivalent, and retain multiple prompt strata only when they improve the frontier. Packed dynamic prefill can mix arbitrary prompt lengths in one microbatch, so the number of length strata need not divide the request batch or GPU count; interleave a near-balanced mixture instead.

Keep calibration samples separate from the fixed RL-validation bank and final rollout. Record prompt construction, seeds, sampling settings, sample counts, score summaries, uncertainty, chosen mixture, and rationale in the stage summary and `RUNLOG.md`.

For the realized PhiX workflow, read the
[example README](../../examples/README.md) before planning or rerunning it. Treat that document as
the source of truth for the current review stop, evidence paths, selection schema and handoff,
completion markers, and resume procedure. When the user delegates selection, make the
evidence-based choice described there and record its rationale; use the bundled default only when
the user explicitly selects it.

The example shell script is a reference implementation, not a mandatory launcher. On a different
GPU or scheduler environment, inspect the available hardware and adapt topology, batch and worker
settings, or the launch method itself while preserving whole-genome context, effective batch,
sampling semantics, validation independence, and the durable selection record. Do not alter an
active run's sampling semantics in place; use a new result root and SFT-anchored RL attempt for a
material change, retaining the earlier run as evidence.
