# RL-to-SFT and prompt lineage contract

Unless the user requires fresh training/no reuse, discover SFT candidates in active and configured external result roots. Compare compatibility, validation evidence, format, and availability. If several are plausible, explicitly ask reuse versus new; otherwise do not search for reusable prior-run assets.

Every RL request, manifest, OUTPUTS.yaml, and SUMMARY.md records:

- SFT project/result-root identity, literal stage_name and stage_type, and run/attempt ID;
- checkpoint step, resolved path and/or provider artifact ID/version, and content/manifest hash;
- evidence tying checkpoint path/identity/hash to that named stage;
- base Evo2 provider, public ID/version, format, and hash;
- training dataset and leakage-controlled split-manifest hashes;
- SFT resolved-config/source-state hashes;
- selection metric/evidence, best step, stop step, rationale, and paths to SFT summary/output manifest.

Also record prompt lineage:

- prompt-manifest path and SHA-256;
- source reference-genome ID/hash/topology;
- prompt slicing, rotation/orientation, soft-prefix and token derivation;
- tokenizer identity/version and prompt token counts/length strata;
- generation seed/procedure version and stable prompt IDs;
- fixed validation-generation manifest/hash, seeds, sampling parameters, counts, and filter/tool versions.

Cross-project references are supported. Store portable logical identity plus locally resolved path; do not copy a large checkpoint merely to colocate it. Revalidate hashes before every launch/relaunch.

## RL lineage

Record objective/filter hashes; policy initialization and KL reference independently; resume type fresh, exact-resume, weights-only, or stagewise; optimizer/scheduler/RNG identity for exact resume; prior RL checkpoint and rationale for stagewise; and runtime/config/source/environment hashes.

Exact resume may not alter KL reference, prompt derivation, or fixed validation manifest. A weights-only restart may initialize recorded weights but remains anchored to selected non-RL SFT. A stagewise run is a new experiment, not a resume.
