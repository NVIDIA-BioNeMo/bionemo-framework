# Storage planning

Estimate sequence storage from total bases, not genome count alone:

`B = sum_i L_i ~= N * L_mean`

For the current one-token/base, uint8 preparation pipeline, measured retained sizes are about 1.02 bytes/base for split FASTA, 1.13 bytes/base for training-ready tokens and indices, 2.34 bytes/base for a compact reproducible set, and 6.40 bytes/base for the retained preparation workdir. Thus 60 million bases (10,000 genomes averaging 6 kb) require about 68 MB training-ready, 140 MB compact, or 384 MB for retained preparation. These are planning anchors, not universal constants, and exclude transient clustering scratch or dense similarity output.

Record genome count, total bases, p50/p90/p99/p99.5/p99.9/max length, split counts, tokenizer bytes/tokens per base, physical augmentation, retained sequence copies, preprocessing attempts, and a benchmarked clustering scratch factor. Generation artifacts scale with total generated bases and retained generations; one measured RL workload produced about 42 MB per step.

For the current full-state format, use about 91 GB per SFT checkpoint and 78 GB per RL checkpoint. Estimate peak use as sequence representations plus model/QC assets, checkpoints retained by role, one checkpoint write, and transient QC/clustering space. Context length also changes activation memory and compute, so capacity-probe the agreed target-length shape.

Retain the latest resumable, best/nondominated, user-pinned, and selected handoff checkpoints. After durable metrics, generations, hashes, and uploads exist, remove smoke, failed, superseded, and dominated state. Never prune active, incomplete, selected, or uploading state.
