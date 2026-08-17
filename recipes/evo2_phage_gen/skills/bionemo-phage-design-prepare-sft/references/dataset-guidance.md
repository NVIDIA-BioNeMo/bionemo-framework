# SFT dataset guidance

Operate on unprefixed full genomes. Group exact biological equivalents, including circular rotations and reverse complements where topology supports them, then cluster before assigning entire clusters to train, validation, or test. Run an independent final leakage check with the same identity/coverage semantics.

For a single clear target, a target-similarity control prefix may be used as a whole-genome conditioning signal. Define target identity, similarity method, bucket edges, and tokenizer-valid control tokens from the observed distribution. Apply the frozen assignment only after deduplication, clustering, and splitting; it is not an edit mask.

Keep the target and its prompt-source neighborhood in training when required by the RL design. Preserve explicit split inputs, conditioning settings, serialization, safety-screen counts, context calculation, and the evidence used to choose the training budget.
