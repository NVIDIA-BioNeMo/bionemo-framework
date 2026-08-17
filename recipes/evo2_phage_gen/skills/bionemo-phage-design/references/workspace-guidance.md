# Workspace guidance

Use the checkout and `recipes/evo2_phage_gen` root selected by the portable skill or user. Keep generated project data beneath that recipe's `results/` directory, not beside installed skill files.

Use commands, configs, source, and recipe-local skills from the same checkout. Do not substitute a similarly named recipe or infer the repository root from a globally installed skill path.

When changing source, follow the checkout's repository instructions and preserve unrelated user changes. Recipe-local skills and tests stay owned by this recipe.
