# Project lab notebook

Keep one project directory beneath the selected recipe's `results/` directory.

At the project root, use:

- `PROJECT.yaml` for the target, host, intended use, mode, and current stage;
- `planning/DESIGN_SPEC.yaml` for whole-genome or explicitly narrowed design scope;
- `SUMMARY.md` for current scientific results and next steps; and
- `RUNLOG.md` as the electronic lab notebook.

For each stage, keep its main inputs, resolved settings, outputs, and a concise summary together. In the runlog, record dated commands, software/data versions where they affect interpretation, random seeds and sampling settings, job/checkpoint locations, key metrics, failures, resumes, and reasons for scientific decisions. Avoid dumping credentials, entire environments, repetitive polls, or metadata that does not help reproduce or interpret the experiment.
