# Project lab notebook

Keep one project directory beneath the selected recipe's `results/` directory.

At the project root, keep `SUMMARY.md` for the target, intended use, design scope, current scientific results, and next steps, plus `RUNLOG.md` as the electronic lab notebook.

For each stage, keep its main inputs, resolved settings, outputs, and a concise summary together. In the runlog, record dated commands, software/data versions where they affect interpretation, random seeds and sampling settings, job/checkpoint locations, key metrics, failures, resumes, and reasons for scientific decisions. Record scientific settings and relevant identifiers or releases. Avoid credentials and repetitive polls.

When RL objectives are planned or changed, keep `artifacts/RL_OBJECTIVES.yaml` and the scientist-facing `artifacts/RL_SCORE_DEFINITIONS.md` with the run. The planning or implementation agent writes these artifacts; the fixed E2E script does not.
