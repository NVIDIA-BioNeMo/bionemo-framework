# BioNeMo Recipes Documentation

## Previewing The Docs Locally

From the repository root:

```bash
docker build -t bionemo-docs -f docs/Dockerfile .
docker run --rm -it -p 8000:8000 \
  -v ${PWD}/docs:/docs \
  -v ${PWD}/models:/models \
  -v ${PWD}/recipes:/recipes \
  -v ${PWD}/interpretability:/interpretability \
  bionemo-docs:latest
```

Then open `http://0.0.0.0:8000`.

## Adding Docs

Model and recipe documentation should live beside the code in `models/`, `recipes/`, or `interpretability/`. The docs build imports README files, examples, notebooks, and assets from those directories using `docs/scripts/gen_ref_pages.py`.

## Notebook Rendering

To hide notebook cells from rendered MkDocs HTML, add a `remove-cell` tag to the cell metadata. Use `remove-output` to hide outputs while keeping inputs visible.
