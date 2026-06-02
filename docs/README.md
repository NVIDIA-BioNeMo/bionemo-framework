# BioNeMo Framework Documentation

## Viewing the current documentation on github pages

The documentation should be viewable at [https://nvidia-bionemo.github.io/bionemo-framework/](https://nvidia-bionemo.github.io/bionemo-framework/).

## Previewing the documentation locally

From the repository root:

```bash
# Build the Docker image
docker build -t nvcr.io/nvidian/cvai_bnmo_trng/bionemo-framework-docs -f docs/Dockerfile .

# Run the Docker container
docker run --rm -it -p 8000:8000 \
  -v ${PWD}/docs:/docs \
  -v ${PWD}/sub-packages:/sub-packages \
  -v ${PWD}/bionemo-recipes:/bionemo-recipes \
  nvcr.io/nvidian/cvai_bnmo_trng/bionemo-framework-docs:latest
```

And then navigate to [`http://0.0.0.0:8000`](http://0.0.0.0:8000) on your local
machine.

## Sub-package Documentation

When adding documentation for a new sub-package, ensure it is properly integrated into the documentation site by:

1. Adding an entry to `docs/docs/main/examples/SUMMARY.md` to include it in the Tutorials section
2. Adding an entry to `docs/docs/main/developer-guide/SUMMARY.md` to include it in the Developer Guide section

This ensures the sub-package documentation is properly indexed and accessible through the navigation menu.

The sub-package specific documentation itself must be placed alongside the sub-package code in the `sub-packages/bionemo-<sub-package-name>/` directory:

- `README.md` - A root level file that describes the sub-package and how to use it.
- `examples/` - A directory that contains documentation or examples specific to the sub-package, in the form of `.md` or `.ipynb` files.
- `assets/` - A folder that contains any static assets used in any of the above files, e.g. `.png` files.

When the docs are built, these documentation files will be fetched (via the [scripts/gen_ref_pages.py](./scripts/gen_ref_pages.py) script) for rendering in the main documentation site.

- The `README.md` will be rendered as an individual page in the `User Guide -> Developer Guide -> <sub-package-name>/` section of the documentation site.
- Every file in the `examples/` directory will be rendered as an individual page in the `User Guide -> Tutorials -> <sub-package-name>/` section of the documentation site.

An example sub-package structure is shown below:

```
bionemo-<sub-package-name>/
└── assets/
    ├── example_1.png
├── examples/
│   ├── example_1.md
│   └── example_2.ipynb
├── src/
├── tests/
├── LICENSE
├── pyproject.toml
├── README.md
├── VERSION
```

## Hiding/collapsing `.ipynb` cells

To remove cells from the rendered `mkdocs` html you can add a `remove-cell` tag to the cell. Note that `remove-output` is also an option to hide outputs but not the code cell. Unfortunately
`remove-input` does not seem to be supported.

To collapse jupyter-lab rendered code cells, for example in a `brev.dev` or user ran `jupyter lab` session, you can add a special `jupyter` block to the `metadata` block for that cell in the
json representation of your `.ipynb` file. You can do this in vscode by clicking the `...` above the cell and selecting `Edit cell tags (JSON)`.

A metadata field with both changes, (removed from the rendered docs and collapsed in jupyter) would look like the following:

```json
"metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": [
     "remove-cell"
    ]
   },
```

aliases for these options can be found in the `- mkdocs-jupyter:` section of `mkdocs.yml` in this folder.
