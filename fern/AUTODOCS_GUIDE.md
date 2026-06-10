# Fern library API (`fern docs md generate`)

This repo uses Fern’s **`libraries:`** block in `docs.yml` to generate Python API reference
Markdown from source (same pattern as NeMo Curator).

## Prerequisites

- Node.js and `fern-api` (or `npx fern-api`)
- **GitHub Actions:** org/repo secret **`DOCS_FERN_TOKEN`** (from `fern token` for the NVIDIA Fern org). Used by all Fern workflows below.

## CI workflows (`.github/workflows/`)

| Workflow | Purpose |
|----------|---------|
| `fern-docs-ci.yml` | On PRs touching `fern/**`, runs `fern docs md generate` to validate autodocs. |
| `fern-docs-preview-build.yml` | **Preview (1/2):** uploads `fern/` + PR metadata as an artifact (no secrets; safe on forks). |
| `fern-docs-preview-comment.yml` | **Preview (2/2):** on `workflow_run` success, downloads artifact, runs `fern docs md generate` + `fern generate --docs --preview`, posts PR comment. |
| `fern-docs-preview.yml` | Alternate **preview** via `pull_request_target` + comment (uses `DOCS_FERN_TOKEN`; same as NeMo Curator). You may disable this or the build/comment pair if you want a single preview path. |
| `publish-fern-docs.yml` | On `docs/v*` tags or manual dispatch: `fern docs md generate` then `fern generate --docs` to publish. |

`install-fern-ci` from the convert-to-fern toolkit is optional; this repo mirrors Curator’s workflow set directly.

## Generate locally

From the `fern/` directory:

```bash
cd fern
npx fern-api@latest upgrade   # optional: align CLI with fern.config.json
fern docs md generate
```

Output is written under `fern/product-docs/` (gitignored). The **API Reference** tab in
`versions/latest.yml` and `versions/v2.7.yml` points at
`product-docs/bionemo-core/Full-Library-Reference`.

## Adding another sub-package

1. Add a new top-level key under `libraries:` in `docs.yml` (see `bionemo-core` as a template).
2. Set `subpath` to a **concrete Python package directory** (with `__init__.py`) that Fern/Pyright accept. Namespace-only dirs (PEP 420, no `__init__.py`) at the namespace root (e.g. `src/bionemo`) can fail with `PACKAGE_NOT_FOUND`; prefer `src/bionemo/<submodule>` (e.g. `sub-packages/bionemo-core/src/bionemo/core`).
3. Set `output.path` to a unique folder under `./product-docs/`.
4. Add a matching `- folder: ../product-docs/<name>/Full-Library-Reference` entry under the **api** tab in the version YAML files.

## Preview

```bash
cd fern
fern docs dev
```

Use the **API Reference** tab to browse generated pages after running `fern docs md generate`.

## Jupyter notebooks (`NotebookViewer`)

Static tutorial notebooks are rendered with the **`NotebookViewer`** React component (ported from NeMo Data Designer), optional **Colab** link, and **`fern/styles/notebook-viewer.css`** (registered in `docs.yml`).

1. **Dependencies:** `pygments` (for syntax-highlighted code cells when converting).

2. **Convert** `.ipynb` → JSON + TypeScript module for MDX import:

   ```bash
   cd fern
   python scripts/ipynb-to-fern-json.py ../path/to/notebook.ipynb -o components/notebooks/my-notebook.json
   ```

3. **Use** in MDX:

   ```mdx
   import { NotebookViewer } from "@/components/NotebookViewer";
   import notebook from "@/components/notebooks/my-notebook";

   <NotebookViewer
     notebook={notebook}
     colabUrl="https://colab.research.google.com/github/NVIDIA/bionemo-framework/blob/main/..."
   />
   ```

`experimental.mdx-components` in `docs.yml` already includes `./components`, so `@/components/...` resolves.
