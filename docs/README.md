# BioNeMo Recipes Documentation

## Installing Docs Dependencies

The docs can be built either with the Docker image defined in `docs/Dockerfile`
or with a local Python environment.

From the repository root, build the docs image:

```bash
docker build -t bionemo-docs -f docs/Dockerfile .
```

For local Python builds, install the MkDocs and mike dependencies from the
`docs` directory:

```bash
cd docs
python -m pip install -r requirements.txt
```

Using a virtual environment is recommended.

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

Then open `http://localhost:8000`.

## Building Static HTML

To build an unversioned static site:

```bash
cd docs
mkdocs build --strict
python scripts/check_internal_links.py site
```

The generated HTML files are written to `docs/site`. The top-level page is
`docs/site/index.html`.

## Building Versioned Docs With mike

BioNeMo Recipes uses [mike](https://github.com/jimporter/mike) with Material
for MkDocs to manage versioned documentation. Mike builds one documentation
version at a time and writes the rendered site to a dedicated deployment
branch.

The canonical deployment branch for released versioned docs is `nvidia-docs`.
Deploy from a release tag instead of an arbitrary branch tip so the generated
docs match the released source exactly. The source tag and the published docs
version do not need to be identical: for example, build from source tag
`v3.0.0` and publish it as docs version `3.0`.

Use only the numeric version identifier for the published docs folder, such as
`3.0` instead of `v3.0`.

From the repository root, create a temporary worktree at the release tag:

```bash
git fetch upstream --tags
git worktree add --detach /tmp/bionemo-recipes-docs-v3.0.0 v3.0.0
```

Mike creates commits on the deployment branch, so make sure Git has a committer
identity configured before deploying. If using Docker, repo-local Git config is
the most reliable because the container may not see your host global Git config.

```bash
git config --local user.name
git config --local user.email
```

If either command is empty, set the missing local value before running
`mike deploy`.

If using a local Python environment, install the requirements in that worktree:

```bash
cd /tmp/bionemo-recipes-docs-v3.0.0/docs
python -m pip install -r requirements.txt
```

Then deploy that tagged source tree to the versioned docs branch:

```bash
mike deploy 3.0 latest \
  --title "3.0" \
  --update-aliases \
  --alias-type copy \
  --remote upstream \
  --branch nvidia-docs
```

If using Docker, run the same mike command through the docs image. The image
defaults to the MkDocs entrypoint, so use `--entrypoint mike` for versioned
docs commands:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
DOCS_WORKTREE=/tmp/bionemo-recipes-docs-v3.0.0

docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  -v "${REPO_ROOT}:${REPO_ROOT}" \
  -v "${DOCS_WORKTREE}:${DOCS_WORKTREE}" \
  -w "${DOCS_WORKTREE}/docs" \
  --entrypoint mike \
  bionemo-docs:latest deploy 3.0 latest \
    --title "3.0" \
    --update-aliases \
    --alias-type copy \
    --remote upstream \
    --branch nvidia-docs
```

This creates a `3.0/` folder in the deployment branch and updates the `latest`
alias to point at that version. `--alias-type copy` is recommended for static
object stores because it creates real files for aliases instead of symlinks.

Set the site root to redirect to the current release:

```bash
mike set-default latest \
  --remote upstream \
  --branch nvidia-docs
```

Or with Docker:

```bash
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  -v "${REPO_ROOT}:${REPO_ROOT}" \
  -v "${DOCS_WORKTREE}:${DOCS_WORKTREE}" \
  -w "${DOCS_WORKTREE}/docs" \
  --entrypoint mike \
  bionemo-docs:latest set-default latest \
    --remote upstream \
    --branch nvidia-docs
```

To preview the versioned site locally:

```bash
mike serve --remote upstream --branch nvidia-docs
```

Or with Docker:

```bash
docker run --rm -it -p 8000:8000 \
  --user "$(id -u):$(id -g)" \
  -v "${REPO_ROOT}:${REPO_ROOT}" \
  -v "${DOCS_WORKTREE}:${DOCS_WORKTREE}" \
  -w "${DOCS_WORKTREE}/docs" \
  --entrypoint mike \
  bionemo-docs:latest serve \
    --dev-addr 0.0.0.0:8000 \
    --remote upstream \
    --branch nvidia-docs
```

Then open `http://localhost:8000`.

To inspect the generated files, check out the deployment branch in a temporary
worktree:

```bash
git worktree add /tmp/bionemo-recipes-versioned-docs nvidia-docs
find /tmp/bionemo-recipes-versioned-docs -maxdepth 2 -type f | sort
cat /tmp/bionemo-recipes-versioned-docs/versions.json
```

The deployment branch should contain a layout like:

```text
index.html
versions.json
3.0/
latest/
```

Remove the temporary worktree when finished:

```bash
git worktree remove /tmp/bionemo-recipes-versioned-docs
git worktree remove /tmp/bionemo-recipes-docs-v3.0.0
```

## Publishing Versioned Docs

After reviewing the generated branch, push it if your publication flow expects
the branch to exist on the remote:

```bash
git push upstream nvidia-docs
```

For an object-store based publication flow, sync the contents of the generated
branch to the target prefix:

```bash
git worktree add /tmp/bionemo-recipes-versioned-docs nvidia-docs
aws s3 sync /tmp/bionemo-recipes-versioned-docs/ s3://BUCKET/PREFIX/ --delete
git worktree remove /tmp/bionemo-recipes-versioned-docs
```

Upload the contents of the generated branch, not the `docs/site` directory from
a plain `mkdocs build`, when publishing versioned docs.

If your publication flow expects an archive, package the generated branch
contents after reviewing them:

```bash
git archive --format=zip \
  --output /tmp/bionemo-recipes-versioned-docs.zip \
  nvidia-docs
```

## Adding Docs

Model and recipe documentation should live beside the code in `models/`, `recipes/`, or `interpretability/`. The docs build imports README files, examples, notebooks, and assets from those directories using `docs/scripts/gen_ref_pages.py`.

## Notebook Rendering

To hide notebook cells from rendered MkDocs HTML, add a `remove-cell` tag to the cell metadata. Use `remove-output` to hide outputs while keeping inputs visible.
