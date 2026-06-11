# bionemo-fw

The BioNeMo Framework (FW): a production grade framework for AI-enabled Drug Discovery.

The `bionemo-fw` Python package contains framework-spanning code under the `bionemo.fw` namespace.
All other namespaces of the BioNeMo Framework (`bionemo.*`) are dependencies of this package.

## Developer Setup

After following the setup specified in the [README](https://github.com/NVIDIA/bionemo-framework/blob/main/README.md),
you may install this project's code in your environment via executing:

```bash
pip install -e .
```

To run unit tests with code coverage, execute:

```bash
pytest -v --cov=bionemo --cov-report=term .
```
