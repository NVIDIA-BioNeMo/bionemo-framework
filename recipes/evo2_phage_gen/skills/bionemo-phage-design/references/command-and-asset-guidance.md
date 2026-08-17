# Command and asset guidance

Resolve current commands from the selected checkout and active environment. Inspect `pyproject.toml`, relevant source/configs/tests, and `<command> --help`; old runlogs and skill examples are clues rather than authoritative syntax. Build the recipe before assuming an entry point is missing.

For external models, databases, or archives, record the provider identifier/version and license, use a provider checksum when supplied, validate the downloaded format/content, and run a small interface smoke test before scaling. Record which release was used; if it changes during the work, rerun affected controls before interpreting comparisons. Do not substitute profiles, consensus data, HTML pages, or similarly named files when downstream science requires raw sequences or another specific payload.

Record the command and tool/data versions that affect interpretation in the project runlog. Keep credentials out of commands and artifacts.
