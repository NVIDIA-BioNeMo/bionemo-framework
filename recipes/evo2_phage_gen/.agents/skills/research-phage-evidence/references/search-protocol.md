# Bounded evidence and data discovery

## Query plan

Extract target terms before searching: phage/virus name and aliases, host/taxon, phenotype or design property, assay or model family, dataset type, approximate year, and known paper identifier. Create no more than three lanes:

1. **Evidence lane:** target + property + assay/method synonyms; prioritize recent primary papers and preprints.
2. **Dataset lane:** target/taxon + genome/dataset + `Data Availability`, accession, repository, archive, or supplement terms.
3. **Transfer lane:** closest taxonomic/functional relatives + property + calibration/threshold/benchmark terms.

Use at most two discovery queries per lane, ten results per query, one API cursor page, and five detailed fetches per lane. Log each query verbatim and why a result was opened. Stop a lane after two no-yield refinements. Prefer title/abstract and identifier endpoints over rendering full pages.

## Resolution sequence

For every promising primary paper:

1. Resolve canonical title, authors, DOI/preprint ID, version, publication date, and license.
2. Read abstract/methods relevant to the decision; then locate the paper's Data Availability, code, and supplement statements.
3. Follow named repository records to the stable concept and exact version. Distinguish a versioned record from similarly titled uploads, scripts, assemblies, and decoy artifacts.
4. Fetch compact repository metadata: creators, description, version/date, related identifiers, file names, sizes, MIME types, checksums, licenses, and links.
5. Select the claimed biological payload by paper-to-record-to-file evidence. Validate a small HTTP range/prefix or archive member listing when supported: FASTA headers/sequence alphabet, compression magic, spreadsheet/ZIP magic, or declared format. Do not infer payload from extension alone.
6. Hand exact identifiers, URLs/API records, expected size/checksum, license, and validation evidence to the download stage.

Avoid broad repository search once a DOI, accession, or related identifier is known. For a recent article, check both published and preprint versions and follow their explicit related records. For an ambiguous repository hit, compare paper authors, title, date, description, file inventory, and related identifiers before selection.

## Evidence extraction

Use a decision table:

| Claim or parameter | Direct result | Context/assay | Evidence tier | Transfer risk | Source/version |
|---|---|---|---|---|---|

Separate:

- measured fitness/bootability from sequence-model likelihood;
- condition-specific fitness-conferring genes from universally essential genes;
- gene presence from intact function;
- conserved order from functionally required synteny;
- classifier scope from claimed biological host range;
- online shaping rewards from final pass/fail evidence.

Capture denominators, controls, confidence intervals, sample size, negative results, assay conditions, and failure modes. If a threshold comes from a model, record model version, training domain, calibration cohort, operating point, and baseline distribution. If no justified threshold exists, specify a target-specific calibration dataset and preregister how the cutoff will be selected.

## `SOURCES.yaml` entry

```yaml
- source_id: stable-local-id
  type: primary-paper
  title: "..."
  identifiers: {doi: null, accession: null, repository_record: null}
  version: "..."
  published: "..."
  accessed: "..."
  license: "..."
  url: "..."
  evidence_tier: direct-target
  supports: []
  limitations: []
  files:
    - {name: "...", size: null, checksum: null, payload_check: "..."}
```

Use stable IDs in `EVIDENCE.md`; keep short quotations within license limits and rely on paraphrase. Record unsuccessful searches in `SEARCH_LOG.jsonl` so future agents do not repeat them.
