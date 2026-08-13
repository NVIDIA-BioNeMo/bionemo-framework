# Bounded evidence and data discovery

## Bounded identification plan

For resolving a paper, accession, repository record, or exact payload, extract target terms before searching: phage/virus name and aliases, host/taxon, phenotype or design property, assay or model family, dataset type, approximate year, and known paper identifier. Create no more than three lanes:

1. **Evidence lane:** target + property + assay/method synonyms; prioritize recent primary papers and preprints.
2. **Dataset lane:** target/taxon + genome/dataset + `Data Availability`, accession, repository, archive, or supplement terms.
3. **Transfer lane:** closest taxonomic/functional relatives + property + calibration/threshold/benchmark terms.

Use at most two discovery queries per lane, ten results per query, one API cursor page, and five detailed fetches per lane. Log each query verbatim and why a result was opened. Stop a lane after two no-yield refinements. Prefer title/abstract and identifier endpoints over rendering full pages.

These limits govern identification and payload resolution, not the biological evidence review for an
adapted design.

## Coverage-driven design evidence

Start from the project's whole-genome/lifecycle coverage table. Open a focused evidence lane for each
applicable unresolved axis: adsorption and entry; strain defenses and phage counter-defense;
takeover/replication; morphogenesis/packaging; lysis/progeny; [therapeutic suitability and
safety-related exclusion criteria](../../bionemo-phage-design/references/ema-2025-draft-phage-therapy-quality-guideline.md)
when applicable; essential/key genes,
regulatory architecture, synteny, topology, composition, viable-reference similarity, predictive
models, and production-host-dependent DNA modification. Combine axes only when the same sources and
decision genuinely cover them.

Prefer direct target-phage/target-strain evidence, then same-family or same-host-system evidence, then
calibrated transfer. For each axis, stop when decision-relevant primary evidence is triangulated or
two successive refinements add no evidence; label the gap, uncertainty, and proposed experiment.
Contradictory findings, a newly discovered defense mechanism, or an uncharacterized production-host
effect justifies another logged lane. Do not impose a global three-lane ceiling on this review.

For interaction-based host models, add a data-lineage lane that maps every phenotype row to versioned
host and phage assemblies and hashes; records missing sequences, duplicate or conflicting labels,
assay/batch differences and class balance; and separates interaction rows from unique biological
entities. Predeclare deployment-matched held-out-phage, held-out-host, and combined holdout tests as
applicable. Compare per-dataset and harmonized same-taxon pooled models; do not assume pooling helps.

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
| ------------------ | ------------- | ------------- | ------------- | ------------- | -------------- |

Separate:

- measured fitness/bootability from sequence-model likelihood;
- condition-specific fitness-conferring genes from universally essential genes;
- gene presence from intact function;
- conserved order from functionally required synteny;
- classifier scope from claimed biological host range;
- adsorption or receptor binding from productive infection and lysis;
- sequence motifs from the physical methylation or modification state of produced virions;
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
