# Genome collection contract

## Bounded identification

Reuse `bionemo-phage-design-research-evidence` and its search limits: at most three lanes, two discovery queries per lane, ten results per query, one cursor page, five detailed fetches per lane, and stop after two no-yield refinements. Once an article identifier is known, follow this chain:

```text
canonical article/version
  -> Data Availability or supplement statement
  -> stable repository concept/accession
  -> exact version
  -> compact file metadata/checksum
  -> capped byte-range or prefix payload validation
```

Compare article authors, title, date, related identifiers, repository description, and file inventory. Prefer archive APIs over scraping rendered pages. Do not download a full archive merely to discover its contents when metadata or a range request can identify it.

## Attempt artifacts

```text
artifacts/
  raw/                     # immutable verified source payloads
  genomes.fasta            # validated collection or manifest of shards
  genomes.tsv              # one row per emitted record
  excluded.tsv
  SOURCE_MANIFEST.yaml
  QUERY_LOG.jsonl
  CHECKSUMS.sha256
```

Large payloads remain here or in a documented durable data path linked from here; never copy them into `.agents/skills/**` or a tracked distribution folder.

`genomes.tsv` includes at least:

- emitted ID and sequence SHA-256;
- source accession and version;
- source record/file ID and local raw file;
- organism/taxon and host evidence;
- molecule topology and completeness status;
- length, GC fraction, ambiguous-base count;
- retrieval date, license, and inclusion rationale.

`SOURCE_MANIFEST.yaml` records selection query and eligibility policy, API/tool versions, pagination, exact URLs or accessions, source and local checksums, expected and observed sizes/MIME/magic, archive members used, and retry history. Keep secrets and signed query parameters out of it.

## Payload checks

Require the expected compression or file magic, at least one FASTA header, nonempty nucleotide sequences, and no obvious HTML/JSON error body. Reject mixed payloads unless the selected archive members are explicitly mapped. Use IUPAC nucleotide validation and report ambiguous symbols; do not silently coerce them.

Count at four checkpoints: source-declared records, parsed sequences, policy-valid sequences, and distinct sequence hashes. The last count governs the 15,000 target and 10,000 stop gate. Leave circular canonicalization and similarity clustering to SFT preparation, but retain source duplicates in the audit trail.

## Reproducibility

Prefer versioned archive records and accession versions. If an upstream query is inherently mutable, capture retrieval time, ordered returned IDs, response hash, and API parameters. On rerun, reuse checksum-valid raw files; do not overwrite a versioned payload with changed bytes. A changed upstream object becomes a new attempt or explicitly versioned artifact.
