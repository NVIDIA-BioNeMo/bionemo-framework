/**
 * Extract UniProt accession from a SwissProt protein ID or AlphaFold ID.
 * Examples:
 *   ("sp|Q8N1N4|K2C78_HUMAN", "AF-Q8N1N4-F1") -> "Q8N1N4"
 *   ("sp|Q8N1N4|K2C78_HUMAN", null)            -> "Q8N1N4"
 *   ("P12345", null)                            -> "P12345"
 */
export function getAccession(proteinId, alphafoldId) {
  if (alphafoldId) return alphafoldId.replace('AF-', '').replace(/-F\d+$/, '')
  if (proteinId && proteinId.includes('|')) return proteinId.split('|')[1]
  return proteinId || ''
}

export function uniprotUrl(accession) {
  return `https://www.uniprot.org/uniprot/${accession}`
}
