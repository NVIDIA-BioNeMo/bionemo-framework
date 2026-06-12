/**
 * Build a human-readable label for a genomic region example.
 * Expects an object with sequence_id, start, end fields. Falls back
 * gracefully if any of those are missing.
 */
export function getRegionLabel(example) {
  if (!example) return ''
  const sid = example.sequence_id || example.protein_id || ''
  if (example.start != null && example.end != null) {
    const range = `${example.start}-${example.end}`
    return sid ? `${sid}:${range}` : range
  }
  return sid
}

/**
 * Parse a DNA sequence into an array of single-base tokens.
 * No codon framing — each base is rendered independently.
 */
export function parseBases(sequence) {
  if (!sequence) return []
  return sequence.split('')
}
