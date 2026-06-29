/**
 * Extract UniProt accession from a protein ID.
 */
export function getAccession(proteinId) {
  if (proteinId && proteinId.includes('|')) return proteinId.split('|')[1]
  // Extract gene name from mutation IDs like "FLT3_2438_GCC-GCA" or "ABL1_ref"
  if (proteinId && proteinId.includes('_')) return proteinId.split('_')[0]
  return proteinId || ''
}

export function uniprotUrl(accession) {
  // If it looks like a UniProt accession (e.g., P36888), link directly
  if (/^[A-Z][0-9][A-Z0-9]{3}[0-9]$/.test(accession) || /^[A-Z][0-9][A-Z0-9]{3}[0-9]{2}$/.test(accession)) {
    return `https://www.uniprot.org/uniprotkb/${accession}`
  }
  // Otherwise search by gene name
  return `https://www.uniprot.org/uniprotkb?query=${accession}`
}

/**
 * Standard genetic code: DNA codon -> amino acid (1-letter code).
 */
const CODON_TO_AA = {
  'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
  'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
  'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
  'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
  'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
  'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
  'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
  'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
  'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
  'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
  'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
  'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
  'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
  'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
  'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
  'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

/**
 * Translate a DNA codon to its amino acid.
 * Returns '?' for unknown codons.
 */
export function codonToAA(codon) {
  return CODON_TO_AA[codon.toUpperCase()] || '?'
}

/**
 * Parse a codon sequence string into an array of codon triplets.
 * Handles both space-separated ("ATG AAA GCC") and concatenated ("ATGAAAGCC") formats.
 */
export function parseCodons(sequence) {
  if (!sequence) return []
  if (sequence.includes(' ')) {
    return sequence.split(' ').filter(c => c.length > 0)
  }
  // Concatenated: split into triplets
  const codons = []
  for (let i = 0; i < sequence.length; i += 3) {
    codons.push(sequence.slice(i, i + 3))
  }
  return codons
}
