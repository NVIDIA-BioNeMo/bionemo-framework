/**
 * Text utilities for the Nemotron SAE feature atlas.
 *
 * The codon dashboard worked with DNA codon triplets; this dashboard works with
 * tokenized natural-language text, so the helpers here deal with token strings
 * (rendering whitespace, splitting raw text) instead of codon translation.
 */

/**
 * Render a token string for display, making leading/trailing whitespace and
 * newlines visible so highlighted boundaries are unambiguous.
 *
 * Returns { display, isSpace } where `display` is the printable form and
 * `isSpace` flags pure-whitespace tokens (rendered with a faint glyph).
 */
export function displayToken(token) {
  if (token == null) return { display: '', isSpace: false }
  if (token === '') return { display: '∅', isSpace: true }
  // Newlines -> visible return glyph
  if (/^[\n\r]+$/.test(token)) return { display: '↵', isSpace: true }
  // Pure whitespace -> middle dots
  if (/^\s+$/.test(token)) return { display: '·'.repeat(token.length), isSpace: true }
  // Replace leading space (common in BPE tokenizers, e.g. "▁the" / " the") with a thin marker
  return { display: token.replace(/\n/g, '↵'), isSpace: false }
}

/**
 * Parse a tokens field into an array of token strings.
 * Accepts either an array of strings (preferred) or a JSON-encoded string.
 */
export function parseTokens(tokens) {
  if (!tokens) return []
  if (Array.isArray(tokens)) return tokens
  if (typeof tokens === 'string') {
    try {
      const parsed = JSON.parse(tokens)
      if (Array.isArray(parsed)) return parsed
    } catch {
      // Fall back to whitespace splitting for plain text
      return tokens.split(/(\s+)/).filter(t => t.length > 0)
    }
  }
  return []
}

/**
 * Join tokens back into a readable text snippet (for tooltips / exports).
 */
export function joinTokens(tokens) {
  return parseTokens(tokens).join('')
}
