// Mots ajoutés par une reformulation : diff mot à mot entre la puce d'origine et la puce
// proposée (plus longue sous-suite commune), pour les surligner.

export interface DiffPart {
  text: string
  /** Mot absent de la puce d'origine (espaces jamais marqués). */
  added: boolean
}

// Ponctuation ignorée pour comparer deux mots (« GitLab. » = « GitLab »).
const EDGE_PUNCTUATION = /^[\s.,;:!?()«»"“”'’…]+|[\s.,;:!?()«»"“”'’…]+$/g

function key(word: string): string {
  return word.normalize('NFC').toLocaleLowerCase('fr-FR').replace(EDGE_PUNCTUATION, '')
}

/** Parties de `after` dans l'ordre (mots et espaces), chaque mot marqué ajouté ou non. */
export function wordDiff(before: string, after: string): DiffPart[] {
  const tokens = after.split(/(\s+)/).filter((t) => t !== '')
  const words = tokens.filter((t) => !/^\s+$/.test(t))
  const original = before.split(/\s+/).filter(Boolean).map(key)
  const target = words.map(key)

  // Table de la plus longue sous-suite commune (puces courtes : coût négligeable), à plat.
  const width = target.length + 1
  const lcs = new Array<number>((original.length + 1) * width).fill(0)
  const at = (i: number, j: number): number => lcs[i * width + j] ?? 0
  for (let i = original.length - 1; i >= 0; i--) {
    for (let j = target.length - 1; j >= 0; j--) {
      lcs[i * width + j] = original[i] === target[j] ? at(i + 1, j + 1) + 1 : Math.max(at(i + 1, j), at(i, j + 1))
    }
  }
  const kept = new Set<number>()
  let i = 0
  let j = 0
  while (i < original.length && j < target.length) {
    if (original[i] === target[j]) {
      kept.add(j)
      i++
      j++
    } else if (at(i + 1, j) >= at(i, j + 1)) {
      i++
    } else {
      j++
    }
  }

  let wordIndex = 0
  return tokens.map((text) => {
    if (/^\s+$/.test(text)) return { text, added: false }
    // Un mot réduit à de la ponctuation (« : », « ; ») n'est pas surligné.
    const added = !kept.has(wordIndex) && key(text) !== ''
    wordIndex++
    return { text, added }
  })
}

/** Mots ajoutés seulement (pour les tests et les lecteurs d'écran). */
export function addedWords(before: string, after: string): string[] {
  return wordDiff(before, after)
    .filter((part) => part.added)
    .map((part) => part.text)
}
