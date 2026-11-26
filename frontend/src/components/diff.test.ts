import { describe, expect, it } from 'vitest'
import { addedWords, wordDiff } from './diff'

describe('wordDiff', () => {
  it('surligne seulement les mots ajoutés', () => {
    const before = 'Pipeline de tests et de déploiement lancé à chaque fusion sur la branche principale, sur GitLab.'
    const after = 'Pipeline CI/CD de tests et de déploiement lancé à chaque fusion sur la branche principale, sur GitLab.'

    expect(addedWords(before, after)).toEqual(['CI/CD'])
    expect(wordDiff(before, after).map((p) => p.text).join('')).toBe(after) // texte intact
  })

  it('ignore la casse et la ponctuation collée aux mots', () => {
    expect(addedWords('Maintenance et tests JUnit.', 'Maintenance et tests automatisés JUnit')).toEqual(['automatisés'])
    expect(addedWords('tests JUnit', 'Tests JUnit.')).toEqual([])
  })

  it('un intitulé ajouté en tête, ponctuation seule jamais surlignée', () => {
    const before = 'Classement de 5 000 tickets par catégorie'
    const after = 'Analyse de données de tickets : classement de 5 000 tickets par catégorie'

    expect(addedWords(before, after)).toEqual(['Analyse', 'de', 'données', 'de', 'tickets'])
  })

  it('aucun ajout : rien de surligné', () => {
    expect(addedWords('Même puce.', 'Même puce.')).toEqual([])
  })
})
