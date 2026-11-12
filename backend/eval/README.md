# Évaluation de l'extraction des exigences

Hors pytest : ces scripts appellent le vrai LLM (clé dans `.env`), donc le réseau et le quota.

```bash
cd backend
.venv/bin/python eval/eval_requirements.py                                   # modèle du .env
.venv/bin/python eval/eval_requirements.py --model mistral-large-4-0 --model ministral-14b-2512
.venv/bin/python eval/eval_requirements.py --offer eval/offers/hager.txt --json
.venv/bin/python eval/eval_requirements.py --offer eval/offers/sii.txt --model mistral-large-4-0 --runs 3
```

Offres de référence (publiques, sans donnée personnelle) : `hager.txt` (stage ingénieur informatique & IA, 14 exigences attendues) et `sii.txt` (stage IA applicatif, 13 exigences attendues et 3 acceptables ; « Analyse de données et historique » et « Approche centrée utilisateur » sont reconnues par les mots-clés `match`).

Les modèles, et les exécutions d'un même modèle, sont évalués l'un après l'autre ; tous les appels passent par le même limiteur que l'application (`LLM_RPS`, `LLM_CONCURRENCY` du `.env`, nouvelles tentatives sur 429).

## Limites connues

- **« Compréhension des besoins utilisateurs » (SII)** : manquée 2/3 avec le prompt d'extraction v2 (règle d'importance élargie), contre trouvée 3/3 en v1. Savoir-être souhaité, de poids faible dans le score ; non tranché entre variance du modèle et effet de la nouvelle règle (trois exécutions ne suffisent pas à les distinguer).

## Cas d'évaluation

`offers/<nom>.txt` : texte de l'offre (uniquement des offres publiques, sans donnée personnelle).
`offers/<nom>.expected.json` : exigences attendues.

```json
{
  "requirements": [
    {"label": "Java", "category": "compétence technique", "importance": "obligatoire", "match": [["java"]]}
  ],
  "acceptable": [
    {"label": "Claude / API Anthropic", "match": [["claude"], ["anthropic"]]}
  ],
  "must_not_extract": [
    {"label": "Télétravail", "match": [["teletravail"]]}
  ]
}
```

`acceptable` (facultatif) liste les exigences **défendables mais non exigées** : celles qu'un lecteur raisonnable pourrait extraire ou non. Exemple : SII écrit « Tu travailles avec Claude et nos LLM hébergés en interne » ; cela décrit l'environnement du stage, et en faire une exigence se défend aussi. Une acceptable absente n'est **pas manquante** ; une extraite qui lui correspond n'est **pas inventée** et ne compte pas dans les trouvées : le rapport la liste à part (« acceptables trouvées »). C'est le corrigé qui porte l'ambiguïté, pas le prompt : on n'ajuste pas le prompt d'extraction à un cas de test.

**Transparence** : la liste `acceptable` de SII (Claude, LLM hébergés en interne, restitution / reporting des résultats) a été complétée **après** avoir observé des exécutions, pas écrite à l'aveugle avant. Chaque entrée a été relue contre le texte de l'offre : elle n'y figure que parce que l'offre elle-même permet deux lectures (environnement du stage ou exigence ; étape du parcours ou compétence). Une ambiguïté doit venir de l'offre, jamais du désaccord du modèle : si le modèle se trompe sur une formulation claire, c'est une erreur à corriger (dans le prompt, par une règle générale), pas une entrée à ajouter ici. Exemple : « À l'aise en Python », énoncé sans nuance dans le profil, n'est pas une lecture ambiguë ; l'importance « souhaitée » observée venait d'une règle d'importance trop étroite, corrigée par une règle générale (prompt v2). De même, certains mots-clés `match` ont été ajoutés après observation, uniquement quand le libellé extrait reprend la même exigence de l'offre sous une autre forme : SII « Analyse de données et historique », « Approche centrée utilisateur » ; Hager « Capacités d'analyse » (libellé tronqué de « capacités d'analyse et de synthèse »).

`must_not_extract` liste ce qui ne doit jamais devenir une exigence (conditions logistiques : dates, durée, lieu, rémunération, télétravail) : une extraite qui y correspond est comptée comme **interdite**.

## Méthode de rapprochement

Approximative et volontairement simple :

1. Les libellés sont normalisés : minuscules, accents retirés, découpage en mots (`[a-z0-9]+`). « Rétro-ingénierie » donne `retro`, `ingenierie` ; « Bac+5 » donne `bac`, `5`.
2. Une exigence extraite correspond à une attendue si, pour **l'une** des alternatives de `match`, **tous** les mots sont présents dans son libellé (`[["plm"], ["windchill"]]` : « plm » ou « windchill »). Les mots sont comparés entiers : `java` ne correspond pas à « JavaScript ».
3. Sans `match`, repli sur la similarité `difflib.SequenceMatcher` des libellés normalisés, avec un seuil de 0,6.

Le rapport distingue :

- **trouvées** : attendues rapprochées d'au moins une extraite ;
- **manquantes** : attendues sans aucune extraite ;
- **inventées** : extraites sans aucune attendue (pas forcément fausses : à relire, puis à ajouter à l'attendu si elles sont légitimes) ;
- **interdites** : extraites qui correspondent à `must_not_extract` (comptées à part, pas dans les inventées) ;
- **erreurs d'importance / de catégorie** : comparées sur la première extraite rapprochée ;
- **découpées** : une attendue couverte par plusieurs extraites (ex. « initiative » et « autonomie » séparées) ;
- **fusionnées** : une extraite qui couvre plusieurs attendues (ex. « PLM et rétro-ingénierie » en une seule), ce qui enfreint la règle « un seul critère » ;
- **acceptables trouvées** : extraites qui correspondent à la liste `acceptable` (ni manquantes, ni inventées).

Une extraite qui correspond à `must_not_extract` est interdite, même si elle correspond aussi à une acceptable.

## Plusieurs exécutions (`--runs N`)

Le LLM ne répond pas toujours de la même façon, même à température 0 et avec une graine : une seule exécution peut être chanceuse ou malchanceuse. Avec `--runs N`, chaque modèle est exécuté N fois sur la même offre (appels successifs, au rythme de `LLM_RPS`), chaque exécution est comparée au corrigé comme ci-dessus et affichée, puis le rapport les agrège :

- **taux de trouvées** : moyenne et minimum de « trouvées / attendues » sur les exécutions, avec le détail par exécution (ex. `13/13, 12/13, 13/13`) ; le minimum dit ce qu'on obtient dans le moins bon cas ;
- **inventées** et **interdites** : nombre moyen par exécution, puis chaque libellé avec sa fréquence ;
- **manquées au moins une fois** : chaque exigence attendue absente d'au moins une exécution, avec sa fréquence (`Analyse de données de tickets : manquée 1/3`) ; une exigence manquée 3/3 est un vrai trou de l'extraction, une manquée 1/3 relève de la variance ;
- **erreurs d'importance et de catégorie**, **fusions**, **découpages** et **acceptables trouvées** : chacun avec sa fréquence.

Une exécution en erreur (quota, réseau) est signalée et exclue des taux et des fréquences, qui portent sur les exécutions réussies. `--json` donne la synthèse agrégée au format JSON.
