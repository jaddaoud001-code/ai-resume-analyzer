# Évaluation : extraction des exigences et matching

Hors pytest : ces scripts appellent le vrai LLM (clé dans `.env`), donc le réseau et le quota. Deux évaluations : l'**extraction** des exigences d'une offre (`eval_requirements.py`, ci-dessous) et le **matching** CV / exigences (`eval_matching.py`, [plus bas](#évaluation-du-matching)). Leurs calculs sont testés hors ligne (`tests/test_eval_requirements.py`, `tests/test_eval_matching.py`).

## Évaluation de l'extraction

```bash
cd backend
.venv/bin/python eval/eval_requirements.py                                   # modèle du .env
.venv/bin/python eval/eval_requirements.py --model mistral-large-4-0 --model ministral-14b-2512
.venv/bin/python eval/eval_requirements.py --offer eval/offers/hager.txt --json
.venv/bin/python eval/eval_requirements.py --offer eval/offers/sii.txt --model mistral-large-4-0 --runs 3
```

Offres de référence (publiques, sans donnée personnelle) : `hager.txt` (stage ingénieur informatique & IA, 14 exigences attendues) et `sii.txt` (stage IA applicatif, 13 exigences attendues et 3 acceptables ; « Analyse de données et historique » et « Approche centrée utilisateur » sont reconnues par les mots-clés `match`).

Les modèles, et les exécutions d'un même modèle, sont évalués l'un après l'autre ; tous les appels passent par le même limiteur que l'application (`LLM_RPS`, `LLM_CONCURRENCY` du `.env`, nouvelles tentatives sur 429).

### Limites connues

- **« Compréhension des besoins utilisateurs » (SII)** : manquée 2/3 avec le prompt d'extraction v2 (règle d'importance élargie), contre trouvée 3/3 en v1. Savoir-être souhaité, de poids faible dans le score ; non tranché entre variance du modèle et effet de la nouvelle règle (trois exécutions ne suffisent pas à les distinguer).

### Cas d'évaluation

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

### Méthode de rapprochement

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

### Plusieurs exécutions (`--runs N`)

Le LLM ne répond pas toujours de la même façon, même à température 0 et avec une graine : une seule exécution peut être chanceuse ou malchanceuse. Avec `--runs N`, chaque modèle est exécuté N fois sur la même offre (appels successifs, au rythme de `LLM_RPS`), chaque exécution est comparée au corrigé comme ci-dessus et affichée, puis le rapport les agrège :

- **taux de trouvées** : moyenne et minimum de « trouvées / attendues » sur les exécutions, avec le détail par exécution (ex. `13/13, 12/13, 13/13`) ; le minimum dit ce qu'on obtient dans le moins bon cas ;
- **inventées** et **interdites** : nombre moyen par exécution, puis chaque libellé avec sa fréquence ;
- **manquées au moins une fois** : chaque exigence attendue absente d'au moins une exécution, avec sa fréquence (`Analyse de données de tickets : manquée 1/3`) ; une exigence manquée 3/3 est un vrai trou de l'extraction, une manquée 1/3 relève de la variance ;
- **erreurs d'importance et de catégorie**, **fusions**, **découpages** et **acceptables trouvées** : chacun avec sa fréquence.

Une exécution en erreur (quota, réseau) est signalée et exclue des taux et des fréquences, qui portent sur les exécutions réussies. `--json` donne la synthèse agrégée au format JSON.

## Évaluation du matching

Mesure, exigence par exigence, le niveau attribué par le pipeline complet (recherche, LLM d'évaluation, vérification des citations, seconde chance, plafonds) face à un corrigé établi à la main. Elle sert de **référence** : le prompt de matching n'est pas ajusté à ces cas, et un futur changement (agent, autre modèle) se compare à ces chiffres.

```bash
cd backend
.venv/bin/python eval/eval_matching.py                        # 4 cas, une exécution
.venv/bin/python eval/eval_matching.py --case java_cv__sii    # un cas (répétable)
.venv/bin/python eval/eval_matching.py --runs 3 --json        # agrégation, sortie JSON
```

**Budget** : un appel au LLM par exigence (plus une seconde chance éventuelle), soit environ 30 000 tokens par cas et 130 000 pour les 4 cas, ~65 % du quota journalier gratuit de Groq (200 000 tokens) ; au rythme de `LLM_RPS=0.05`, ~20 minutes. D'où **une exécution par jour** avec l'offre gratuite.

### Données

- **Exigences figées** (`offers/<offre>.requirements.json`) : extraites une fois par `freeze_requirements.py` avec le modèle du `.env`, puis relues et corrigées à la main d'après `offers/<offre>.expected.json` (importances, exigences manquantes, découpées ou en trop). Chaque correction est listée dans le champ `review` du fichier. L'évaluation n'appelle jamais l'extraction (`analyze(requirements=…)`, `requirements_source="provided"`) : ses variations ne se mélangent pas à celles du matching.

  ```bash
  .venv/bin/python eval/freeze_requirements.py eval/offers/sii.txt [--model X] [--force]
  ```

- **CV fictifs** (`cvs/`) : `demo_cv.pdf` (deux colonnes, profil IA, `make_demo_cv.py`) et `java_cv.pdf` (une colonne, développeur backend Java junior sans IA, `make_java_cv.py`). Identités, établissements, entreprises et coordonnées inventés.
- **Séries** : `référence` (les 4 cas du run de référence : `demo_cv` et `java_cv` × `hager` et `sii`, offres publiques) et `difficile` (2 cas conçus par l'auteur du projet : `backend_cv` et `java_cv` × `logistique`, offre **fictive**, champ `"series": "difficile"` du corrigé). Par défaut, `eval_matching.py` n'évalue que la série de référence ; `--series difficile` ou `--series toutes` pour les autres. Les séries ne se mélangent pas : la série difficile vise des faiblesses connues du code (voir les biais ci-dessous), et les 6 cas ensemble (~185 000 tokens) frôleraient le quota journalier gratuit de Groq.
- **Cas difficiles** : l'offre `logistique.txt` exige des durées (2 ans de backend, 3 ans de Python, 1 an de file de messages) et des compétences prouvables sans être nommées ; son corrigé d'extraction (`logistique.expected.json`) est aussi écrit par l'auteur. Le CV `backend_cv.pdf` (`make_backend_cv.py`) disperse ses preuves sur trois expériences : durées à additionner (aucune expérience seule ne suffit), RabbitMQ et Kafka sur deux périodes qui se chevauchent (11 mois une fois fusionnées, 14 en somme naïve, pour 1 an exigé), première expérience découpée en deux chunks dont le second répète l'intitulé mais pas les dates. Garder le titre « EXPÉRIENCE » : chaque échec doit pouvoir être attribué à une seule cause (les titres inhabituels sont une limite documentée à part). Pour les durées, chaque ligne du corrigé donne le calcul en mois (périodes, chevauchements fusionnés, date de référence `EVAL_TODAY`). Biais : offre, CV et corrigés écrits par l'auteur du projet, qui connaît le code ; deux phrases ont dû être reformulées parce qu'elles reprenaient 5 mots consécutifs des prompts (le test des prompts génériques couvre automatiquement les nouveaux fichiers : c'est le cas qu'on ajuste, jamais le prompt).
- **Corrigé des niveaux** (`matching/<cv>__<offre>.expected.json`, 6 cas) :

  ```json
  {"cv": "demo_cv.pdf", "offer": "sii", "requirements": [
    {"id": "R1", "label": "Diplôme d'ingénieur", "levels": ["démontré", "solide"],
     "justification": "« Diplôme d'ingénieure, informatique et IA · École d'ingénieurs… » : diplôme exact"}]}
  ```

  `levels` liste les niveaux acceptés, le premier étant le niveau principal ; deux niveaux voisins seulement pour un cas vraiment ambigu, justifié. Le script refuse un corrigé dont les identifiants ou libellés ne correspondent plus à la liste figée.

### Établissement du corrigé (à l'aveugle)

Le corrigé a été écrit **avant tout run du matching**, en lisant uniquement le texte extrait des CV (tel que le pipeline le découpe) et les définitions des niveaux du prompt, puis validé par le développeur. Les niveaux suivent ce qui est **réellement écrit** dans le texte extrait, pas l'intention de l'auteur des CV fictifs ; quand le texte ne prouvait pas ce qui était voulu, c'est signalé plutôt que corrigé (Docker « (notions) » du CV Java n'est demandé par aucune offre ; « B1 (TOEIC 785) » rend « anglais professionnel » ambigu). Règles appliquées :

- celles du prompt : liste de compétences = « mentionné », usage dans une expérience, un projet ou une formation = « démontré », usage répété ou significatif = « solide » ; diplôme, certification ou score officiel = au moins « démontré » ; implications directes seulement (framework -> langage, SGBD -> SQL) ;
- celles du code : savoir-être plafonné à « démontré » (aucune durée minimale dans ces listes) ;
- **une règle propre au corrigé** : un savoir-être doit être **nommé ou décrit** (« travail en équipe de quatre ») ; il ne se déduit pas d'une réalisation technique. Le prompt ne l'impose pas : l'écart est mesuré à part (voir ci-dessous).

**Corrigé modifié après des runs** (tracé dans le champ `review` du fichier) : le 2026-11-16, dans `java_cv__hager`, R2 « Anglais professionnel » (« Anglais : B1 (TOEIC 785) ») est passé de « démontré / mentionné » à « démontré » seul, pour l'aligner sur une règle ajoutée **après** les runs v1 et v2 : le plancher des certifications de langue, appliqué par le code. Les résultats publiés de v1 et v2 restent tels quels (calculés avec l'ancien corrigé, où « mentionné » était accepté) ; un run ultérieur se compare donc à eux en tenant compte de ce changement.

**Transparence** : pour le cas `demo_cv__sii`, des résultats de l'outil avaient été vus avant l'écriture du corrigé (niveaux d'« École d'ingénieurs » et de Python, scores de 60 à 65) ; ce cas est le moins aveugle des quatre.

### Mesures

Toutes les mesures sont données **séparément pour deux groupes** : exigences **techniques** (compétences, formation, langues) et **savoir-être**, puis au total. La séparation empêche que l'écart attendu sur les savoir-être (règle du corrigé absente du prompt) masque les erreurs techniques.

- **exactitude** : le niveau obtenu fait partie des niveaux acceptés ;
- **exactitude à un niveau près** : au plus un cran d'écart avec le niveau accepté le plus proche ;
- **faux positifs**, la mesure la plus importante (un outil qui affirme une compétence absente est pire qu'un outil prudent) : niveau obtenu « démontré » ou « solide » alors que **tous** les niveaux acceptés sont « non démontré » ou « mentionné » ; chacun est listé avec ses citations ;
- **faux négatifs** : l'inverse ; un corrigé ambigu entre « mentionné » et « démontré » ne produit ni l'un ni l'autre ;
- **matrice de confusion** : niveau principal du corrigé × niveau obtenu (plus une colonne « erreur ») ;
- **écart de score** : score obtenu moins score du corrigé (niveaux principaux, mêmes points et poids que l'application), avec la fourchette du corrigé (niveaux acceptés les plus bas et les plus hauts) ;
- **citations rejetées**, **secondes chances** (et combien ont retrouvé une preuve exacte), **erreurs**, **durée** ;
- **rapports bruts** (`--reports fichier.json`, depuis le 2026-11-19) : le rapport complet de chaque cas (chunks envoyés, preuves, périodes, drapeaux, justifications), pour rejouer hors ligne une nouvelle règle du code sans nouveau run ; les runs v1 à v3 de la série de référence ne l'ont pas gardé ;
- **coût en tokens** (depuis le 2026-11-17) : usage renvoyé par le fournisseur à chaque réponse (entrée, sortie, dont raisonnement), par cas et au total, avec la **moyenne par exigence** ; les runs antérieurs (v1 à v3) ne l'ont pas mesuré. Mesure clé pour comparer une autre version du matching ou un agent, qui ferait plus d'appels ;
- avec `--runs N` : agrégation, et les exigences dont le niveau varie d'une exécution à l'autre.

Le rapport indique la version du prompt d'évaluation (`MATCH_PROMPT_VERSION`) : deux runs ne se comparent qu'en sachant laquelle a servi (v1 : référence ; v2 : règle des savoir-être nommés ou décrits, rappel des scores officiels ; v3 : une affirmation du profil sans expérience qui l'illustre vaut « mentionné » ; v4 : pour « solide », une citation par expérience ou projet distinct). Le plancher des certifications de langue, appliqué par le code depuis la v3, n'a pas de numéro de version propre : il est daté dans `Structure.md`.

Une exigence en erreur n'est comptée ni exacte ni fausse (colonne « erreur »). Un refus durable du fournisseur (quota journalier, modèle refusé) arrête l'évaluation ; un autre échec marque le cas en échec et passe au suivant. La date est fixée (`EVAL_TODAY`) pour que les périodes « en cours » ne changent pas d'un jour à l'autre.
