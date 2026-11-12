# AI Resume Analyzer

> **Note** : `CLAUDE.md` et ce fichier `Structure.md` servent de consignes à l'assistant de code (Claude Code) utilisé pour développer le projet : architecture, conventions, décisions et état d'avancement. Ils documentent aussi le projet en détail ; la présentation générale est dans le `README.md`.

Application web qui analyse un CV (PDF) et évalue sa correspondance avec une offre d'emploi via un pipeline LLM : extraction du texte, embeddings, recherche vectorielle FAISS, évaluation par exigence, restitution en temps réel.

## Stack
- **Backend** : Python 3.11+, FastAPI, LangChain, FAISS (faiss-cpu), PyMuPDF
- **Embeddings** : configurables — local via fastembed (ONNX, sans PyTorch, aucun envoi externe), OpenAI ou Mistral
- **LLM** : Mistral (défaut) ou OpenAI, configurable, température 0, sortie structurée (`with_structured_output`)
- **Frontend** : React + Vite + TypeScript, Tailwind
- **Temps réel** : Server-Sent Events (SSE) du backend vers le frontend
- **Tests** : pytest (backend), Vitest (frontend)

## Structure
```
backend/
  app/
    main.py            # FastAPI, routes ; POST /analyze (SSE), créneaux d'analyse, préparation avant flux
    config.py          # settings (pydantic-settings, lit .env)
    pdf_parser.py      # extraction + nettoyage du texte du CV
    pii.py             # anonymisation (emails, téléphones, URLs)
    chunking.py        # découpage par section puis par item, anonymisé ([NAME] compris)
    providers.py       # MissingAPIKeyError, require_key (communs aux fabriques)
    llm_calls.py       # RateLimiter partagé, AnalysisMonitor (attentes, appels par analyse), call_with_retry (429), invoke_structured
    embeddings.py      # fabrique get_embeddings(settings) : local | openai | mistral
    llm.py             # fabrique get_llm(settings) : mistral | openai
    vectorstore.py     # CVIndex : index FAISS en mémoire, recherche par similarité cosinus
    requirements.py    # extraction des exigences de l'offre (LLM, sortie structurée)
    matcher.py         # matching exigence <-> passages du CV, vérifications, score ; analyze() = générateur d'événements
    schemas.py         # modèles Pydantic (Requirement, JobRequirements, MatchResult, Report)
  tests/
  eval/                # évaluation avec le vrai LLM, hors pytest (voir eval/README.md)
    eval_requirements.py   # extraction vs attendu : trouvées, manquantes, inventées, interdites, importance
    offers/            # cas de référence : <nom>.txt (offre publique) + <nom>.expected.json
frontend/
  src/
    api/               # types.ts (alignés sur schemas.py), sse.ts (parseur SSE), events.ts (toAnalysisEvent),
                       # analyze.ts (POST /analyze + lecture du flux), client.ts (/health, /config)
      __fixtures__/events.sample.json   # contrat : un exemple de chaque événement, généré par le backend
    state/             # analysisReducer.ts : événements -> état de l'écran (fonction pure)
    hooks/             # useAnalysis (reducer + AbortController + chrono), useBackendHealth, usePublicConfig
    components/        # Header, AnalysisForm, ProgressPanel, RequirementList/Card, ResultSummary, ErrorBanner
    test/              # flux SSE simulés (ReadableStream), aucun appel réseau
```

## Pipeline
1. Upload du CV (PDF) + texte de l'offre
2. Extraction du texte du PDF (ordre des colonnes, puces, nettoyage) ; la gouttière entre deux colonnes est détectée (bord droit qui met face à face, à même hauteur, le plus de blocs de part et d'autre), pas supposée au milieu de la page : une colonne latérale étroite est reconnue, et une page à une seule colonne n'est pas découpée
3. Anonymisation du texte : emails, téléphones et URLs -> `[EMAIL]`, `[PHONE]`, `[URL]` ; nom (1re ligne de l'en-tête) -> `[NAME]`. Toujours appliquée, quel que soit le fournisseur
4. Découpage selon la structure du CV : en-tête, sections (titres en majuscules), items (intitulé + date + puces + ligne de technologies ; une ligne de technologies ne se rattache que directement après une puce, donc un intitulé « Poste · Entreprise » qui suit ouvre un nouvel item) ; `RecursiveCharacterTextSplitter` seulement pour un item > `CHUNK_MAX_CHARS` (800), l'intitulé étant répété dans chaque morceau
5. Embeddings des chunks (fournisseur configuré) -> index FAISS en mémoire ; score = similarité cosinus (plus haut = plus proche)
6. Extraction des exigences de l'offre par le LLM, en sortie structurée (`ExtractedRequirements` -> `JobRequirements`) :
   - chaque `Requirement` : `id` (R1…, attribué par le code), `label`, `category` (compétence technique, expérience, formation, langue, savoir-être, autre), `importance` (obligatoire / souhaitée), `min_years` (si l'offre donne une durée), `search_queries` (2 ou 3 reformulations avec termes proches, pour la recherche FAISS)
   - une exigence = un seul critère prouvable par un CV (sauf un niveau de diplôme dans un domaine, qui reste une seule exigence) ; ignorés : présentation de l'entreprise, avantages, processus de recrutement et conditions logistiques (dates, durée, disponibilité, lieu, mobilité, contrat, rémunération, télétravail) ; les compétences supposées par les missions deviennent des exigences
   - importance (prompt v2) : une compétence technique, une formation ou une langue énoncée dans la partie profil / candidat est « obligatoire » par défaut, quelle que soit la formulation (« vous maîtrisez », « à l'aise en », « vous avez une expérience de », « formation Bac+5 », simple mention) ; elle devient « souhaitée » seulement si l'offre la nuance (« un plus », « idéalement », « apprécié », « serait un atout », « curieux de », « intérêt pour »…), la nuance valant pour toute la phrase ou l'énumération. Savoir-être « souhaités » sauf mention explicite ; compétences tirées des missions « souhaitées » sauf si le profil les énonce aussi. Exemples génériques dans le prompt, jamais tirés des offres de référence
   - `EXTRACTION_PROMPT_VERSION` (clé du cache) est à incrémenter à chaque modification du prompt : un test compare l'empreinte sha256 du prompt à celle enregistrée pour la version courante et échoue sinon
   - ordre imposé par le code (ordre du LLM conservé dans chaque groupe) : obligatoires, puis souhaitées techniques, de formation et autres, puis savoir-être souhaités ; au plus `MAX_REQUIREMENTS` (15), les savoir-être souhaités sautant en premier ; offre limitée à `MAX_JOB_OFFER_CHARS` (15 000, sinon 413) ; aucune exigence -> 422
   - cache des exigences (`RequirementsCache`, créé par `create_app`) : même offre -> mêmes exigences, sans appel au LLM. Clé = sha256 de l'offre normalisée (NFC, espaces fusionnés) + fournisseur + modèle d'extraction + `EXTRACTION_PROMPT_VERSION` (constante de `requirements.py`, **à incrémenter à chaque modification du prompt d'extraction ou de `_finalize`**) + `MAX_REQUIREMENTS`. LRU en mémoire, borné par `REQUIREMENTS_CACHE_SIZE`, jamais écrit sur disque, aucune donnée du CV ; les échecs ne sont pas mis en cache ; `requirements_from_cache` dans le rapport et `from_cache` dans l'événement `requirements_extracted`
   - l'offre est une donnée non fiable : isolée entre `<offre>` et `</offre>` (balises du texte neutralisées), et le prompt interdit d'en suivre les instructions
   - erreur du fournisseur (réseau, délai, sortie invalide) : une seule nouvelle tentative ; 429 : voir « Débit LLM » ; puis `LLMProviderError` -> 502
7. Pour chaque exigence (`matcher.py`, modèle `MATCH_LLM_MODEL`) :
   - recherche FAISS avec le label et chaque `search_query`, fusion sans doublon (meilleur score par chunk), puis sélection sensible aux sections (`select_chunks`) car les listes de compétences, très proches des requêtes, occupaient tout le top k : (a) tout chunk d'expérience ou de projet contenant le label comme mot entier (« Git » dans « Git flow », pas dans « GitHub »), quitte à dépasser `MATCH_K` ; (b) les meilleurs autres chunks de ces sections jusqu'à `MATCH_RESERVED_K` (3) places ; (c) les meilleurs chunks restants jusqu'à `MATCH_K` (5) ; sections reconnues par mot-clé du titre (`MATCH_RESERVED_SECTIONS`) ; seuls les chunks anonymisés sont envoyés, isolés dans `<cv>` / `<chunk id section>` (balises du texte neutralisées), le CV étant une donnée non fiable
   - le LLM renvoie `RequirementAssessment` : `level` (non démontré / mentionné / démontré / solide), `evidence` [{chunk_id, quote}], `periods` [{start, end}] si `min_years`, `gaps`, `justification` ; règle du prompt : un diplôme, une certification ou un score officiel (TOEIC, IELTS, certification cloud…) vaut au moins « démontré » ; avec plusieurs preuves, le niveau se fonde sur la plus forte ; une technologie qui implique directement la compétence en est une preuve (framework -> langage, SGBD relationnel -> SQL, CI sur une forge Git -> Git), implication écrite dans la justification, sans aller au-delà (exemples génériques dans le prompt, jamais tirés d'un vrai CV)
   - vérifications par le code : chaque citation doit exister dans le chunk cité parmi ceux fournis (normalisation casse, accents, espaces, typographie ; morceaux séparés par « … » d'au moins 3 mots, dans l'ordre ; un « … » entre deux passages contigus du chunk tient lieu de retour à la ligne : la citation est alors acceptée telle quelle, quelle que soit la longueur des morceaux) ; preuves invalides supprimées, comptées et conservées dans `rejected_evidence_items` ; niveau autre que « non démontré » sans preuve valide -> « non démontré » + `downgraded`, sauf **seconde chance** : si toutes les citations sont rejetées, un seul appel supplémentaire au LLM (par le limiteur) reçoit les citations rejetées et le texte exact des chunks cités, et doit recopier ses preuves mot pour mot ou baisser le niveau ; sa réponse est revérifiée avec les mêmes règles, `second_chance` l'indique et les citations rejetées des deux tentatives sont conservées (sans doublon) ; une période n'est retenue que si ses années de début et de fin (sauf « en cours ») figurent dans un même chunk cité avec une preuve valide ; durées additionnées en fusionnant les chevauchements (mois inclus, « en cours » = mois courant), et « démontré »/« solide » plafonné à « mentionné » (`capped_for_duration`) si la durée est inférieure à `min_years` ; un savoir-être « solide » est plafonné à « démontré » (`capped_soft_skill`) : un CV ne peut pas le prouver solidement
   - une erreur sur une exigence ne bloque pas les autres : résultat `status="error"`, exclu du score et listé dans `errors`
8. Score = moyenne pondérée des points (`MATCH_LEVEL_POINTS` : 0 / 40 / 75 / 100), poids `MATCH_WEIGHTS` (obligatoire 2, souhaitée 1, savoir-être souhaité 0,5), arrondi au demi supérieur ; le rapport liste à part les obligatoires « non démontrées »
9. `analyze()` est un générateur asynchrone : `started` aussitôt, puis `requirements_extracted` et `cv_indexed` (extraction et indexation en parallèle, dans l'ordre où elles terminent), un `requirement_evaluated` par exigence dès qu'elle termine, des `status` quand un 429 fait attendre l'analyse, puis `report`. Une seule `asyncio.Queue` reçoit les fins de tâches, les attentes signalées et l'expiration du délai. Les erreurs fatales sont levées : `LLMProviderError`, `NoRequirementsError`, `EmbeddingsError`, `AnalysisTimeoutError` (au-delà de `ANALYSIS_TIMEOUT_S`). À la fermeture du générateur (fin, erreur, délai, consommateur parti), toutes les tâches non terminées sont annulées : une évaluation encore en attente dans le limiteur n'appelle jamais le LLM ; le nombre d'appels interrompus et évités est journalisé (« Analyse interrompue : … »)
10. `POST /analyze` (multipart `file` + `job_offer`) diffuse ces événements en Server-Sent Events (voir « Flux SSE »)

## Flux SSE (`POST /analyze`)

**Avant d'ouvrir le flux** (réponses HTTP JSON classiques `{"detail": "…"}`, comme les routes `/debug`) :
- créneau d'analyse : au plus `ANALYSIS_CONCURRENCY` (2) analyses simultanées, le quota du fournisseur étant partagé ; au-delà, **429** « Trop d'analyses en cours, réessayez dans quelques instants. » (pas de file d'attente) ;
- préparation, dans un thread (la boucle asynchrone n'est jamais bloquée) : taille de l'offre (**413**), lecture et extraction du PDF (**400** PDF invalide ou sans texte, **413** trop lourd ou trop de pages), construction des embeddings et des deux LLM (**500** si la clé du fournisseur manque), modèle d'embeddings chargé (**503** si son chargement échoue encore) ; champ vide ou absent : **422** (FastAPI) ;
- les embeddings sont construits **une seule fois par application**. Avec `EMBEDDINGS_PROVIDER=local`, le modèle (fastembed, ~2 Go) est chargé **au démarrage** (lifespan), dans un thread en tâche de fond : le serveur répond tout de suite (un premier téléchargement prend plusieurs minutes), le journal indique « Modèle d'embeddings chargé en X s », et une analyse arrivée pendant le chargement l'attend (verrou de `app.embeddings`). Si le chargement échoue, l'application démarre quand même (erreur journalisée) et la première analyse réessaie ; en cas de nouvel échec, 503 avant le flux.

**Pendant le flux** (`text/event-stream`, sse-starlette) : un événement SSE par événement d'analyse, avec `event:` = le type, `id:` = 1, 2, 3… (croissant) et `data:` = le JSON du modèle Pydantic (son champ `type` répète le type). Un commentaire `: ping …` toutes les `SSE_PING_S` (15) s garde la connexion ouverte pendant les longues attentes (à ignorer côté client). Le flux se ferme après `report` (succès) ou `error` (échec).

| `event` | Quand | Champs de `data` (outre `type`) |
|---|---|---|
| `requirements_extracted` | exigences extraites (ou servies par le cache) | `job_title`, `requirements` (liste de `Requirement` : `id`, `label`, `category`, `importance`, `min_years`, `search_queries`), `seconds`, `from_cache` |
| `started` | dès l'ouverture du flux, avant l'indexation et l'extraction | `phases` (`["indexing", "extraction", "evaluation"]` : indexation et extraction en parallèle, puis évaluation), `timeout_s` |
| `cv_indexed` | CV découpé et indexé | `chunk_count`, `seconds` |
| `status` | un 429 fait attendre l'analyse | `reason` (`"rate_limit"`), `wait_s`, `message` (« En attente du quota du fournisseur, reprise dans 60 s ») |
| `requirement_evaluated` | une exigence évaluée (dans l'ordre où elles terminent) | `result` (`RequirementMatch` : `requirement`, `status` `ok`/`error`, `level`, `llm_level`, `evidence` [{`chunk_id`, `quote`}], `rejected_evidence`, `rejected_evidence_items`, `periods`, `rejected_periods`, `total_months`, `downgraded`, `capped_for_duration`, `capped_soft_skill`, `second_chance`, `gaps`, `justification`, `retrieved_chunk_ids`, `points`, `weight`, `error`), `done`, `total` |
| `report` | dernier événement d'un succès | `report` (`MatchReport` : `job_title`, `score`, `requirements` (dans l'ordre des exigences), `missing_mandatory`, `rejected_evidence_count`, `errors`, `extraction_model`, `match_model`, `requirements_from_cache`, `chunk_count`, `timings` {`extraction_s`, `indexing_s`, `retrieval_s`, `evaluation_s`}) |
| `error` | dernier événement d'un échec | `code`, `message` (sans détail interne) |

Codes d'`error` : `llm_provider_error` (fournisseur LLM en échec après les nouvelles tentatives), `no_requirements` (aucune exigence dans l'offre), `embeddings_error` (calcul des embeddings), `timeout` (au-delà de `ANALYSIS_TIMEOUT_S`, 600 s), `internal_error`.

**Status pour toutes les analyses** : chaque analyse place un `AnalysisMonitor` dans le contexte (`ContextVar`) de ses seules tâches ; le limiteur partagé y compte ses appels en attente et en cours. Sur 429 : l'analyse dont l'appel attend reçoit un `status` ; si le limiteur se met en pause (quota de tokens épuisé, débit par minute), **chaque analyse qui a des appels en attente** en reçoit un aussi. `llm_calls` ne dépend pas de l'API web.

**Déconnexion du client** : sse-starlette ferme le générateur, qui ferme `analyze()` (`aclosing`) : tâches annulées, aucun nouvel appel au LLM, journal « Analyse interrompue : N évaluation(s) annulée(s) ; appels LLM : X en cours interrompu(s), Y jamais lancé(s) (évités) ». Le créneau est libéré dans tous les cas (fin, erreur, délai, déconnexion) : `release()` idempotent, appelé à la fin du générateur et en tâche de fond de la réponse.

**Autres routes publiques** : `GET /health` -> `{"status": "ok", "embeddings_ready": bool}` (`false` pendant le chargement du modèle local) ; `GET /config` -> limites publiques `{"max_pdf_size_bytes", "max_pdf_pages", "max_job_offer_chars"}`, lues par le frontend au chargement.

**Contrat backend / frontend** : `backend/tests/test_contract.py` produit, à partir des modèles Pydantic, un exemple valide de chaque type d'événement dans `frontend/src/api/__fixtures__/events.sample.json`, et échoue si le fichier n'est pas à jour ; `frontend/src/api/events.test.ts` passe chaque exemple dans `toAnalysisEvent` et vérifie que tous les types sont reconnus avec leurs champs. Après une modification des schémas : `cd backend && UPDATE_CONTRACT_FIXTURES=1 .venv/bin/pytest tests/test_contract.py`, puis adapter `frontend/src/api/types.ts` et `events.ts` si besoin.

**Lecture côté client** : `EventSource` ne permet pas de POST (ni d'envoyer un fichier). Le frontend lira le flux avec `fetch` (corps `FormData`) puis `response.body` (`ReadableStream`) décodé par `TextDecoderStream`, en découpant les événements sur les lignes vides et en lisant les lignes `event:`, `id:` et `data:` (ignorer celles qui commencent par `:`), ou avec une petite librairie dédiée comme `@microsoft/fetch-event-source`. Avant de lire le flux, vérifier `response.ok` : en cas d'erreur de validation, la réponse est du JSON `{"detail"}` avec le code HTTP. Exemple brut :
```
event: requirement_evaluated
id: 5
data: {"type":"requirement_evaluated","result":{…},"done":3,"total":14}

: ping - 2026-11-09 10:12:45.123+00:00

```

## Configuration des fournisseurs
Variables de `.env` (modèle complet dans `.env.example`) :

| Variable | Valeurs | Défaut |
|---|---|---|
| `EMBEDDINGS_PROVIDER` | `local` \| `openai` \| `mistral` | `local` |
| `EMBEDDINGS_MODEL` | nom du modèle | `intfloat/multilingual-e5-large` (local), `text-embedding-3-small` (openai), `mistral-embed` (mistral) |
| `EMBEDDINGS_CACHE_DIR` | dossier des modèles locaux | `~/.cache/ai-resume-analyzer/fastembed` |
| `LLM_PROVIDER` | `mistral` \| `openai` | `mistral` |
| `LLM_MODEL` | nom du modèle | `mistral-small-latest` (mistral), `gpt-4o-mini` (openai) |
| `MATCH_LLM_MODEL` | modèle d'évaluation des exigences | `ministral-14b-2512` (mistral), `gpt-4o-mini` (openai) |
| `LLM_TIMEOUT_S` | secondes | `60` |
| `LLM_SEED` | graine (`random_seed` Mistral, `seed` OpenAI) ; vide : aucune | `42` |
| `LLM_RPS`, `LLM_CONCURRENCY` | limiteur partagé par tous les appels LLM | `0.5`, `4` |
| `LLM_RATE_LIMIT_RETRIES`, `LLM_RATE_LIMIT_BACKOFF_S` | nouvelles tentatives sur 429 | `3`, `2.0` |
| `MAX_JOB_OFFER_CHARS`, `MAX_REQUIREMENTS` | limites de l'offre | `15000`, `15` |
| `REQUIREMENTS_CACHE_SIZE` | offres gardées dans le cache des exigences (0 : désactivé) | `64` |
| `ANALYSIS_CONCURRENCY` | analyses simultanées sur `/analyze` (429 au-delà) | `2` |
| `ANALYSIS_TIMEOUT_S` | durée maximale d'une analyse (événement `error` `timeout` au-delà) | `600` |
| `SSE_PING_S` | intervalle des pings du flux SSE (0 : aucun) | `15` |
| `MATCH_K` | chunks par exigence | `5` |
| `MATCH_RESERVED_K`, `MATCH_RESERVED_SECTIONS` | places réservées aux expériences et projets ; mots-clés de leurs titres (JSON) | `3`, `["EXPERIENCE", "PROJET", "STAGE"]` |
| `MATCH_LEVEL_POINTS`, `MATCH_WEIGHTS` | JSON | voir pipeline, étape 8 |
| `OPENAI_API_KEY`, `MISTRAL_API_KEY` | clés | aucune |

- Garde-fou du `.env` (`Settings` ignore les variables inconnues, `extra="ignore"`) : au démarrage (`create_app()` sans settings explicites), un warning pour chaque variable dont le premier mot est celui d'un paramètre (`LLM_`, `MATCH_`, `EMBEDDINGS_`, `MAX_`, `REQUIREMENTS_`…, singulier compris) mais qu'aucun paramètre ne connaît, avec le nom le plus proche (ressemblance des caractères + même dernier mot / même premier mot ; aucune suggestion sous le seuil), et pour chaque variable définie plusieurs fois (seule la dernière compte). Seuls les noms sont lus, jamais les valeurs.
- Seule la clé du fournisseur choisi est exigée, au moment de construire les embeddings ou le LLM (pas au démarrage) : `MissingAPIKeyError` avec un message explicite, renvoyée en 500 par l'API.
- Mistral, offre gratuite : les modèles autorisés dépendent de l'abonnement, et les alias (`*-latest`) peuvent être refusés (429 avec `x-ratelimit-limit-req-minute: 0`, ou 403 `tier_not_allowed`). Vérifié le 2026-11-03 sur le compte du projet : `mistral-large-4-0` répond (50 req/min, 20 000 tokens/min), `mistral-small-*` et `mistral-large-2512` sont refusés ; d'où `LLM_MODEL` à fixer explicitement dans `.env`. Temps mesurés sur l'offre Hager (4 000 caractères) : 16 à 30 s avec `mistral-large-4-0`, 6 à 22 s avec `ministral-14b-2512` (forte variance d'un appel à l'autre).
- **Débit LLM** : l'offre gratuite est limitée à **1 requête par seconde** pour le compte, plus des limites par modèle. Tous les appels au LLM du processus (extraction et évaluation) passent par un seul `RateLimiter` (créé par `create_app`) : départs espacés d'au moins 1 / `LLM_RPS` s, au plus `LLM_CONCURRENCY` appels en cours ; chaque nouvelle tentative repasse par le limiteur. Sur 429 (au plus `LLM_RATE_LIMIT_RETRIES` nouvelles tentatives, attente plafonnée à 60 s) : `Retry-After` s'il existe ; sinon, si l'en-tête `x-ratelimit-limit-req-minute` est présent, attente de 60 / limite s et limiteur ralenti à cet intervalle jusqu'à la fin de l'analyse (`slow_down`, puis `le débit normal n'est rétabli que lorsque plus aucune analyse n'est active : compteur `begin_analysis()` / `end_analysis()` du limiteur, car le ralentissement vaut pour le compte entier) ; si `x-ratelimit-remaining-tokens-minute` vaut 0 (quota de tokens épuisé, cas des 429 sans en-tête de requêtes), attente de 60 s, fenêtre entière car le fournisseur n'indique pas sa fin, pendant laquelle le limiteur ne lance aucun appel (`hold`) ; sinon attente croissante (2, 4, 8 s) ; puis abandon ; sur une autre erreur, une seule nouvelle tentative. Chaque 429 est journalisé avec les en-têtes `x-ratelimit-*`.
- Limites constatées le 2026-11-08 (en-têtes `x-ratelimit-*`) : `mistral-large-4-0` 50 req/min et 20 000 tokens/min ; `ministral-14b-2512` et `ministral-8b-2512` **2 req/min** (30 req/min le 2026-11-03) et 40 000 tokens/min. Avec 2 req/min, ministral termine grâce à l'attente de 60 / limite, mais en ~450 s pour 15 exigences. Choix retenu (2026-11-09) dans `.env` : `MATCH_LLM_MODEL=mistral-large-4-0` et `LLM_RPS=0.15`.
- Vérification réelle du 2026-11-09 (CV réel, offre Hager, config lue depuis `.env`) : 15 exigences, 0 erreur, 0 preuve rejetée, aucun 429, 148 s au total (extraction 31 s, indexation 7 s en parallèle, recherche 2 s, évaluation 115 s). Run précédent : deux 429 sur la limite de tokens (20 000/min) rattrapés par l'attente de 2 s. Variance d'un run à l'autre : l'extraction donne 14 ou 15 exigences, et un même passage peut être jugé « démontré » ou « mentionné ».
- Après la sélection sensible aux sections et les règles « preuve la plus forte » / « implication directe » (2026-11-09, `LLM_SEED=42`, 3 analyses) : scores 45, 47, 50 (contre 30 et 37 avant). La graine ne rend **pas** les résultats reproductibles chez Mistral : l'extraction varie encore (formulation et découpage des exigences, donc requêtes et chunks envoyés), et l'évaluation aussi (Git « démontré »/« solide », IA générative « démontré »/« mentionné »). Chaque analyse a subi 2 ou 3 attentes de 60 s sur la limite de tokens (20 000/min), d'où ~185 s au lieu de ~150 s.
- Avec le cache des exigences, `LLM_RPS=0.1` et `LLM_RATE_LIMIT_RETRIES=5` (2026-11-09, 3 analyses dans un même processus) : scores 48, 48, 51 ; 266 s (extraction 21 s, un 429 rattrapé), puis 175 s et 150 s (exigences du cache) ; 11 exigences sur 14 au même niveau dans les 3 runs. Détail dans le README.
- LLM : température 0 et `max_retries=0` sur les clients LangChain (par défaut Mistral fait jusqu'à 5 tentatives et OpenAI 3) ; les nouvelles tentatives sont gérées par `llm_calls.call_with_retry`. Les deux fournisseurs utilisent `method="function_calling"` (le mode `json_schema` strict d'OpenAI gère mal les champs optionnels).
- Local : fastembed (ONNX), car PyTorch — donc sentence-transformers — n'a pas de wheel macOS x86_64 pour Python 3.13. Le modèle E5 exige les préfixes `query: ` / `passage: `, ajoutés par `LocalEmbeddings` (fastembed ne le fait pas). Premier lancement : téléchargement d'environ 2,2 Go ; ensuite chargé une fois par processus (~2,5 s).
- Mistral : un tokenizer local est fourni à `MistralAIEmbeddings`, sinon il télécharge celui de Mixtral depuis Hugging Face à la construction.
- FAISS n'existe que dans `langchain-community`, en fin de vie : à remplacer si un paquet officiel autonome apparaît (le paquet `langchain-faiss` sur PyPI n'est pas publié par LangChain).

## Conventions
- Typage partout (type hints Python, TypeScript strict)
- Toute sortie LLM passe par un modèle Pydantic (pas de parsing de texte libre)
- Les clés (`OPENAI_API_KEY`, `MISTRAL_API_KEY`) viennent de `.env`, jamais en dur ; fournir `.env.example`
- Aucun appel réseau ni téléchargement de modèle dans les tests (réseau bloqué par une fixture) : embeddings factices et déterministes, appels LLM mockés
- Aucune donnée personnelle (email, téléphone, URL, nom) n'est envoyée à un fournisseur : l'anonymisation est faite dans `chunk_cv`
- Petits commits, un par étape fonctionnelle
- Pas de stockage persistant des CV (données personnelles) : tout reste en mémoire le temps de l'analyse

## Commandes
- Backend : `cd backend && uvicorn app.main:app --reload`
- Tests backend : `cd backend && pytest` (sans réseau) ; `pytest -m integration` appelle les vrais fournisseurs (clé requise)
- Évaluation de l'extraction : `cd backend && .venv/bin/python eval/eval_requirements.py [--model X --model Y]` (vrai LLM, appels sérialisés à 1 requête/s)
- Frontend : `cd frontend && npm run dev` (http://localhost:5173 ; backend : `VITE_API_URL`, défaut http://localhost:8000) ; `npm test`, `npm run build` (TypeScript strict), `npm run lint` (oxlint)
- Qualité backend : `cd backend && .venv/bin/ruff check . && .venv/bin/ruff format --check .` (dépendances de développement : `requirements-dev.txt`)
- Docker : `docker compose up --build` -> http://localhost (Nginx relaie `/api` au backend ; modèle d'embeddings dans le volume `models`)
- CV de démonstration : `cd backend && .venv/bin/python eval/cvs/make_demo_cv.py` (régénère `eval/cvs/demo_cv.pdf`, entièrement fictif)
- CI : `.github/workflows/ci.yml` (ruff, pytest hors ligne, oxlint, Vitest, build), sans secret
- Régénérer le contrat des événements : `cd backend && UPDATE_CONTRACT_FIXTURES=1 .venv/bin/pytest tests/test_contract.py`

## Frontend (`frontend/`)
- **Lecture du flux** (`api/analyze.ts`) : `analyze(file, jobOffer, signal)` est un générateur asynchrone : POST multipart avec `fetch`, puis lecture de `response.body` (`ReadableStream`, `TextDecoder` en mode flux) par `SSEParser` (`api/sse.ts`, écrit à la main : événement coupé entre deux morceaux, `\r\n` coupé, plusieurs lignes `data:`, commentaires et pings, `\n` / `\r\n` / `\r`, BOM ; un événement sans ligne vide finale n'est pas émis), puis `toAnalysisEvent` (`api/events.ts`) : type inconnu ignoré, événement connu mal formé -> `ProtocolError`. Le flux se termine après `report` ou `error`.
- **Erreurs** : `HttpError(status, message)` pour une réponse en erreur avant le flux (message `detail` du backend ; 422 de FastAPI -> message générique en français) ; `ConnectionLostError` si le serveur est injoignable ou si le flux se coupe avant `report`/`error` ; **aucune reconnexion automatique** (jamais de nouvelle analyse sans action de l'utilisateur). L'`AbortController` du bouton « Annuler » interrompt `fetch` : la connexion est fermée et le backend annule l'analyse.
- **État** (`state/analysisReducer.ts`, fonction pure) : statut (`idle`, `running`, `done`, `error`, `cancelled`), phases (`indexing` et `extraction` actives dès `started`, `evaluation` active quand les deux sont finies), exigences (en attente puis évaluées, obligatoires d'abord), `evaluated`/`total`, dernier message d'attente (effacé au prochain événement), rapport, erreur. Après une annulation ou une erreur, les événements suivants sont ignorés et les exigences déjà évaluées restent affichées.
- **Écran** : en-tête avec indicateur du serveur (prêt, « Modèle en cours de chargement » avec nouvelle vérification toutes les 2 s, injoignable avec « Réessayer ») ; formulaire (PDF par glisser-déposer ou sélection, type et taille vérifiés côté client ; offre avec compteur de caractères ; limites lues sur `GET /config`, valeurs de repli `FALLBACK_CONFIG` si la route ne répond pas ; utilisable pendant le chargement du modèle) ; progression (phases, `x/N`, temps écoulé, messages d'attente, « Annuler », zone `aria-live`) ; une carte par exigence dès `requirements_extracted` (libellé, importance, badge de niveau, preuves citées, manques, drapeaux expliqués) ; résultat (score, encadré des obligatoires non démontrées, variation d'environ ±5 points) ; mention de confidentialité.
- **Tests** (Vitest, jsdom) : `fetch` simulé avec un `ReadableStream` (`src/test/stream.ts`), aucun appel réseau ; les flux de test sont construits à partir des exemples du contrat.
