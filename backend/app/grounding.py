"""Contrôle d'ancrage des reformulations de l'agent de suggestions (vérifié par le code).

L'agent ne peut réécrire qu'UNE puce existante d'une expérience ou d'un projet. Chaque fait de
la nouvelle puce porte sa citation source, prise dans le même item (même expérience ou même
projet) ; aucun élément nouveau (nombre, date, nom propre, technologie) n'y apparaît, sauf une
compétence qui découle directement d'un fait du même item selon la table des implications
(app.implications), ou un nom plus large à sens unique (app.terms.ONE_WAY_SYNONYMS : « LLM » ->
« IA générative »). Le premier mot significatif de la puce d'origine (ce qui a été fait) doit
rester. Le contrôle rejette avec des motifs précis, réinjectés dans la proposition
suivante ; il ne juge pas le sens (verbe gonflé, portée ajoutée) : c'est le rôle du juge LLM,
qui ne peut que rejeter.

`apply_rewrite` applique une puce validée sur des chunks en mémoire (copie ; rien n'est écrit,
le CV n'est jamais conservé), pour la réévaluation par le vrai matcher."""

import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from app.implications import IMPLICATIONS, contains_term, implication_for, normalize
from app.matcher import experience_items, quote_in_chunk
from app.schemas import Chunk, SearchResult
from app.terms import ONE_WAY_SYNONYMS, mentions_term

PLACEHOLDERS = ("[NAME]", "[EMAIL]", "[PHONE]", "[URL]")
_PLACEHOLDER = re.compile(r"\[(?:NAME|EMAIL|PHONE|URL)\]")
_BULLET_LINE = re.compile(r"^\s*[-•*▪◦]\s+")

# Technologies et notions techniques reconnues même en minuscules : un terme de cette liste
# absent de l'item est un élément nouveau (« déploiement cloud » tiré de « Docker (notions) »).
# Pas de mot qui soit aussi un mot français courant (« vue », « ci », « go ») : écrits avec une
# majuscule en milieu de phrase (« Go », « Vue »), ils restent détectés comme noms propres.
TECH_TERMS: frozenset[str] = frozenset(
    normalize(term)
    for term in (
        *(i.source for i in IMPLICATIONS),
        *(i.implies for i in IMPLICATIONS),
        "python", "java", "sql", "php", "ruby", "rust", "scala", "kotlin", "javascript", "typescript",
        "c++", "c#", "node.js", "react", "angular", "docker", "kubernetes", "helm", "terraform",
        "ansible", "cloud", "aws", "azure", "gcp", "kafka", "rabbitmq", "redis", "mongodb", "elasticsearch",
        "spark", "hadoop", "airflow", "git", "gitlab", "github", "jenkins", "linux", "grafana", "prometheus",
        "jsp", "xml", "json", "rest", "graphql", "soap", "ci/cd", "devops", "mlops", "llm", "rag",
        "embeddings", "faiss", "pytorch", "tensorflow", "scikit-learn", "pandas", "itsm", "servicenow", "jira",
        "windchill", "plm", "sap", "openapi", "junit", "pytest", "maven", "jpa", "hibernate", "thymeleaf",
        "file de messages", "message queue", "observabilite", "supervision", "monitoring",
        "integration continue", "deploiement continu", "ia generative",
        # Traitement des données personnelles : « anonymisation » ne s'écrit que si l'item le dit
        # déjà (une pseudonymisation n'est pas une anonymisation), sans exception.
        "anonymisation", "anonymisations", "anonymiser", "anonymise", "anonymisee", "anonymises", "anonymisees",
    )
)  # fmt: skip

# Termes génériques qui NOMMENT une pratique : acceptés, même absents de l'item, seulement si la
# puce d'origine décrit déjà cette pratique (« Pipeline de tests et de déploiement lancé à
# chaque fusion » -> « CI/CD »). Aucune exception pour « cloud », « REST », « DevOps » ou
# « file de messages » : décrire une pratique ne prouve ni un hébergement, ni un style d'API,
# ni une technologie.
_TRIGGER = r"(?:a chaque (?:fusion|commit|push|merge)|automatique)"
_CI = re.compile(rf"\b(?:tests?|build|verifications?)\b.*{_TRIGGER}|{_TRIGGER}.*\b(?:tests?|build|verifications?)\b")
_DEPLOY = r"\b(?:deploiements?|mise en production|livraison)\b"
_CD = re.compile(rf"{_DEPLOY}.*{_TRIGGER}|{_TRIGGER}.*{_DEPLOY}")
_OBSERVABILITY = re.compile(
    r"\balertes?\b|\b(?:tableaux? de (?:bord|suivi)|metriques)\b.*\b(?:latence|erreurs?|disponibilite|performances?)\b"
)
PRACTICE_TERMS: dict[str, tuple[str, Callable[[str], bool]]] = {
    "ci/cd": ("tests et déploiement déclenchés à chaque fusion", lambda b: bool(_CI.search(b) and _CD.search(b))),
    "integration continue": ("tests déclenchés à chaque fusion", lambda b: bool(_CI.search(b))),
    "deploiement continu": ("déploiement déclenché à chaque fusion", lambda b: bool(_CD.search(b))),
    **{
        term: ("alertes ou tableaux de suivi de la latence, des erreurs…", lambda b: bool(_OBSERVABILITY.search(b)))
        for term in ("observabilite", "supervision", "monitoring")
    },
}
_WORD = re.compile(r"[\wÀ-ÿ][\wÀ-ÿ+#./-]*")
_LETTERS = re.compile(r"[A-Za-zÀ-ÿ]+")
_FILLERS = frozenset({"les", "des", "une", "aux", "pour", "avec", "sur", "dans", "par", "the"})


_ROOT_LENGTH = 6
# Motif du rejet par la règle du premier mot, repéré par l'évaluation (rejets comptés à part).
ACTION_LOST = "(ce que décrit la puce d'origine) doit rester dans la reformulation"


def lead_word(text: str) -> str | None:
    """Premier mot significatif d'une puce, qui nomme ce qui a été fait (« Classement de 5 000
    tickets » -> « Classement ») : sa racine doit rester dans la reformulation. Protège contre le
    changement d'action et le gonflement du verbe (« Participation » -> « Pilotage »)."""
    return next((w for w in _LETTERS.findall(text) if len(w) >= 3 and w.lower() not in _FILLERS), None)


def root(word: str) -> str:
    """Racine simple : les 6 premières lettres normalisées (sans casse ni accents), le mot entier
    s'il est plus court, sans « s » ou « x » final. « Développement » et « Développé » -> « develo »,
    « Classement » et « classé » -> « classe »."""
    plain = normalize(word)
    if len(plain) > 3 and plain[-1] in "sx":
        plain = plain[:-1]
    return plain[:_ROOT_LENGTH]


def keeps_root(text: str, word: str) -> bool:
    """Un mot du texte commence par la racine de `word`."""
    stem = root(word)
    return any(normalize(w).startswith(stem) for w in _LETTERS.findall(text))


_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


class Fact(BaseModel):
    """Une affirmation de la puce réécrite et sa citation exacte dans le CV."""

    claim: str = Field(min_length=1)
    chunk_id: int
    quote: str = Field(min_length=1)


class ImpliedSkill(BaseModel):
    """Une compétence ajoutée, déduite d'un fait du même item par la table des implications."""

    skill: str = Field(min_length=1)
    chunk_id: int
    quote: str = Field(min_length=1)


class BulletRewrite(BaseModel):
    """Proposition de l'agent : remplacer la puce `bullet_index` (0 = première puce) du chunk
    `chunk_id` par `new_text`, une seule ligne."""

    chunk_id: int
    bullet_index: int = Field(ge=0)
    new_text: str = Field(min_length=1)
    facts: list[Fact] = Field(default_factory=list)
    implied_skills: list[ImpliedSkill] = Field(default_factory=list)


@dataclass
class GroundingIssue:
    code: str
    message: str


@dataclass
class GroundingResult:
    issues: list[GroundingIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def codes(self) -> list[str]:
        return [issue.code for issue in self.issues]


@dataclass(frozen=True)
class Bullet:
    index: int
    text: str  # sans la puce, lignes repliées jointes par une espace
    first_line: int
    last_line: int  # incluse


def bullets(chunk_text: str) -> list[Bullet]:
    """Puces d'un chunk : une ligne qui commence par « - » (ou « • »…), suivie de ses lignes
    repliées (ni puce, ni ligne de technologies « A · B · C »)."""
    lines = chunk_text.splitlines()
    found: list[Bullet] = []
    i = 0
    while i < len(lines):
        if not _BULLET_LINE.match(lines[i]):
            i += 1
            continue
        start, parts = i, [_BULLET_LINE.sub("", lines[i]).strip()]
        i += 1
        while i < len(lines) and lines[i].strip() and not _BULLET_LINE.match(lines[i]) and " · " not in lines[i]:
            parts.append(lines[i].strip())
            i += 1
        found.append(Bullet(len(found), " ".join(parts), start, i - 1))
    return found


def _items(chunks: Iterable[Chunk], keywords: Iterable[str]) -> dict[int, str | None]:
    return experience_items([SearchResult(chunk=c, score=0.0) for c in chunks], keywords)


def _entities(text: str) -> list[str]:
    """Éléments factuels d'un texte : nombres, termes techniques connus, mots à majuscule hors
    début de phrase, sigles et mots qui portent des chiffres ou des symboles (« C++ »)."""
    found: list[str] = [m.group(0) for m in _NUMBER.finditer(text)]
    plain = normalize(text)
    found += [term for term in TECH_TERMS if contains_term(plain, term)]
    for match in _WORD.finditer(text):
        word = match.group(0).rstrip(".,;:")
        before = text[: match.start()].rstrip()
        sentence_start = not before or before[-1] in ".!?:;-•"
        if any(ch.isdigit() for ch in word) or any(ch in "+#" for ch in word):
            found.append(word)
        elif len(word) > 1 and word.isupper():
            found.append(word)  # sigle, même en début de phrase
        elif word[:1].isupper() and not sentence_start:
            found.append(word)
    return found


def check_grounding(
    rewrite: BulletRewrite,
    chunks: Iterable[Chunk],
    sections: Iterable[str],
    max_length_ratio: float = 1.6,
) -> GroundingResult:
    """Contrôle d'ancrage d'une reformulation (voir le module). `sections` : mots-clés des
    sections d'expériences (MATCH_RESERVED_SECTIONS) ; les sections reconnues par leur contenu
    le sont aussi."""
    by_id = {c.index: c for c in chunks}
    items = _items(by_id.values(), sections)
    result = GroundingResult()

    def issue(code: str, message: str) -> None:
        result.issues.append(GroundingIssue(code, message))

    target = by_id.get(rewrite.chunk_id)
    if target is None:
        issue("chunk_inconnu", f"Le chunk {rewrite.chunk_id} n'existe pas.")
        return result
    item = items.get(target.index)
    if item is None:
        issue("hors_experience", "Seule une puce d'expérience ou de projet peut être reformulée.")
        return result
    target_bullets = bullets(target.text)
    if rewrite.bullet_index >= len(target_bullets):
        issue("puce_inconnue", f"Le chunk {target.index} n'a pas de puce {rewrite.bullet_index}.")
        return result
    original = target_bullets[rewrite.bullet_index].text
    new_text = rewrite.new_text.strip()
    if "\n" in new_text or _BULLET_LINE.match(new_text):
        issue("plusieurs_puces", "Une seule puce, sur une seule ligne : aucune puce ajoutée.")

    # Texte de l'item entier (tous les morceaux d'une expérience découpée), seule source admise.
    item_text = "\n".join(c.text for c in by_id.values() if items.get(c.index) == item)

    for fact in rewrite.facts:
        source = by_id.get(fact.chunk_id)
        if source is None or items.get(source.index) is None:
            issue("fait_hors_experience", f"« {fact.claim} » : la source n'est pas une expérience ou un projet.")
        elif items.get(source.index) != item:
            issue("fait_autre_item", f"« {fact.claim} » vient d'une autre expérience ou d'un autre projet.")
        elif not quote_in_chunk(fact.quote, source.text):
            issue("citation_introuvable", f"« {fact.quote} » ne figure pas mot pour mot dans le chunk {source.index}.")
    if not rewrite.facts:
        issue("aucun_fait", "Chaque affirmation de la puce doit citer sa source dans le CV.")

    allowed_skills: set[str] = set()
    for implied in rewrite.implied_skills:
        source = by_id.get(implied.chunk_id)
        if source is None or items.get(source.index) != item or not quote_in_chunk(implied.quote, source.text):
            issue("implication_sans_source", f"« {implied.skill} » : citation source absente de cet item.")
        elif implication_for(implied.skill, implied.quote) is None:
            issue(
                "implication_non_autorisee", f"« {implied.skill} » ne découle pas directement de « {implied.quote} »."
            )
        else:
            allowed_skills.add(normalize(implied.skill))

    # Noms plus larges autorisés à sens unique (« LLM » cité dans l'item -> « IA générative ») :
    # les éléments factuels du nom large sont alors admis.
    for narrow, broad in ONE_WAY_SYNONYMS:
        if contains_term(item_text, narrow) and mentions_term(new_text, broad):
            allowed_skills |= {normalize(e) for e in _entities(broad)}

    lead = lead_word(original)
    if lead is not None and not keeps_root(new_text, lead):
        issue("action_perdue", f"« {lead} » {ACTION_LOST} (même racine acceptée).")

    described = normalize(original)
    entities = {normalize(e): e for e in reversed(_entities(new_text))}  # un seul signalement par terme
    for key, entity in entities.items():
        if key in allowed_skills or contains_term(item_text, entity):
            continue
        practice = PRACTICE_TERMS.get(key)
        if practice is not None and practice[1](described):
            continue  # la puce d'origine décrit exactement la pratique que le terme nomme
        issue("element_nouveau", f"« {entity} » n'apparaît pas dans cette expérience.")

    if set(_PLACEHOLDER.findall(original)) != set(_PLACEHOLDER.findall(new_text)):
        issue("placeholder", "Les éléments anonymisés ([NAME], [EMAIL]…) doivent rester intacts.")

    if len(new_text) > max_length_ratio * max(len(original), 40):
        issue("trop_long", f"Puce trop longue ({len(new_text)} caractères pour {len(original)}).")
    repeated = [
        term for term, count in Counter(t for t in _tech_occurrences(new_text)).items()
        if count > max(1, _tech_occurrences(original).count(term))
    ]  # fmt: skip
    if repeated:
        issue("bourrage", f"Termes répétés : {', '.join(sorted(repeated))}.")
    return result


def _tech_occurrences(text: str) -> list[str]:
    plain = normalize(text)
    found: list[str] = []
    for term in TECH_TERMS:
        found += [term] * len(re.findall(rf"(?<![\w+#]){re.escape(term)}(?![\w+#])", plain))
    return found


def apply_rewrite(chunks: Iterable[Chunk], rewrite: BulletRewrite) -> list[Chunk]:
    """Copie des chunks où la puce visée est remplacée par `new_text` (une ligne) ; les autres
    chunks sont inchangés. À n'appeler qu'après un contrôle d'ancrage réussi."""
    result: list[Chunk] = []
    for chunk in chunks:
        if chunk.index != rewrite.chunk_id:
            result.append(chunk.model_copy())
            continue
        lines = chunk.text.splitlines()
        target = bullets(chunk.text)[rewrite.bullet_index]
        lines[target.first_line : target.last_line + 1] = [f"- {rewrite.new_text.strip()}"]
        result.append(chunk.model_copy(update={"text": "\n".join(lines)}))
    return result
