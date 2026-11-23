"""Termes de l'offre et preuves implicites (agent de suggestions, v2).

Une exigence est prouvée **implicitement** quand le matcher la juge « démontré » ou « solide »
mais qu'aucune preuve citée ne contient le terme de l'offre ni un équivalent direct : la preuve
vient d'une technologie qui l'implique (table des implications), de la ligne de technologies
d'une expérience ou d'une pratique décrite sans son nom. Un lecteur humain ou un tri par
mots-clés peut alors la manquer ; l'agent propose de nommer le terme dans la puce qui porte la
preuve. Détection par le code, jamais par le LLM.

Termes d'une exigence : son libellé sans parenthèses, ses parties (séparées par « et », « ou »,
virgules), les éléments entre parenthèses, chacun aussi sans formule d'introduction générique
(« Outils d'IA générative » -> « IA générative »), et les équivalents directs de la table
`DIRECT_SYNONYMS` (petite, générique : deux noms d'une même chose, jamais une implication)."""

import re

from app.implications import normalize
from app.schemas import Requirement, RequirementMatch

# Équivalents directs : deux façons de nommer la même chose. Une implication (une technologie
# qui prouve une compétence) n'en est pas une : elle relève de app/implications.py.
DIRECT_SYNONYMS: tuple[tuple[str, ...], ...] = (
    ("CI/CD", "intégration continue", "déploiement continu"),
    ("observabilité", "supervision", "monitoring"),
    ("API REST", "REST", "RESTful"),
    ("message queue", "file de messages", "message broker"),
    ("tests automatisés", "tests automatiques"),
)

_INTRO = re.compile(
    r"^(?:outils?|ma[iî]trise|connaissances?|exp[ée]riences?|pratique)\s+(?:d['’]\s*|de\s+|des\s+|du\s+|en\s+)",
    re.IGNORECASE,
)
_SEPARATORS = re.compile(r"\s*(?:,|;|\bet\b|\bou\b)\s*")
_PARENS = re.compile(r"\(([^)]*)\)")


def _singular(text: str) -> str:
    """Pluriels simples ignorés (« bases relationnelles » = « base relationnelle »)."""
    return re.sub(r"\b(\w{3,}?)[sx]\b", r"\1", text)


def _key(term: str) -> str:
    return _singular(normalize(term))


def mentions_term(text: str, term: str) -> bool:
    """Le terme figure comme mot (ou suite de mots) entier dans le texte, après normalisation
    (casse, accents, espaces, pluriels simples)."""
    return re.search(rf"(?<![\w+#]){re.escape(_key(term))}(?![\w+#])", _key(text)) is not None


def offer_terms(requirement: Requirement) -> list[str]:
    """Termes de l'offre pour cette exigence (voir le module), sans doublon, dans l'ordre."""
    label = requirement.label
    head = _PARENS.sub("", label).strip()
    raw = [head, *_SEPARATORS.split(head)]
    for inside in _PARENS.findall(label):
        raw += _SEPARATORS.split(inside)
    terms: list[str] = []
    for term in raw:
        for variant in (term.strip(), _INTRO.sub("", term.strip())):
            if variant:
                terms.append(variant)
    keys = {_key(t) for t in terms}
    for group in DIRECT_SYNONYMS:
        if keys & {_key(t) for t in group}:
            terms += group
    unique: dict[str, str] = {}
    for term in terms:
        unique.setdefault(_key(term), term)
    return list(unique.values())


def names_offer_term(text: str, requirement: Requirement) -> bool:
    return any(mentions_term(text, term) for term in offer_terms(requirement))


def is_implicit(result: RequirementMatch) -> bool:
    """Compétence technique « démontré » ou « solide » dont aucune preuve citée ne nomme un
    terme de l'offre."""
    return (
        result.status == "ok"
        and result.requirement.category == "compétence technique"
        and result.level in ("démontré", "solide")
        and bool(result.evidence)
        and not any(names_offer_term(e.quote, result.requirement) for e in result.evidence)
    )


def to_describe(result: RequirementMatch) -> bool:
    """Compétence technique « mentionné » ou « non démontré » : carte « si vous l'avez
    réellement pratiqué, décrivez-le », sans texte proposé."""
    return (
        result.status == "ok"
        and result.requirement.category == "compétence technique"
        and result.level in ("mentionné", "non démontré")
    )
