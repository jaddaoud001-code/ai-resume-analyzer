"""Certifications et scores officiels de langue cités dans une preuve.

Règle appliquée par le code (et non laissée au LLM, qui l'a ignorée) : pour une exigence de
catégorie « langue », une preuve valide qui cite un score ou un niveau officiel dans la
langue demandée vaut au moins « démontré ». Un niveau seulement déclaré (« Anglais : B2 »)
n'est pas officiel ; une année (« B2 (2023) ») n'est pas un score."""

import re
import unicodedata

from app.schemas import Requirement

# Langues reconnues (formes normalisées : minuscules, sans accents).
LANGUAGES: dict[str, tuple[str, ...]] = {
    "anglais": ("anglais", "english"),
    "francais": ("francais", "french", "fle"),
    "allemand": ("allemand", "german", "deutsch"),
    "espagnol": ("espagnol", "spanish", "espanol", "castillan"),
    "italien": ("italien", "italian"),
    "portugais": ("portugais", "portuguese"),
    "neerlandais": ("neerlandais", "dutch"),
    "chinois": ("chinois", "mandarin", "chinese"),
    "japonais": ("japonais", "japanese"),
    "arabe": ("arabe", "arabic"),
    "russe": ("russe", "russian"),
}

_LEVEL = r"[abc][12]"
# (motif sur le texte normalisé, langue certifiée ; None : à lire dans le segment)
_CERTIFICATIONS: list[tuple[re.Pattern[str], str | None]] = [
    # Score chiffré exigé : « préparation au TOEIC » n'est pas un score.
    (re.compile(r"\b(?:toeic|toefl(?: ibt)?|ielts)\b\D{0,15}\d"), "anglais"),
    (re.compile(rf"\bcambridge\b.{{0,30}}\b(?:{_LEVEL}|fce|cae|cpe|first|advanced|proficiency)\b"), "anglais"),
    # Diplômes officiels par eux-mêmes, niveau collé ou non.
    (re.compile(r"\b(?:delf|dalf)\b"), "francais"),
    (re.compile(r"\b(?:tcf|tef)\b\D{0,15}\d"), "francais"),
    (re.compile(r"\bdele\b"), "espagnol"),
    (re.compile(r"\bgoethe zertifikat\b|\bgoethe\W{0,3}[abc][12]\b"), "allemand"),
]
# Niveau CECRL accompagné d'un score (« B2, score 82/100 », « C1 (92 %) ») ; pas d'année.
_CEFR = re.compile(rf"\b{_LEVEL}\b")
_SCORE = re.compile(r"\bscore\W{0,3}\d{2,3}\b|\b\d{2,3}(?:[.,]\d)?\s*(?:/\s*\d{2,3}\b|%|points?\b|pts\b)")
# Appliqué au texte brut, avant normalisation (qui efface « · » et « – ») ; une virgule
# décimale (« IELTS 7,5 ») ne sépare pas.
_SEGMENT_SEPARATORS = re.compile(r"[·;\n|•…]|,(?!\d)")


def _normalize(text: str) -> str:
    plain = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return re.sub(r"[\s\-]+", " ", plain)


def _languages_in(text: str) -> set[str]:
    words = set(re.findall(r"[a-z]+", text))
    return {language for language, names in LANGUAGES.items() if words & set(names)}


def requirement_language(requirement: Requirement) -> str | None:
    """Langue demandée, d'après le libellé puis les requêtes ; None si elle n'est pas nommée
    (« Langues étrangères ») ou ambiguë."""
    for text in (requirement.label, " ".join(requirement.search_queries)):
        found = _languages_in(_normalize(text))
        if len(found) == 1:
            return found.pop()
        if found:
            return None
    return None


def certified_languages(quote: str) -> list[str | None]:
    """Langue de chaque certification ou score officiel de la citation (None : langue non
    identifiable). Chaque segment (« Anglais : B1 (TOEIC 785) » dans « … · Allemand :
    notions ») est lu à part, pour ne pas prêter la certification d'une langue à une autre."""
    found: list[str | None] = []
    for raw in _SEGMENT_SEPARATORS.split(quote):
        segment = _normalize(raw)
        named = _languages_in(segment)
        own = next((language for pattern, language in _CERTIFICATIONS if pattern.search(segment)), None)
        if own is not None:
            found.append(own)
        elif _CEFR.search(segment) and _SCORE.search(segment):
            found.append(named.pop() if len(named) == 1 else None)
    return found


def proves_language(quote: str, requirement: Requirement) -> bool:
    """La citation contient une certification ou un score officiel dans la langue demandée
    (dans n'importe quelle langue si l'exigence n'en nomme pas)."""
    wanted = requirement_language(requirement)
    return any(wanted is None or language == wanted for language in certified_languages(quote))
