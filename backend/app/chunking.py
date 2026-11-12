"""Découpage du CV en chunks qui suivent sa structure : sections, puis items.

Un item (poste, projet, diplôme…) garde toujours son intitulé, sa date et ses puces.
RecursiveCharacterTextSplitter n'intervient qu'en dernier recours, pour un item trop long.
Le texte est anonymisé avant découpage : aucun chunk ne contient d'email, de téléphone,
d'URL ni le nom du candidat.
"""

import re
import unicodedata

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.pii import redact_pii
from app.schemas import Chunk

HEADER_SECTION = "EN-TÊTE"
NAME_PLACEHOLDER = "[NAME]"

# Mots-clés (sans accents) dont l'un doit figurer dans le titre de la première section :
# avant elle, une ligne en majuscules peut être le nom du candidat.
_SECTION_KEYWORDS = {
    "EXPERIENCE",
    "EXPERIENCES",
    "FORMATION",
    "FORMATIONS",
    "EDUCATION",
    "ETUDES",
    "COMPETENCES",
    "SKILLS",
    "PROJETS",
    "PROJECTS",
    "LANGUES",
    "LANGUAGES",
    "ENGAGEMENT",
    "ENGAGEMENTS",
    "BENEVOLAT",
    "PROFIL",
    "PROFILE",
    "RESUME",
    "SUMMARY",
    "CERTIFICATIONS",
    "PUBLICATIONS",
    "INTERETS",
    "INTERESTS",
    "LOISIRS",
    "STAGES",
    "PARCOURS",
    "REALISATIONS",
}
# Un titre ne contient ni chiffres ni séparateurs de liste ("AWS · GCP" n'en est pas un).
_HEADING = re.compile(r"[^\W\d_][^\d·,|:/()]{2,59}")
_YEAR_OR_ONGOING = re.compile(r"\b(?:19|20)\d{2}\b|\b(?:en cours|présent|aujourd'hui|present|current)\b", re.IGNORECASE)
_LIST_SEPARATORS = re.compile(r"\s*[·,|]\s*")
_MAX_DATE_LINE_CHARS = 40
_MAX_TAG_CHARS = 30
_NAME_MAX_WORDS = 5


def chunk_cv(text: str, max_chars: int = 800, overlap: int = 80) -> list[Chunk]:
    text = redact_pii(text)
    sections = _split_sections(_paragraphs(text))
    if sections and sections[0][0] == HEADER_SECTION:
        sections = _redact_name(sections)

    chunks: list[Chunk] = []
    for section, paragraphs in sections:
        items = [paragraphs] if section == HEADER_SECTION else _split_items(paragraphs)
        for item in items:
            for piece in _fit(item, max_chars, overlap):
                chunks.append(Chunk(section=section, index=len(chunks), text=piece))
    return chunks


def _paragraphs(text: str) -> list[str]:
    """Paragraphes (séparés par une ligne vide) ; un titre en tête de paragraphe en est
    détaché, pour ne pas dépendre du découpage en blocs du PDF."""
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        first, _, rest = block.strip().partition("\n")
        if rest and _looks_like_heading(first):
            paragraphs += [first, rest.strip()]
        elif block.strip():
            paragraphs.append(block.strip())
    return paragraphs


def _split_sections(paragraphs: list[str]) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    current: tuple[str, list[str]] = (HEADER_SECTION, [])
    for paragraph in paragraphs:
        is_first_section = current[0] == HEADER_SECTION
        if _looks_like_heading(paragraph) and (not is_first_section or _has_keyword(paragraph)):
            if current[1]:
                sections.append(current)
            current = (paragraph, [])
        else:
            current[1].append(paragraph)
    if current[1]:
        sections.append(current)
    return sections


def _split_items(paragraphs: list[str]) -> list[list[str]]:
    """Un paragraphe qui n'est pas une puce ouvre un item, sauf une date seule (rattachée à
    l'intitulé) ou une ligne de technologies qui suit directement une puce. Un item se
    termine par au plus une ligne de technologies : après elle, un intitulé court de la
    forme « Poste · Entreprise » ouvre bien un nouvel item."""
    items: list[list[str]] = []
    for paragraph in paragraphs:
        current = items[-1] if items else None
        if current is not None and (
            _is_bullet(paragraph) or _is_date_line(paragraph) or (_is_bullet(current[-1]) and _is_tag_line(paragraph))
        ):
            current.append(paragraph)
        else:
            items.append([paragraph])
    return items


def _fit(item: list[str], max_chars: int, overlap: int) -> list[str]:
    """L'item entier s'il tient, sinon des morceaux qui répètent tous l'intitulé."""
    text = "\n".join(item)
    if len(text) <= max_chars:
        return [text]
    title, body = item[0], "\n".join(item[1:])
    if not body or len(title) > max_chars // 2:
        title, body = "", text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars - len(title) - 1,
        chunk_overlap=overlap,
        separators=["\n- ", "\n", ". ", " ", ""],
        keep_separator="start",
    )
    return [f"{title}\n{piece.strip()}".strip() for piece in splitter.split_text(body)]


def _redact_name(sections: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
    """Remplace la première ligne de l'en-tête (le nom) par [NAME], partout dans le CV."""
    header = sections[0][1]
    name = header[0].split("\n", 1)[0].strip()
    if not name or len(name.split()) > _NAME_MAX_WORDS or re.search(r"[\d@\[]", name):
        return sections
    pattern = re.compile(re.escape(name), re.IGNORECASE)
    return [(section, [pattern.sub(NAME_PLACEHOLDER, p) for p in paragraphs]) for section, paragraphs in sections]


def _looks_like_heading(paragraph: str) -> bool:
    return (
        "\n" not in paragraph
        and paragraph == paragraph.upper()
        and _HEADING.fullmatch(paragraph) is not None
        and any(len(word) >= 4 for word in re.findall(r"[^\W\d_]+", paragraph))
    )


def _has_keyword(heading: str) -> bool:
    plain = unicodedata.normalize("NFKD", heading).encode("ascii", "ignore").decode()
    return any(word in _SECTION_KEYWORDS for word in re.findall(r"[A-Z]+", plain))


def _is_bullet(paragraph: str) -> bool:
    return paragraph.startswith("- ")


def _is_date_line(paragraph: str) -> bool:
    return (
        "\n" not in paragraph
        and len(paragraph) <= _MAX_DATE_LINE_CHARS
        and _YEAR_OR_ONGOING.search(paragraph) is not None
    )


def _is_tag_line(paragraph: str) -> bool:
    if "\n" in paragraph:
        return False
    tags = _LIST_SEPARATORS.split(paragraph)
    return len(tags) >= 2 and all(0 < len(tag) <= _MAX_TAG_CHARS for tag in tags)
