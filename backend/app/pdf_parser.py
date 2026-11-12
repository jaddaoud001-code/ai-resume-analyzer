"""Extraction et nettoyage du texte d'un CV au format PDF (PyMuPDF).

Tout se fait en mémoire à partir des octets uploadés : rien n'est écrit sur disque.
"""

import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Any

import pymupdf

from app.config import Settings, get_settings
from app.schemas import ParsedCV


class PDFParseError(Exception):
    """Erreur d'extraction ; le message est destiné à l'utilisateur."""


class InvalidPDFError(PDFParseError):
    pass


class EncryptedPDFError(PDFParseError):
    pass


class NoExtractableTextError(PDFParseError):
    pass


class PDFTooLargeError(PDFParseError):
    pass


class TooManyPagesError(PDFParseError):
    pass


def extract_text(pdf_bytes: bytes, settings: Settings | None = None) -> ParsedCV:
    settings = settings or get_settings()

    if len(pdf_bytes) > settings.max_pdf_size_bytes:
        max_mb = settings.max_pdf_size_bytes / (1024 * 1024)
        raise PDFTooLargeError(f"Le fichier dépasse la taille maximale de {max_mb:g} Mo.")

    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise InvalidPDFError("Le fichier n'est pas un PDF valide ou est corrompu.") from exc

    with doc:
        if doc.needs_pass:
            raise EncryptedPDFError("Le PDF est protégé par un mot de passe.")
        if not doc.is_pdf or doc.page_count == 0:
            raise InvalidPDFError("Le fichier n'est pas un PDF valide ou est corrompu.")
        if doc.page_count > settings.max_pdf_pages:
            raise TooManyPagesError(f"Le PDF contient {doc.page_count} pages (maximum : {settings.max_pdf_pages}).")

        try:
            pages = [_page_text(page) for page in doc]
        except RuntimeError as exc:
            raise InvalidPDFError("Le PDF est corrompu : lecture du texte impossible.") from exc
        page_count = doc.page_count

    text = clean_text("\n\n".join(pages))
    if len(text) < settings.min_text_chars:
        raise NoExtractableTextError(
            "Aucun texte exploitable n'a été trouvé dans le PDF (document scanné ou composé d'images ?)."
        )

    return ParsedCV(text=text, page_count=page_count, char_count=len(text))


# --- Ordre de lecture -------------------------------------------------------

# Ligatures développées (pas de PRESERVE_LIGATURES), espaces conservés pour clean_text.
_TEXT_FLAGS = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP
# La gouttière entre deux colonnes est cherchée dans cette plage (fraction de la largeur) :
# une colonne latérale étroite (~30 % de la page) doit être reconnue.
_GUTTER_RANGE = (0.2, 0.8)
# Tolérance (points) sur le bord droit d'un bloc de la colonne de gauche.
_EDGE_TOLERANCE_PT = 2.0
# Écart entre deux caractères (en fraction de la taille de police) au-delà duquel on
# insère un espace absent du PDF. Au-dessus de l'interlettrage mesuré sur des titres
# espacés (~0,095 em), en dessous du seuil à partir duquel MuPDF en insère (~0,15 em).
_WORD_GAP_EM = 0.12
# Distance maximale entre une puce isolée et le début du texte qu'elle précède.
_BULLET_MAX_GAP_EM = 3.0

# Puces courantes, dont celles des polices Symbol/Wingdings exportées par Word.
_BULLET_CHARS = "•●▪■◦‣∙·–*"


@dataclass(frozen=True)
class _Line:
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    text: str


@dataclass(frozen=True)
class _Block:
    lines: tuple[_Line, ...]

    @property
    def x0(self) -> float:
        return min(line.x0 for line in self.lines)

    @property
    def y0(self) -> float:
        return min(line.y0 for line in self.lines)

    @property
    def x1(self) -> float:
        return max(line.x1 for line in self.lines)

    @property
    def y1(self) -> float:
        return max(line.y1 for line in self.lines)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


def _page_text(page: pymupdf.Page) -> str:
    blocks = _attach_bullets(_extract_blocks(page))
    return "\n\n".join(_reading_order(blocks, page.rect.width))


def _extract_blocks(page: pymupdf.Page) -> list[_Block]:
    blocks: list[_Block] = []
    for raw_block in page.get_text("rawdict", flags=_TEXT_FLAGS)["blocks"]:
        if raw_block["type"] != 0:
            continue
        lines = tuple(
            _Line(*raw_line["bbox"], size=max(s["size"] for s in raw_line["spans"]), text=text)
            for raw_line in raw_block["lines"]
            if raw_line["spans"] and (text := _line_text(raw_line).strip())
        )
        if lines:
            blocks.append(_Block(lines))
    return blocks


def _line_text(raw_line: dict[str, Any]) -> str:
    """Reconstruit le texte d'une ligne caractère par caractère, en insérant un espace
    quand un écart horizontal sépare deux caractères (même au sein d'un span).

    Les espaces présents dans le PDF sont toujours conservés, quelle que soit leur
    largeur : la règle d'écart ne fait qu'ajouter des espaces manquants."""
    horizontal = raw_line["dir"][0] > 0 and abs(raw_line["dir"][1]) < 1e-3
    parts: list[str] = []
    previous: tuple[dict[str, Any], float] | None = None
    for span in raw_line["spans"]:
        for char in span["chars"]:
            if horizontal and previous is not None:
                prev_char, prev_size = previous
                gap = char["bbox"][0] - prev_char["bbox"][2]
                if (
                    not prev_char["c"].isspace()
                    and not char["c"].isspace()
                    and gap > _WORD_GAP_EM * min(prev_size, span["size"])
                ):
                    parts.append(" ")
            parts.append(char["c"])
            previous = (char, span["size"])
    return "".join(parts)


def _is_bullet(text: str) -> bool:
    return all(ch in _BULLET_CHARS for ch in text.replace(" ", ""))


def _attach_bullets(blocks: list[_Block]) -> list[_Block]:
    """Rattache chaque ligne réduite à une puce au texte situé à sa droite sur la même
    hauteur (préfixe "- "), ou la supprime si aucun texte ne correspond."""
    positions = [(b, i) for b, block in enumerate(blocks) for i in range(len(block.lines))]
    bullets = [pos for pos in positions if _is_bullet(blocks[pos[0]].lines[pos[1]].text)]
    if not bullets:
        return blocks
    candidates = [pos for pos in positions if pos not in bullets]

    prefixed: set[tuple[int, int]] = set()
    for b, i in bullets:
        bullet = blocks[b].lines[i]
        center_y = (bullet.y0 + bullet.y1) / 2
        max_gap = _BULLET_MAX_GAP_EM * bullet.size
        aligned = [
            pos
            for pos in candidates
            if pos not in prefixed
            and (line := blocks[pos[0]].lines[pos[1]]).y0 <= center_y <= line.y1
            and line.x0 >= (bullet.x0 + bullet.x1) / 2
            and line.x0 - bullet.x1 <= max_gap
        ]
        if aligned:
            prefixed.add(min(aligned, key=lambda pos: blocks[pos[0]].lines[pos[1]].x0))

    result: list[_Block] = []
    for b, block in enumerate(blocks):
        lines = tuple(
            replace(line, text=f"- {line.text}") if (b, i) in prefixed else line
            for i, line in enumerate(block.lines)
            if (b, i) not in bullets
        )
        if lines:
            result.append(_Block(lines))
    return result


def _column_split(blocks: list[_Block], page_width: float) -> float | None:
    """Abscisse de la gouttière entre deux colonnes, ou None pour une seule colonne.

    Deux colonnes sont côte à côte : un bloc de gauche et un bloc de droite partagent une
    même hauteur. Candidates : les bords droits des blocs situés dans _GUTTER_RANGE ; on
    retient celle qui met face à face le plus de blocs (à hauteur égale, un de chaque côté),
    puis, à égalité, celle que traversent le moins de blocs. Les colonnes ne sont pas
    supposées symétriques (colonne latérale étroite), et une page à une seule colonne, dont
    les lignes courtes n'ont rien en face d'elles, n'est pas découpée."""
    low, high = (page_width * bound for bound in _GUTTER_RANGE)

    def facing(a: _Block, b: _Block) -> bool:
        return a.y0 < b.y1 and b.y0 < a.y1

    best: tuple[int, int, float] | None = None
    for x in sorted({b.x1 for b in blocks if low <= b.x1 <= high}):
        left = [b for b in blocks if b.x1 <= x + _EDGE_TOLERANCE_PT]
        right = [b for b in blocks if b.x0 > x + _EDGE_TOLERANCE_PT]
        paired = sum(1 for a in left if any(facing(a, b) for b in right)) + sum(
            1 for b in right if any(facing(b, a) for a in left)
        )
        if not paired:
            continue
        key = (paired, -(len(blocks) - len(left) - len(right)), x)
        if best is None or key[:2] > best[:2]:
            best = key
    return None if best is None else best[2]


def _reading_order(blocks: list[_Block], page_width: float) -> list[str]:
    """Ordonne les blocs : colonne gauche puis droite, les blocs qui traversent la
    gouttière (titres, en-têtes pleine largeur) servant de séparateurs entre segments."""
    by_position = sorted(blocks, key=lambda b: (b.y0, b.x0))
    split = _column_split(blocks, page_width)
    if split is None:
        return [b.text for b in by_position]

    ordered: list[str] = []
    left: list[str] = []
    right: list[str] = []
    for block in by_position:
        if block.x1 <= split + _EDGE_TOLERANCE_PT:
            left.append(block.text)
        elif block.x0 > split + _EDGE_TOLERANCE_PT:
            right.append(block.text)
        else:
            ordered += left + right + [block.text]
            left, right = [], []
    return ordered + left + right


# --- Nettoyage --------------------------------------------------------------

_LINE_BREAKS = str.maketrans({"\r": "\n", "\v": "\n", "\f": "\n", " ": "\n", " ": "\n"})
_SPACES = re.compile(r"[^\S\n]+")  # tout blanc horizontal (tab, insécable…)
# Puce en début de ligne ; si elle est seule sur sa ligne, on la colle à la ligne
# suivante, sauf si celle-ci est vide ou commence elle-même par une puce.
_BULLET = re.compile(rf"^[{re.escape(_BULLET_CHARS)}] *(?:\n(?![{re.escape(_BULLET_CHARS)}\n]))?", re.MULTILINE)
_HYPHEN_BREAK = re.compile(r"(?<=\w)-\n(?=\w)")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").translate(_LINE_BREAKS)
    # Caractères de contrôle et de format (tiret conditionnel, zéro-largeur, BOM…).
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or unicodedata.category(ch) not in ("Cc", "Cf"))
    text = "\n".join(_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    # Mot coupé en fin de ligne : on recolle en gardant le tiret (full-\nstack -> full-stack).
    text = _HYPHEN_BREAK.sub("-", text)
    text = _BULLET.sub("- ", text)
    # Puce restée sans texte : non fiable, on la supprime.
    text = "\n".join(line for line in text.split("\n") if line.strip() != "-")
    return _BLANK_LINES.sub("\n\n", text).strip()
