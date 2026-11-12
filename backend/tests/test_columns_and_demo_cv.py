"""Ordre de lecture des colonnes (gouttière détectée, pas supposée au milieu de la page),
intitulés « Poste · Entreprise » après une ligne de technologies, et CV de démonstration."""

import sys
from pathlib import Path

import pymupdf

from app.chunking import HEADER_SECTION, NAME_PLACEHOLDER, chunk_cv
from app.pdf_parser import extract_text
from tests.conftest import make_settings

DEMO_DIR = Path(__file__).resolve().parents[1] / "eval" / "cvs"
DEMO_PDF = DEMO_DIR / "demo_cv.pdf"


def _pdf(fill: object) -> bytes:
    document = pymupdf.open()
    try:
        fill(document.new_page())  # type: ignore[operator]
        return document.tobytes()
    finally:
        document.close()


def _text(pdf: bytes) -> str:
    return extract_text(pdf, make_settings()).text


class TestColumnSplit:
    def test_narrow_sidebar_is_read_before_the_main_column(self) -> None:
        """Colonne latérale étroite (bord droit à ~35 % de la page) : la colonne principale
        commence bien avant le milieu de la page et ne doit pas être prise pour du texte
        pleine largeur."""

        sidebar = ["COMPÉTENCES", "Python", "SQL", "LANGUES", "Anglais"]
        main = [
            "PROFIL",
            "Ingénieur données passionné par les pipelines fiables.",
            "EXPÉRIENCE",
            "Ingénieur données · Société Exemple",
            "2022 - 2024",
        ]

        def column(lines: list[str]) -> str:
            # Un paragraphe = un bloc ; les deux colonnes s'entrelacent en hauteur.
            return "".join(f'<p style="font-size:10px;margin:0 0 14px 0">{line}</p>' for line in lines)

        def fill(page: pymupdf.Page) -> None:
            page.insert_text((40, 60), "Alex Exemple", fontsize=16)
            page.insert_htmlbox(pymupdf.Rect(40, 100, 200, 400), column(sidebar))
            page.insert_htmlbox(pymupdf.Rect(230, 104, 560, 400), column(main))

        text = _text(_pdf(fill))

        order = [
            text.index(marker) for marker in ("COMPÉTENCES", "Python", "LANGUES", "Anglais", "PROFIL", "EXPÉRIENCE")
        ]
        assert order == sorted(order)

    def test_single_column_page_with_a_two_column_footer(self) -> None:
        """Page à une colonne terminée par deux blocs côte à côte : les lignes courtes de la
        colonne unique ne doivent pas déplacer la gouttière."""

        def fill(page: pymupdf.Page) -> None:
            page.insert_text((40, 60), "Alex Exemple", fontsize=16)
            page.insert_text(
                (40, 100),
                "EXPÉRIENCE\nIngénieur logiciel · Société Exemple\n2020 - 2024\n"
                "- Conception d'un service de facturation en Go, utilisé par toutes les équipes du groupe.\n"
                "FORMATION\nDiplôme d'ingénieur\n2015 - 2020",
                fontsize=10,
            )
            page.insert_text((40, 400), "LANGUES\nAnglais courant, allemand scolaire", fontsize=10)
            page.insert_text((320, 400), "ENGAGEMENT\nTrésorier d'une association sportive", fontsize=10)

        chunks = chunk_cv(_text(_pdf(fill)))

        by_text = {c.text.splitlines()[0]: c.section for c in chunks}
        assert by_text["Anglais courant, allemand scolaire"] == "LANGUES"
        assert by_text["Trésorier d'une association sportive"] == "ENGAGEMENT"


def test_job_title_after_a_tag_line_opens_a_new_item() -> None:
    text = (
        "Alex Exemple\n\nEXPÉRIENCE\n\nStage data · Société Alpha\n\n2024 - 2025\n\n"
        "- Tableaux de bord de suivi.\n\nSQL · Python\n\n"
        "Alternance data · Société Beta\n\n2023 - 2024\n\n- Nettoyage de données."
    )

    chunks = chunk_cv(text)

    alpha = next(c for c in chunks if "Société Alpha" in c.text)
    beta = next(c for c in chunks if "Société Beta" in c.text)
    assert alpha is not beta
    assert alpha.text.endswith("SQL · Python")
    assert beta.text.startswith("Alternance data · Société Beta")


class TestDemoCV:
    def test_pdf_is_up_to_date_with_its_script(self) -> None:
        sys.path.insert(0, str(DEMO_DIR))
        try:
            from make_demo_cv import build
        finally:
            sys.path.remove(str(DEMO_DIR))

        assert build() == DEMO_PDF.read_bytes(), "régénérer : .venv/bin/python eval/cvs/make_demo_cv.py"

    def test_two_columns_sections_and_items(self) -> None:
        chunks = chunk_cv(_text(DEMO_PDF.read_bytes()))

        sections = list(dict.fromkeys(c.section for c in chunks))
        assert sections == [
            HEADER_SECTION,
            "COMPÉTENCES",
            "LANGUES",
            "CENTRES D'INTÉRÊT",
            "PROFIL",
            "EXPÉRIENCE PROFESSIONNELLE",
            "PROJETS",
            "FORMATION",
        ]
        experiences = [c for c in chunks if c.section == "EXPÉRIENCE PROFESSIONNELLE"]
        assert [c.text.splitlines()[0] for c in experiences] == [
            "Stage ingénieure IA · Lumen Analytique",
            "Alternance développeuse data · Coopérative Ardoise",
        ]
        assert experiences[0].text.endswith("Python · FastAPI · FAISS · Docker")

    def test_contact_details_are_masked(self) -> None:
        chunks = chunk_cv(_text(DEMO_PDF.read_bytes()))
        everything = "\n".join(c.text for c in chunks)

        assert chunks[0].text.startswith(NAME_PLACEHOLDER)
        for placeholder in ("[PHONE]", "[EMAIL]", "[URL]"):
            assert placeholder in chunks[0].text
        for personal in ("Camille Lefèvre", "39 98 42 17", "example.com"):
            assert personal not in everything
