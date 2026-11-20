"""Citations avec « … » et morceaux courts (moins de 3 mots), depuis le 2026-11-21 : un
morceau court est accepté s'il est ancré (début de ligne ou de puce du chunk, ou au plus
MATCH_QUOTE_SHORT_GAP caractères après le morceau précédent) et si au moins un morceau fait
3 mots ou plus. Rejoue les citations rejetées des runs du 18 et du 20 novembre 2026."""

from pathlib import Path

import pytest

from app.chunking import chunk_cv
from app.matcher import quote_in_chunk
from app.pdf_parser import extract_text
from tests.conftest import make_settings

CVS = Path(__file__).resolve().parents[1] / "eval" / "cvs"


def _chunks(name: str) -> dict[int, str]:
    return {c.index: c.text for c in chunk_cv(extract_text((CVS / name).read_bytes(), make_settings()).text)}


BACKEND = _chunks("backend_cv.pdf")
JAVA = _chunks("java_cv.pdf")


@pytest.mark.parametrize(
    ("chunk", "quote"),
    [
        # 18 et 20 nov., CV backend : intitulés abrégés à 2 mots (début de ligne du chunk).
        (BACKEND[5], "Développeuse Python … Sept. 2021 – Déc. 2022 … Scripts Python d'import et de nettoyage"),
        (BACKEND[4], "Développeuse freelance … Janv. 2023 – Déc. 2023 … Applications web Django …"),
        (BACKEND[2], "Ingénieure logicielle … Févr. 2025 – en cours … Développement des endpoints FastAPI"),
        (
            BACKEND[2],
            "Ingénieure logicielle … Développement des endpoints FastAPI … Mise en production des services sur un "
            "cluster Kubernetes …",
        ),
        # « Bases PostgreSQL » : 135 caractères après le morceau précédent, mais il commence une puce.
        (BACKEND[4], "Développeuse freelance … Applications web Django … Bases PostgreSQL …"),
        (BACKEND[4], "Développeuse freelance · missions pour des PME … Applications web Django … Bases PostgreSQL …"),
        # Morceaux courts tout proches du précédent (milieu de ligne).
        (
            BACKEND[2],
            "Mise en production des services sur un cluster Kubernetes managé … images Docker … manifestes Helm … "
            "déploiement à chaque fusion",
        ),
        (BACKEND[2], "Pipelines GitHub Actions … tests … à chaque fusion sur la branche principale"),
    ],
)
def test_rejected_quotes_of_the_runs_are_now_accepted(chunk: str, quote: str) -> None:
    assert quote_in_chunk(quote, chunk)


@pytest.mark.parametrize(
    ("chunk", "quote", "reason"),
    [
        # Runs du 18 : aucun morceau de 3 mots ou plus.
        (JAVA[7], "Langages … SQL …", "aucun morceau de 3 mots"),
        (JAVA[2], "Requêtes SQL … PostgreSQL …", "aucun morceau de 3 mots"),
        # Mots absents du chunk.
        (BACKEND[4], "Développeuse freelance … Applications web Django … Kafka", "absent"),
        # Mots éloignés en milieu de ligne : assemblage rejeté.
        (BACKEND[2], "Développement des endpoints FastAPI … latence", "éloigné"),
        (BACKEND[2], "chauffeurs … Tableaux de bord Grafana", "éloigné (premier morceau)"),
        # Ordre inversé.
        (BACKEND[4], "Applications web Django … Développeuse freelance", "ordre"),
    ],
)
def test_far_or_absent_words_stay_rejected(chunk: str, quote: str, reason: str) -> None:
    assert not quote_in_chunk(quote, chunk), reason


def test_gap_is_configurable() -> None:
    chunk = "Poste · Société\n- Développement des services de facturation pour les clients professionnels, avec Kafka"
    quote = "Développement des services … Kafka"  # « Kafka » en milieu de ligne, ~60 caractères plus loin

    assert quote_in_chunk(quote, chunk, max_short_gap=80)
    assert not quote_in_chunk(quote, chunk, max_short_gap=40)


def test_line_start_anchor_ignores_the_gap() -> None:
    chunk = "Poste · Société\n- Une longue puce qui décrit en détail tout le travail mené sur la plateforme\n- Kafka"

    assert quote_in_chunk("Une longue puce qui décrit … Kafka", chunk, max_short_gap=0)


def test_default_gap_setting() -> None:
    assert make_settings().match_quote_short_gap == 80
