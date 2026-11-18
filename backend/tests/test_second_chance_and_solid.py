"""Seconde chance élargie aux durées, et plafond de « solide » (au moins deux expériences ou
projets distincts). Faux LLM, aucun appel réseau."""

import asyncio
from datetime import date
from pathlib import Path

from app.chunking import chunk_cv
from app.llm_calls import RateLimiter
from app.matcher import evaluate_requirement, experience_items, verify_assessment
from app.pdf_parser import extract_text
from app.schemas import Chunk, Requirement, RequirementMatch, SearchResult
from tests.conftest import FakeStructuredChatModel, make_settings
from tests.test_matcher import _assess

EVAL_TODAY = date(2026, 11, 14)  # date fixe de l'évaluation du matching
SECTIONS = ["EXPERIENCE", "PROJET", "STAGE"]
BACKEND_PDF = Path(__file__).resolve().parents[1] / "eval" / "cvs" / "backend_cv.pdf"


def _backend_chunks() -> dict[int, Chunk]:
    text = extract_text(BACKEND_PDF.read_bytes(), make_settings()).text
    return {c.index: c for c in chunk_cv(text)}


def _python(min_years: int | None = 3) -> Requirement:
    return Requirement(
        id="R2",
        label="Python",
        category="compétence technique",
        importance="obligatoire",
        min_years=min_years,
        search_queries=["Python", "développement Python"],
    )


def _evaluate(requirement: Requirement, chunk_ids: list[int], responses: list[object]) -> tuple[RequirementMatch, int]:
    chunks = _backend_chunks()
    retrieved = [SearchResult(chunk=chunks[i], score=0.9 - n / 100) for n, i in enumerate(chunk_ids)]
    settings = make_settings()
    model = FakeStructuredChatModel(responses=list(responses))
    result = asyncio.run(
        evaluate_requirement(requirement, retrieved, model, RateLimiter.from_settings(settings), settings, EVAL_TODAY)
    )
    return result, len(model.calls)


# Première réponse du run du 2026-11-18 (CV backend × logistique, R2 Python) : les trois
# citations d'expérience commencent par un intitulé abrégé à 2 mots, donc rejetées ; seule la
# liste de compétences reste valide, et les trois périodes sont perdues.
FIRST_PYTHON = _assess(
    "démontré",
    (7, "Langages : Python, SQL, Go"),
    (5, "Développeuse Python … Sept. 2021 – Déc. 2022 … Scripts Python d'import et de nettoyage"),
    (4, "Développeuse freelance … Janv. 2023 – Déc. 2023 … Applications web Django"),
    (2, "Ingénieure logicielle … Févr. 2025 – en cours … Développement des endpoints FastAPI"),
    periods=[("2021-09", "2022-12"), ("2023-01", "2023-12"), ("2025-02", "en cours")],
)
CORRECTED_PYTHON = _assess(
    "solide",
    (5, "Scripts Python d'import et de nettoyage des relevés de trafic"),
    (4, "Applications web Django pour trois clients"),
    (2, "Développement des endpoints FastAPI consommés par l'application mobile des chauffeurs"),
    periods=[("2021-09", "2022-12"), ("2023-01", "2023-12"), ("2025-02", "en cours")],
)


class TestExtendedSecondChance:
    def test_replays_the_backend_cv_python_case(self) -> None:
        result, calls = _evaluate(_python(), [5, 0, 4, 7, 2], [FIRST_PYTHON, CORRECTED_PYTHON])

        assert calls == 2 and result.second_chance
        # 16 + 12 + 22 mois (2021-09 → 2022-12, 2023, 2025-02 → 2026-11), trois expériences.
        assert (result.total_months, result.level, result.capped_for_duration) == (50, "solide", False)
        assert result.rejected_evidence == 3  # citations de la première réponse, conservées

    def test_without_the_second_chance_the_periods_were_lost(self) -> None:
        # Ce que le run du 2026-11-18 a donné (seconde chance seulement si tout est rejeté).
        chunks = _backend_chunks()
        shown = {i: chunks[i].text for i in (5, 0, 4, 7, 2)}

        first = verify_assessment(FIRST_PYTHON, _python(), shown, EVAL_TODAY)

        assert (first.level, first.total_months, first.capped_for_duration) == ("mentionné", 0, True)

    def test_no_second_chance_without_a_duration(self) -> None:
        first = FIRST_PYTHON.model_copy(update={"periods": []})

        result, calls = _evaluate(_python(min_years=None), [5, 0, 4, 7, 2], [first])

        assert calls == 1 and not result.second_chance

    def test_no_second_chance_when_only_a_skill_list_quote_is_rejected(self) -> None:
        first = _assess(
            "démontré",
            (5, "Scripts Python d'import et de nettoyage des relevés de trafic"),
            (7, "Langages : Python, Go, Rust"),  # rejetée, mais hors expérience
            periods=[("2021-09", "2022-12")],
        )

        result, calls = _evaluate(_python(min_years=1), [5, 7], [first])

        assert calls == 1 and not result.second_chance and result.rejected_evidence == 1

    def test_at_most_one_extra_call(self) -> None:
        result, calls = _evaluate(_python(), [5, 0, 4, 7, 2], [FIRST_PYTHON, FIRST_PYTHON])

        assert calls == 2 and result.second_chance

    def test_a_worse_second_answer_keeps_the_first(self) -> None:
        # Seconde réponse sans aucune preuve valide : la première (avec sa preuve) est gardée.
        worse = _assess("solide", (2, "Expert Python depuis dix ans"))

        result, calls = _evaluate(_python(), [5, 0, 4, 7, 2], [FIRST_PYTHON, worse])

        assert calls == 2 and result.second_chance
        assert [e.chunk_id for e in result.evidence] == [7]  # preuve de la première réponse
        assert result.rejected_evidence == 4  # rejetées des deux réponses, sans doublon


class TestSolidCap:
    def test_split_experience_counts_as_one_item(self) -> None:
        chunks = _backend_chunks()
        items = experience_items([SearchResult(chunk=chunks[i], score=0.5) for i in (2, 3, 4, 7)], SECTIONS)

        assert items[2] == items[3]  # deux morceaux de la même expérience
        assert items[2] != items[4] and items[7] is None  # autre expérience ; liste de compétences

    def test_solid_needs_two_distinct_items(self) -> None:
        # Deux morceaux de la même expérience découpée : un seul item.
        one_item = _assess(
            "solide",
            (2, "Tableaux de bord Grafana et alertes Prometheus"),
            (3, "documentation d'architecture rédigée en anglais"),
        )

        result, _ = _evaluate(_python(min_years=None), [2, 3, 7], [one_item])

        assert (result.level, result.llm_level, result.capped_single_item) == ("démontré", "solide", True)

    def test_skill_list_does_not_count_as_an_item(self) -> None:
        answer = _assess("solide", (2, "Tests pytest sur les endpoints critiques"), (7, "Langages : Python, SQL"))

        result, _ = _evaluate(_python(min_years=None), [2, 7], [answer])

        assert (result.level, result.capped_single_item) == ("démontré", True)

    def test_two_distinct_experiences_keep_solid(self) -> None:
        answer = _assess(
            "solide",
            (5, "Scripts Python d'import et de nettoyage des relevés de trafic"),
            (4, "Applications web Django pour trois clients"),
        )

        result, _ = _evaluate(_python(min_years=None), [5, 4], [answer])

        assert (result.level, result.capped_single_item) == ("solide", False)

    def test_lower_levels_are_untouched(self) -> None:
        result, _ = _evaluate(
            _python(min_years=None), [2], [_assess("démontré", (2, "Tests pytest sur les endpoints critiques"))]
        )

        assert (result.level, result.capped_single_item) == ("démontré", False)

    def test_without_items_the_cap_is_not_applied(self) -> None:
        # verify_assessment appelé sans les sections (outil, test) : pas de plafond.
        text = "Poste\n- Tests pytest sur les endpoints critiques"

        result = verify_assessment(
            _assess("solide", (0, "Tests pytest sur les endpoints critiques")), _python(None), {0: text}, EVAL_TODAY
        )

        assert (result.level, result.capped_single_item) == ("solide", False)


JAVA_PDF = Path(__file__).resolve().parents[1] / "eval" / "cvs" / "java_cv.pdf"


def _java_sql(responses: list[object]) -> tuple[RequirementMatch, int]:
    text = extract_text(JAVA_PDF.read_bytes(), make_settings()).text
    chunks = {c.index: c for c in chunk_cv(text)}
    retrieved = [SearchResult(chunk=chunks[i], score=0.9 - n / 100) for n, i in enumerate([7, 8, 4, 2, 3])]
    requirement = Requirement(
        id="R3",
        label="SQL et bases de données relationnelles",
        category="compétence technique",
        importance="obligatoire",
        search_queries=["SQL", "bases de données relationnelles"],
    )
    settings = make_settings()
    model = FakeStructuredChatModel(responses=list(responses))
    result = asyncio.run(
        evaluate_requirement(requirement, retrieved, model, RateLimiter.from_settings(settings), settings, EVAL_TODAY)
    )
    return result, len(model.calls)


# Première réponse du run du 2026-11-18 (CV Java × logistique, R3) : seule la citation du projet
# est valide ; celle de l'alternance (autre item) est rejetée (morceaux de moins de 3 mots).
FIRST_JAVA_SQL = _assess(
    "solide",
    (4, "base MySQL …"),
    (7, "Langages … SQL …"),
    (2, "Requêtes SQL … PostgreSQL …"),
)


class TestSolidSecondChance:
    def test_replays_the_java_cv_sql_case(self) -> None:
        corrected = _assess(
            "solide",
            (4, "Application web Spring Boot et Thymeleaf, base MySQL"),
            (2, "Requêtes SQL et index sur PostgreSQL pour accélérer les recherches de colis."),
        )

        result, calls = _java_sql([FIRST_JAVA_SQL, corrected])

        assert calls == 2 and result.second_chance
        assert (result.level, result.capped_single_item) == ("solide", False)  # projet + alternance

    def test_without_it_solid_was_capped(self) -> None:
        result, calls = _java_sql([FIRST_JAVA_SQL, FIRST_JAVA_SQL])

        # Même réponse : toujours un seul item valide, plafond maintenu ; un seul appel en plus.
        assert calls == 2 and result.second_chance
        assert (result.level, result.capped_single_item) == ("démontré", True)

    def test_no_second_chance_when_rejected_quotes_are_outside_experiences(self) -> None:
        # Seule la liste de compétences (hors expérience) a une citation rejetée.
        first = _assess("solide", (4, "base MySQL …"), (7, "Langages … SQL …"))

        result, calls = _java_sql([first])

        assert calls == 1 and not result.second_chance
        assert (result.level, result.capped_single_item) == ("démontré", True)

    def test_no_second_chance_when_the_rejected_quote_is_from_the_same_item(self) -> None:
        first = _assess("solide", (4, "base MySQL …"), (4, "projet MySQL de prêts"))

        result, calls = _java_sql([first])

        assert calls == 1 and not result.second_chance
