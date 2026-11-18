"""Prompt d'évaluation (matching) : règles sur les savoir-être et les scores officiels, version.

Le LLM est simulé dans les tests : on vérifie ici que les règles sont bien dans le prompt et
que l'exigence transmise au LLM porte sa catégorie (sans quoi la règle des savoir-être ne
pourrait pas s'appliquer). Leur effet réel se mesure avec eval/eval_matching.py."""

import hashlib
import re

from app.chunking import chunk_cv
from app.matcher import _SYSTEM_PROMPT, MATCH_PROMPT_VERSION, build_match_messages, verify_assessment
from app.schemas import Requirement, SearchResult
from tests.conftest import GENERIC_CV_TEXT
from tests.test_matcher import TODAY, _assess


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


PROMPT = _flat(_SYSTEM_PROMPT)


def test_prompt_change_requires_a_new_version() -> None:
    """Les runs de l'évaluation du matching se comparent par version du prompt. Après une
    modification voulue : incrémenter MATCH_PROMPT_VERSION et mettre à jour l'empreinte."""
    fingerprints = {4: "fd13839a18cef0f43ca9537f33670e81e46dc9dab065ed68b3066aa038f7018d"}

    assert hashlib.sha256(_SYSTEM_PROMPT.encode()).hexdigest() == fingerprints.get(MATCH_PROMPT_VERSION), (
        "prompt d'évaluation modifié : incrémenter MATCH_PROMPT_VERSION et mettre à jour l'empreinte"
    )


class TestSoftSkillRule:
    def test_soft_skill_is_demonstrated_only_when_named_or_described(self) -> None:
        assert "il n'est « démontré » que s'il est nommé ou décrit explicitement" in PROMPT
        assert "Une réalisation technique ne prouve pas à elle seule un savoir-être" in PROMPT

    def test_each_usual_soft_skill_has_a_counter_example(self) -> None:
        # Les déductions observées au run de référence (v1), sous forme générique.
        for wrong_inference in (
            "ne prouve pas la rigueur",
            "ne prouve pas l'autonomie",
            "ne prouve pas la résolution de problèmes",
            "ne prouve pas la capacité d'analyse et de synthèse",
            "ne prouve pas le travail en équipe",
        ):
            assert wrong_inference in PROMPT, wrong_inference

    def test_without_description_a_named_soft_skill_is_mentioned(self) -> None:
        assert "le savoir-être est « mentionné » si le CV l'affirme ou le nomme explicitement" in PROMPT
        assert "(dans le profil, le résumé ou une liste de qualités), sinon « non démontré »" in PROMPT

    def test_the_llm_is_told_the_requirement_is_a_soft_skill(self) -> None:
        requirement = Requirement(
            id="R1",
            label="Rigueur",
            category="savoir-être",
            importance="souhaitée",
            search_queries=["rigueur", "rigoureux"],
        )
        chunks = chunk_cv(GENERIC_CV_TEXT)

        messages = build_match_messages(requirement, [SearchResult(chunk=c, score=0.5) for c in chunks[:3]])

        assert messages[0].content == _SYSTEM_PROMPT
        assert "Catégorie : savoir-être" in str(messages[1].content)

    def test_code_still_caps_soft_skills_at_demonstrated(self) -> None:
        requirement = Requirement(
            id="R1", label="Autonomie", category="savoir-être", importance="souhaitée", search_queries=["a", "b"]
        )
        text = "Projet mené seul : conception, développement et mise en service."

        result = verify_assessment(_assess("solide", (0, text)), requirement, {0: text}, TODAY)

        assert (result.level, result.capped_soft_skill) == ("démontré", True)


class TestProfileClaimRule:
    """v3 : une affirmation du profil sans expérience qui l'illustre vaut « mentionné »."""

    def test_explicit_profile_claim_is_mentioned_for_any_requirement(self) -> None:
        assert (
            "Une affirmation explicite dans le profil ou le résumé du CV, sans expérience, projet ou formation "
            "qui l'illustre, vaut « mentionné », y compris pour un savoir-être"
        ) in PROMPT

    def test_the_example_is_generic(self) -> None:
        # Exemple propre au prompt (le test des prompts génériques vérifie qu'il ne vient
        # d'aucun cas d'évaluation).
        assert "« je présente régulièrement mes travaux à des publics non techniques »" in PROMPT
        assert "vaut « mentionné » pour « Communication »" in PROMPT


class TestOfficialScoreRule:
    def test_official_score_is_at_least_demonstrated_even_for_a_modest_level(self) -> None:
        assert "il vaut au moins « démontré » pour l'exigence correspondante" in PROMPT
        assert "C'est vrai aussi quand le niveau certifié paraît modeste" in PROMPT
        assert "le niveau reste au moins « démontré », et l'écart avec le niveau demandé s'écrit dans gaps" in PROMPT


class TestSolidCitationRule:
    """v4 : pour « solide », une citation par expérience ou projet distinct (le code exige au
    moins deux items ; sinon le niveau dépendrait du choix de la seule meilleure citation)."""

    def test_solid_asks_for_one_quote_per_distinct_experience_or_project(self) -> None:
        assert (
            "Si le niveau est « solide », cite une preuve dans chaque expérience ou projet distinct qui "
            "l'atteste, pas seulement la meilleure"
        ) in PROMPT
        assert "« solide » demande au moins deux expériences ou projets distincts" in PROMPT

    def test_generic_example(self) -> None:
        assert "deux postes successifs et un projet personnel appelle trois citations" in PROMPT
