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
    fingerprints = {2: "cc7ffad3ab5c9914d44e465d97bc93c4a651596722132521e2ec6d60281ea9d0"}

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

    def test_without_description_the_level_is_at_most_mentioned(self) -> None:
        assert "le savoir-être est au plus « mentionné » si le CV en donne un indice explicite" in PROMPT
        assert "sinon « non démontré »" in PROMPT

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


class TestOfficialScoreRule:
    def test_official_score_is_at_least_demonstrated_even_for_a_modest_level(self) -> None:
        assert "il vaut au moins « démontré » pour l'exigence correspondante" in PROMPT
        assert "C'est vrai aussi quand le niveau certifié paraît modeste" in PROMPT
        assert "le niveau reste au moins « démontré », et l'écart avec le niveau demandé s'écrit dans gaps" in PROMPT
