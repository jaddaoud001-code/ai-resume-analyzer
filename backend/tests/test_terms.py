"""Termes de l'offre et preuves implicites (app/terms.py), libellés génériques."""

from app.schemas import Evidence, Requirement, RequirementMatch
from app.terms import is_implicit, mentions_term, offer_terms, to_describe


def _requirement(label: str, category: str = "compétence technique") -> Requirement:
    return Requirement(
        id="R1",
        label=label,
        category=category,  # type: ignore[arg-type]
        importance="obligatoire",
        search_queries=["a", "b"],
    )


def _match(label: str, level: str, *quotes: str, category: str = "compétence technique") -> RequirementMatch:
    return RequirementMatch(
        requirement=_requirement(label, category),
        status="ok",
        level=level,  # type: ignore[arg-type]
        evidence=[Evidence(chunk_id=1, quote=q) for q in quotes],
    )


def test_offer_terms_split_parts_parentheses_intro_and_direct_synonyms() -> None:
    assert offer_terms(_requirement("Outils de visualisation")) == ["Outils de visualisation", "visualisation"]
    assert offer_terms(_requirement("Go et bases de données NoSQL")) == [
        "Go et bases de données NoSQL",
        "Go",
        "bases de données NoSQL",
    ]
    terms = offer_terms(_requirement("Supervision (logs, traces)"))
    assert terms[:3] == ["Supervision", "logs", "traces"] and {"observabilité", "monitoring"} <= set(terms)


def test_mentions_term_ignores_case_accents_and_simple_plurals() -> None:
    assert mentions_term("Base de donnée relationnelle", "bases de données relationnelles")
    assert mentions_term("Pipeline d'INTEGRATION continue", "intégration continue")
    assert not mentions_term("GitHub Actions", "Git")


def test_implicit_only_for_proven_technical_skills_without_the_term() -> None:
    assert is_implicit(_match("Go", "démontré", "Service écrit avec le framework Gin"))
    assert not is_implicit(_match("Go", "solide", "Service en Go"))  # terme nommé
    assert not is_implicit(_match("Go", "mentionné", "Gin"))  # lacune : carte à décrire
    assert not is_implicit(_match("Anglais", "démontré", "TOEIC 900", category="langue"))
    assert not is_implicit(_match("Rigueur", "démontré", "revues de code", category="savoir-être"))
    assert to_describe(_match("Go", "mentionné", "Gin")) and not to_describe(_match("Go", "démontré", "Go"))
