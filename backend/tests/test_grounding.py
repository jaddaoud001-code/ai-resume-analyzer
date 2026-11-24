"""Contrôle d'ancrage des reformulations (agent de suggestions, étape 1) et table des
implications. Vrais chunks des CV d'évaluation (fictifs), aucun appel au LLM."""

import re
from pathlib import Path

import pytest

from app.chunking import chunk_cv
from app.grounding import BulletRewrite, Fact, ImpliedSkill, apply_rewrite, bullets, check_grounding
from app.implications import IMPLICATIONS, implication_for, implied_by
from app.matcher import _SYSTEM_PROMPT as MATCH_PROMPT
from app.pdf_parser import extract_text
from app.schemas import Chunk
from tests.conftest import make_settings

CVS = Path(__file__).resolve().parents[1] / "eval" / "cvs"
SECTIONS = ["EXPERIENCE", "PROJET", "STAGE"]


def _chunks(name: str) -> list[Chunk]:
    return chunk_cv(extract_text((CVS / name).read_bytes(), make_settings()).text)


BACKEND = _chunks("backend_cv.pdf")  # 2-3 : Ferlane (découpée), 4 : freelance, 5 : données, 7-8 : compétences
JAVA = _chunks("java_cv.pdf")  # 2 : alternance, 3 : stage, 4 : projet, 7-8 : compétences


def _check(chunks: list[Chunk], rewrite: BulletRewrite) -> list[str]:
    return check_grounding(rewrite, chunks, SECTIONS).codes()


def _rewrite(chunk_id: int, index: int, text: str, facts: list[tuple[int, str]], implied=()) -> BulletRewrite:  # type: ignore[no-untyped-def]
    return BulletRewrite(
        chunk_id=chunk_id,
        bullet_index=index,
        new_text=text,
        facts=[Fact(claim=q, chunk_id=c, quote=q) for c, q in facts],
        implied_skills=[ImpliedSkill(skill=s, chunk_id=c, quote=q) for s, c, q in implied],
    )


# --- Table des implications ----------------------------------------------------------------


class TestImplications:
    def test_every_example_of_the_matching_prompt_is_in_the_table(self) -> None:
        """Le prompt du matching reste identique au caractère près (MATCH_PROMPT_VERSION) ;
        ses exemples d'implication doivent tous figurer dans la table, source unique."""
        prompt = re.sub(r"\\s+", " ", MATCH_PROMPT)
        pairs: list[tuple[str, str]] = []
        # Groupes « (A → X, B → Y) » et « (A, B, C → X) » : les sources sans flèche se
        # rattachent à la flèche suivante du groupe.
        for group in re.findall(r"\(([^()]*→[^()]*)\)", prompt):
            pending: list[str] = []
            for segment in group.split(", "):
                if "→" in segment:
                    source, skill = (part.strip() for part in segment.split("→"))
                    pairs += [(s, skill) for s in [*pending, source]]
                    pending = []
                else:
                    pending.append(segment.strip())
        assert len(pairs) == 7, pairs  # Django, Laravel, Ruby on Rails, Oracle Database, SQL Server, MariaDB, GitLab CI
        for source, skill in pairs:
            assert implication_for(skill, source) is not None, (source, skill)

    @pytest.mark.parametrize(
        ("source", "skill"),
        [
            ("Docker", "Kubernetes"),
            ("React", "Node.js"),
            ("Python", "Django"),
            ("Java", "Spring Boot"),
            ("SQL", "PostgreSQL"),
        ],
    )
    def test_no_neighbouring_or_reverse_implication(self, source: str, skill: str) -> None:
        # Contre-exemples du prompt (« Docker ne prouve pas Kubernetes »…), et sens inverse.
        assert implication_for(skill, source) is None

    def test_whole_words_only(self) -> None:
        assert "git" not in implied_by("Revues de code sur GitHub")  # « GitHub » n'est pas « GitHub Actions »
        assert "git" in implied_by("Pipelines GitHub Actions")
        assert "java" in implied_by("API avec Spring Boot et JPA")

    def test_each_line_names_its_rule(self) -> None:
        assert all(i.rule for i in IMPLICATIONS)


# --- Puces --------------------------------------------------------------------------------


def test_bullets_join_wrapped_lines_and_skip_tag_lines() -> None:
    ferlane, second_part = BACKEND[2], BACKEND[3]

    found = bullets(ferlane.text)

    assert len(found) == 6
    assert found[0].text.startswith("Développement des endpoints FastAPI") and found[0].text.endswith("livraison.")
    assert [b.text for b in bullets(second_part.text)] == [
        "Points hebdomadaires en anglais avec l'équipe d'exploitation de Toronto ; documentation d'architecture "
        "rédigée en anglais."
    ]  # la ligne « FastAPI · PostgreSQL · … » n'est pas une puce


# --- Reformulations honnêtes ---------------------------------------------------------------


class TestHonestRewrites:
    def test_framework_implies_its_language(self) -> None:
        rewrite = _rewrite(
            4,
            0,
            "Développement en Python (Django) d'applications web pour trois clients : gestion de stock, "
            "réservations, facturation.",
            [(4, "Applications web Django pour trois clients")],
            [("Python", 4, "Applications web Django")],
        )

        assert _check(BACKEND, rewrite) == []

    def test_plain_rewording_with_facts_of_the_same_item(self) -> None:
        rewrite = _rewrite(
            2,
            1,
            "Optimisation de requêtes SQL et d'index PostgreSQL pour accélérer les recherches de colis.",
            [(2, "Requêtes SQL et index sur PostgreSQL pour accélérer les recherches de colis.")],
        )

        assert _check(JAVA, rewrite) == []

    def test_both_parts_of_a_split_experience_are_the_same_item(self) -> None:
        rewrite = _rewrite(
            3,
            0,
            "Points hebdomadaires en anglais avec l'équipe d'exploitation de Toronto sur les services Kubernetes.",
            [
                (3, "Points hebdomadaires en anglais avec l'équipe d'exploitation de Toronto"),
                (2, "cluster Kubernetes managé"),
            ],
        )

        assert _check(BACKEND, rewrite) == []


# --- Tentatives de triche ------------------------------------------------------------------


class TestCheating:
    def test_keyword_stuffing(self) -> None:
        rewrite = _rewrite(
            2,
            1,
            "Requêtes SQL, SQL avancé, SQL PostgreSQL, index PostgreSQL, SQL et PostgreSQL pour accélérer les "
            "recherches de colis, SQL.",
            [(2, "Requêtes SQL et index sur PostgreSQL")],
        )

        codes = _check(JAVA, rewrite)

        assert "bourrage" in codes and "trop_long" in codes

    def test_technology_taken_from_the_skill_list(self) -> None:
        # « notions de Python » n'apparaît que dans la liste de compétences du CV Java.
        rewrite = _rewrite(
            2,
            0,
            "Développement d'une API REST de suivi des expéditions avec Spring Boot, JPA et Python.",
            [(2, "Développement d'une API REST de suivi des expéditions"), (7, "notions de Python")],
        )

        codes = _check(JAVA, rewrite)

        assert "fait_hors_experience" in codes and "element_nouveau" in codes

    def test_extended_duration(self) -> None:
        rewrite = _rewrite(
            5,
            0,
            "Pendant 3 ans, scripts Python d'import et de nettoyage des relevés de trafic, exécutés chaque nuit.",
            [(5, "Scripts Python d'import et de nettoyage des relevés de trafic, exécutés chaque nuit.")],
        )

        assert _check(BACKEND, rewrite) == ["element_nouveau"]  # « 3 » n'existe pas dans cette expérience

    def test_fact_moved_from_another_experience(self) -> None:
        # Kafka appartient à l'expérience Ferlane (chunk 2), pas à la mission freelance (chunk 4).
        rewrite = _rewrite(
            4,
            1,
            "Bases PostgreSQL et échanges par Kafka avec les logiciels comptables des clients.",
            [(4, "Bases PostgreSQL et interfaces d'échange"), (2, "puis migration vers Kafka")],
        )

        codes = _check(BACKEND, rewrite)

        assert "fait_autre_item" in codes and "element_nouveau" in codes

    def test_neighbouring_skill_is_not_an_implication(self) -> None:
        rewrite = _rewrite(
            4,
            0,
            "Applications web Django déployées sur Kubernetes pour trois clients.",
            [(4, "Applications web Django pour trois clients")],
            [("Kubernetes", 4, "Applications web Django")],
        )

        codes = _check(BACKEND, rewrite)

        # Déclaration non valide ignorée : le texte est vérifié comme si rien n'était déclaré.
        assert codes == ["element_nouveau"]

    def test_reverse_implication_is_refused(self) -> None:
        rewrite = _rewrite(
            5,
            0,
            "Scripts Python et Django d'import et de nettoyage des relevés de trafic.",
            [(5, "Scripts Python d'import et de nettoyage des relevés de trafic")],
            [("Django", 5, "Scripts Python d'import")],
        )

        assert "element_nouveau" in _check(BACKEND, rewrite)  # Django : élément nouveau

    def test_cloud_cannot_come_from_docker_notions(self) -> None:
        rewrite = _rewrite(
            2,
            0,
            "Développement et déploiement cloud d'une API REST de suivi des expéditions avec Spring Boot et JPA.",
            [(2, "Développement d'une API REST de suivi des expéditions avec Spring Boot et JPA.")],
        )

        assert _check(JAVA, rewrite) == ["element_nouveau"]

    def test_invented_number(self) -> None:
        rewrite = _rewrite(
            2,
            3,
            "Tests pytest sur les endpoints critiques ; couverture portée de 45 % à 95 %.",
            [(2, "Tests pytest sur les endpoints critiques")],
        )

        assert _check(BACKEND, rewrite) == ["element_nouveau"]

    def test_quote_must_exist(self) -> None:
        rewrite = _rewrite(4, 1, "Bases PostgreSQL pour les clients.", [(4, "Bases PostgreSQL répliquées")])

        assert "citation_introuvable" in _check(BACKEND, rewrite)

    def test_every_rewrite_needs_sources(self) -> None:
        assert "aucun_fait" in _check(BACKEND, _rewrite(4, 1, "Bases PostgreSQL pour les clients.", []))

    def test_no_new_bullet(self) -> None:
        rewrite = _rewrite(
            4, 1, "Bases PostgreSQL pour les clients.\n- Une puce de plus", [(4, "Bases PostgreSQL et interfaces")]
        )

        assert "plusieurs_puces" in _check(BACKEND, rewrite)

    def test_skill_list_cannot_be_rewritten(self) -> None:
        rewrite = _rewrite(7, 0, "Python, SQL", [(7, "Langages : Python, SQL")])

        assert _check(BACKEND, rewrite) == ["hors_experience"]

    def test_placeholders_stay_intact(self) -> None:
        chunks = [
            Chunk(
                section="EXPÉRIENCE",
                index=0,
                text=(
                    "Support · Société\nJanv. 2023 – Déc. 2024 · Lyon\n- Réponses aux clients par [EMAIL] et téléphone"
                ),
            )
        ]
        removed = _rewrite(0, 0, "Réponses aux clients par courriel et téléphone", [(0, "Réponses aux clients")])
        added = _rewrite(0, 0, "Réponses de [NAME] aux clients par [EMAIL]", [(0, "Réponses aux clients")])

        assert "placeholder" in check_grounding(removed, chunks, SECTIONS).codes()
        assert "placeholder" in check_grounding(added, chunks, SECTIONS).codes()

    def test_unknown_chunk_or_bullet(self) -> None:
        assert _check(BACKEND, _rewrite(99, 0, "x", [(2, "x")])) == ["chunk_inconnu"]
        assert _check(BACKEND, _rewrite(4, 9, "x", [(4, "x")])) == ["puce_inconnue"]


# --- Application en mémoire ----------------------------------------------------------------


def test_apply_rewrite_replaces_one_bullet_in_a_copy() -> None:
    rewrite = _rewrite(
        2,
        0,
        "Développement d'une API en Python (FastAPI) pour l'application mobile des chauffeurs.",
        [(2, "Développement des endpoints FastAPI consommés par l'application mobile des chauffeurs")],
    )
    before = [c.model_copy() for c in BACKEND]

    patched = apply_rewrite(BACKEND, rewrite)

    assert [c.text for c in BACKEND] == [c.text for c in before]  # originaux intacts
    changed = [i for i, (a, b) in enumerate(zip(BACKEND, patched, strict=True)) if a.text != b.text]
    assert changed == [2]
    new_bullets = [b.text for b in bullets(patched[2].text)]
    assert new_bullets[0] == rewrite.new_text  # la puce repliée sur deux lignes est remplacée par une ligne
    assert new_bullets[1:] == [b.text for b in bullets(BACKEND[2].text)][1:]
    assert patched[2].text.splitlines()[:2] == BACKEND[2].text.splitlines()[:2]  # intitulé et dates intacts


# --- Cas de gain (gain_cv.pdf) : reformulations honnêtes attendues -------------------------

GAIN = _chunks("gain_cv.pdf")  # 2 : Velmaris, 3 : Brumelle, 5-6 : compétences


class TestGainCaseRewrites:
    """Esprit des reformulations attendues (eval/suggestions/gain_cv__logistique) : toutes
    doivent passer le contrôle d'ancrage."""

    def test_python_named_from_the_technology_line(self) -> None:
        rewrite = _rewrite(
            2,
            0,
            "Service de calcul des tarifs de livraison développé en Python (FastAPI), appelé par le site marchand et "
            "l'application des livreurs.",
            [
                (
                    2,
                    "Service de calcul des tarifs de livraison, appelé par le site marchand et "
                    "l'application des livreurs.",
                )
            ],
        )

        assert _check(GAIN, rewrite) == []

    def test_sql_named_through_the_implication_table(self) -> None:
        rewrite = _rewrite(
            2,
            3,
            "Traitement des requêtes SQL lentes de la base PostgreSQL des commandes : "
            "index et réécriture des requêtes.",
            [(2, "Traitement des requêtes lentes de la base des commandes : index et réécriture des requêtes.")],
            [("SQL", 2, "Python · FastAPI · PostgreSQL · Docker")],
        )

        assert _check(GAIN, rewrite) == []

    def test_ci_cd_names_the_described_practice(self) -> None:
        rewrite = _rewrite(
            2,
            1,
            "Pipeline CI/CD sur GitLab : tests et déploiement lancés à chaque fusion sur la branche principale.",
            [(2, "Pipeline de tests et de déploiement lancé à chaque fusion sur la branche principale, sur GitLab.")],
        )

        assert _check(GAIN, rewrite) == []

    def test_observability_names_the_described_practice(self) -> None:
        rewrite = _rewrite(
            2,
            2,
            "Observabilité des services : tableaux de suivi de la latence et alertes envoyées à l'astreinte en cas "
            "d'erreurs.",
            [(2, "Tableaux de suivi de la latence et alertes envoyées à l'astreinte en cas d'erreurs.")],
        )

        assert _check(GAIN, rewrite) == []

    def test_brumelle_python_rewrite_is_grounded_too(self) -> None:
        # Ancrée, mais insuffisante pour 3 ans (30 mois) : c'est la réévaluation qui le dira.
        rewrite = _rewrite(
            3,
            0,
            "Développement en Python (Django) des outils internes de gestion des stocks pour les entrepôts du groupe.",
            [(3, "Outils internes de gestion des stocks pour les entrepôts du groupe.")],
        )

        assert _check(GAIN, rewrite) == []


class TestPracticeTermsNeedTheDescribedPractice:
    """Un terme de pratique (CI/CD, observabilité…) n'est accepté que si la puce d'origine
    décrit cette pratique ; « cloud », « REST » ou « file de messages » jamais."""

    def test_merge_request_reviews_are_not_ci_cd(self) -> None:
        # Tentation annotée (CV Java × logistique, R9).
        rewrite = _rewrite(
            3,
            1,
            "Travail en équipe de quatre avec Git et CI/CD : branches, revues de merge requests sur GitLab.",
            [(3, "Travail en équipe de quatre avec Git : branches, revues de merge requests sur GitLab.")],
        )

        assert _check(JAVA, rewrite) == ["element_nouveau"]

    def test_an_operations_guide_is_not_monitoring(self) -> None:
        # Tentation annotée (CV Java × logistique, R7).
        rewrite = _rewrite(
            2,
            3,
            "Rédaction de la documentation technique de l'API (OpenAPI) et supervision de l'exploitation.",
            [(2, "Rédaction de la documentation technique de l'API (OpenAPI) et d'un guide d'exploitation.")],
        )

        assert _check(JAVA, rewrite) == ["element_nouveau"]

    def test_tests_without_trigger_are_not_ci_cd(self) -> None:
        rewrite = _rewrite(
            3,
            0,
            "Maintenance d'une application Java EE en CI/CD : correction d'anomalies et tests JUnit.",
            [(3, "Maintenance d'une application Java EE : correction d'anomalies et tests JUnit.")],
        )

        assert _check(JAVA, rewrite) == ["element_nouveau"]

    @pytest.mark.parametrize("term", ["déployé dans le cloud", "API REST", "via une file de messages"])
    def test_no_exception_for_cloud_rest_or_message_queue(self, term: str) -> None:
        # Le pipeline décrit un déploiement à chaque fusion : ni hébergement cloud, ni style
        # d'API, ni file de messages pour autant.
        rewrite = _rewrite(
            2,
            1,
            f"Pipeline de tests et de déploiement lancé à chaque fusion sur GitLab, {term}.",
            [(2, "Pipeline de tests et de déploiement lancé à chaque fusion sur la branche principale, sur GitLab.")],
        )

        assert "element_nouveau" in _check(GAIN, rewrite)


# --- Règles du 2026-11-23 : anonymisation contrôlée, nom à sens unique, action gardée --------

DEMO = _chunks("demo_cv.pdf")  # 11 : stage (LLM), 12 : alternance (pseudonymisation), 13 : projet
STAGE_EVAL = "Jeu d'évaluation de 200 questions ; comparaison de trois LLM sur l'exactitude, le coût et la latence."
PSEUDO = "Nettoyage et pseudonymisation des données clients avant analyse."


def _with_text(chunks: list[Chunk], index: int, old: str, new: str) -> list[Chunk]:
    return [c.model_copy(update={"text": c.text.replace(old, new)}) if c.index == index else c for c in chunks]


class TestAnonymisation:
    @pytest.mark.parametrize(
        "text",
        [
            "Nettoyage et anonymisation des données clients avant analyse.",
            "Nettoyage des données clients, anonymisées avant analyse.",
            "Nettoyage des données clients avant analyse, pour les anonymiser.",
        ],
    )
    def test_pseudonymisation_never_becomes_anonymisation(self, text: str) -> None:
        # Terme contrôlé sans exception : le code l'arrête, sans compter sur le juge.
        assert "element_nouveau" in _check(DEMO, _rewrite(12, 1, text, [(12, PSEUDO)]))

    def test_accepted_when_the_item_already_says_it(self) -> None:
        chunks = _with_text(DEMO, 12, "pseudonymisation", "anonymisation")
        original = PSEUDO.replace("pseudonymisation", "anonymisation")
        text = "Nettoyage et anonymisation des données clients avant analyse, avec Pandas."

        assert _check(chunks, _rewrite(12, 1, text, [(12, original)])) == []


class TestOneWayName:
    HONEST = "Jeu d'évaluation de 200 questions ; comparaison de trois LLM (IA générative) sur l'exactitude, le coût et la latence."  # noqa: E501

    def test_llm_allows_generative_ai_without_ia_in_the_title(self) -> None:
        # Le sigle « IA » retiré de l'intitulé du stage : l'acceptation repose sur LLM -> IA générative.
        chunks = _with_text(DEMO, 11, "Stage ingénieure IA", "Stage ingénieure")
        assert "IA" not in next(c for c in chunks if c.index == 11).text.replace("LLM", "")

        assert _check(chunks, _rewrite(11, 1, self.HONEST, [(11, STAGE_EVAL)])) == []

    def test_it_goes_through_the_one_way_rule(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import app.grounding as grounding

        chunks = _with_text(DEMO, 11, "Stage ingénieure IA", "Stage ingénieure")
        monkeypatch.setattr(grounding, "ONE_WAY_SYNONYMS", ())

        assert "element_nouveau" in _check(chunks, _rewrite(11, 1, self.HONEST, [(11, STAGE_EVAL)]))

    def test_not_without_an_llm_in_the_item_nor_in_reverse(self) -> None:
        # Alternance (aucun LLM) : « IA générative » reste un élément nouveau.
        bullet = (
            "Tableaux de bord de suivi des demandes de support en SQL et Python ; rapports hebdomadaires automatisés."
        )
        text = "Tableaux de bord d'IA générative : suivi des demandes de support en SQL et Python ; rapports."
        assert "element_nouveau" in _check(DEMO, _rewrite(12, 0, text, [(12, bullet)]))
        # Sens inverse : un item qui dirait « IA générative » n'autorise pas « LLM ».
        chunks = _with_text(DEMO, 11, "trois LLM", "trois modèles d'IA générative")
        original = STAGE_EVAL.replace("trois LLM", "trois modèles d'IA générative")
        reverse = STAGE_EVAL.replace("trois LLM", "trois LLM d'IA générative")
        assert "LLM" not in next(c for c in chunks if c.index == 11).text
        assert "element_nouveau" in _check(chunks, _rewrite(11, 1, reverse, [(11, original)]))


class TestActionKept:
    MAINTENANCE = "Maintenance d'une application Java EE : correction d'anomalies et tests JUnit."

    def test_the_word_that_opens_the_bullet_must_stay(self) -> None:
        kept = "Maintenance d'une application Java EE : correction d'anomalies et tests automatisés JUnit."
        lost = "Développement d'une application Java EE : correction d'anomalies et tests automatisés JUnit."

        assert _check(JAVA, _rewrite(3, 0, kept, [(3, self.MAINTENANCE)])) == []
        assert _check(JAVA, _rewrite(3, 0, lost, [(3, self.MAINTENANCE)])) == ["action_perdue"]

    def test_reordering_and_plural_are_fine(self) -> None:
        text = "Tests automatisés JUnit et correction d'anomalies lors de la maintenance d'une application Java EE."

        assert _check(JAVA, _rewrite(3, 0, text, [(3, self.MAINTENANCE)])) == []


class TestActionRoot:
    """Règle du premier mot : la racine (6 premières lettres normalisées) suffit."""

    @pytest.mark.parametrize(
        ("word", "text", "kept"),
        [
            ("Développement", "Développé les endpoints", True),
            ("Classement", "Classé 5 000 tickets", True),
            ("Classement", "Tickets classés par priorité", True),
            ("Maintenance", "maintenu une application", True),
            ("Tests", "Testé chaque endpoint", True),
            ("Classement", "Analyse de 5 000 tickets", False),
            ("Participation", "Pilotage de la refonte", False),  # verbe gonflé
            ("Développement", "Conception des endpoints", False),
        ],
    )
    def test_root_comparison(self, word: str, text: str, kept: bool) -> None:
        from app.grounding import keeps_root

        assert keeps_root(text, word) is kept

    def test_on_real_bullets(self) -> None:
        classement = bullets(next(c for c in DEMO if c.index == 13).text)[0].text
        assert classement.startswith("Classement de 5 000 tickets")
        classe = classement.replace("Classement de 5 000 tickets", "Classé 5 000 tickets")
        analyse = classement.replace("Classement de 5 000 tickets", "Analyse de 5 000 tickets")
        assert _check(DEMO, _rewrite(13, 0, classe, [(13, classement)])) == []
        assert _check(DEMO, _rewrite(13, 0, analyse, [(13, classement)])) == ["action_perdue"]

        developpement = bullets(next(c for c in BACKEND if c.index == 2).text)[0].text
        assert developpement.startswith("Développement des endpoints FastAPI")
        developpe = developpement.replace("Développement des endpoints", "Développé les endpoints")
        assert _check(BACKEND, _rewrite(2, 0, developpe, [(2, developpement)])) == []


# --- Run du 2026-11-24 : compétences déclarées = permissions seulement ; étiquette de puce ----

GAIN = _chunks("gain_cv.pdf")  # 2 : Velmaris
GAIN_PIPELINE = "Pipeline de tests et de déploiement lancé à chaque fusion sur la branche principale, sur GitLab."
GAIN_SERVICE = "Service de calcul des tarifs de livraison, appelé par le site marchand et l'application des livreurs."


class TestDeclaredSkillsOnlyGrantPermissions:
    @pytest.mark.parametrize(
        ("text", "skill", "quote"),
        [
            # Les 4 reformulations du run du 2026-11-24, avec la compétence déclarée par le modèle.
            ("Pipeline CI/CD de tests et de déploiement lancé à chaque fusion sur la branche principale, sur GitLab.",
             "CI/CD", "GitLab"),
            ("Intégration continue et déploiement continu : pipeline de tests et de déploiement lancé à chaque fusion "
             "sur la branche principale, sur GitLab.", "CI/CD", "GitLab"),
            ("Pipeline de tests automatisés et de déploiement lancé à chaque fusion sur la branche principale, sur "
             "GitLab.", "intégration continue", GAIN_PIPELINE),
            ("Pipeline de Tests automatisés et de déploiement lancé à chaque fusion sur la branche principale, sur "
             "GitLab.", "Intégration continue", GAIN_PIPELINE),
        ],
    )  # fmt: skip
    def test_honest_rewrites_of_the_run_pass_despite_an_invalid_declaration(
        self, text: str, skill: str, quote: str
    ) -> None:
        rewrite = _rewrite(2, 1, text, [(2, GAIN_PIPELINE)], [(skill, 2, quote)])

        assert _check(GAIN, rewrite) == []

    @pytest.mark.parametrize(
        ("chunks", "chunk_id", "index", "original", "text", "skill", "quote"),
        [
            # Les 5 tentations annotées, avec une déclaration commode : toujours rejetées.
            (BACKEND, 2, 1, None, "Déploiement cloud des services sur un cluster Kubernetes managé : images Docker, "
             "manifestes Helm, montée en charge automatique selon le trafic.", "cloud", "Kubernetes"),
            (BACKEND, 2, 0, None, "Développement des endpoints d'API REST FastAPI consommés par l'application "
             "mobile des chauffeurs : authentification, suivi des tournées, notifications de livraison.",
             "API REST", "FastAPI"),
            (DEMO, 12, 1, PSEUDO, "Nettoyage et anonymisation des données clients avant analyse.", "anonymisation",
             "pseudonymisation"),
            (DEMO, 13, 0, None, "Analyse de données de tickets : 5 000 tickets répartis par catégorie et "
             "priorité, avec scikit-learn puis un LLM ; précision 0,87.", "analyse de données",
             "Classement de 5 000 tickets"),
            (GAIN, 2, 0, GAIN_SERVICE, "API REST de calcul des tarifs de livraison, appelée par le site marchand et "
             "l'application des livreurs.", "API REST", "FastAPI"),
        ],
    )  # fmt: skip
    def test_annotated_temptations_stay_rejected_whatever_is_declared(
        self, chunks: list[Chunk], chunk_id: int, index: int, original: str | None, text: str, skill: str, quote: str
    ) -> None:
        source = original or bullets(next(c for c in chunks if c.index == chunk_id).text)[index].text

        assert _check(chunks, _rewrite(chunk_id, index, text, [(chunk_id, source)], [(skill, chunk_id, quote)]))

    def test_a_valid_declaration_still_grants_its_permission(self) -> None:
        # Velmaris cite PostgreSQL : « SQL » déclaré d'après PostgreSQL est admis.
        bullet = "Traitement des requêtes lentes de la base des commandes : index et réécriture des requêtes."
        text = "Traitement des requêtes SQL lentes de la base des commandes : index et réécriture des requêtes."
        tech = "Python · FastAPI · PostgreSQL · Docker"

        assert _check(GAIN, _rewrite(2, 3, text, [(2, bullet)], [("SQL", 2, tech)])) == []


class TestBulletLabel:
    def test_label_at_the_start_of_a_quote_is_removed(self) -> None:
        from app.grounding import strip_bullet_label

        text = "Service de calcul des tarifs de livraison, appelé par le site marchand et l'application des livreurs."
        quote = f"[puce 0] {GAIN_SERVICE}"

        assert _check(GAIN, _rewrite(2, 0, text, [(2, quote)])) == []
        assert strip_bullet_label("[puce 12] Pipeline") == "Pipeline"
        # Format exact, en début de citation seulement.
        for other in ("[Puce 0] Pipeline", "[puce] Pipeline", "Pipeline [puce 0] de tests"):
            assert strip_bullet_label(other) == other
