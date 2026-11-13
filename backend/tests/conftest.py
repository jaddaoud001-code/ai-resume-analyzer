"""Fixtures partagées. Les PDF de test sont générés en mémoire avec PyMuPDF."""

import re
import socket
import unicodedata
import zlib
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import pymupdf
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import Field

from app import embeddings as embeddings_module
from app.config import Settings
from app.main import create_app


@pytest.fixture(autouse=True)
def no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Aucun test ne doit contacter le réseau ni télécharger de modèle, sauf les tests
    d'intégration (pytest -m integration), désactivés par défaut."""
    if request.node.get_closest_marker("integration"):
        return

    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"Appel réseau interdit dans les tests : {args[:1]}")

    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)


@pytest.fixture(autouse=True)
def no_real_embeddings_model(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Aucun test ne charge ni ne télécharge le vrai modèle fastembed : sinon le résultat
    dépendrait du cache de la machine (présent en local, vide en CI). Toute tentative fait
    échouer le test, même depuis le thread de préchargement de l'application (dont l'erreur
    serait sinon seulement journalisée). Un test peut remplacer TextEmbedding par un faux."""
    if request.node.get_closest_marker("integration"):
        yield
        return
    attempts: list[str] = []

    class ForbiddenTextEmbedding:
        def __init__(self, model_name: str, *args: object, **kwargs: object) -> None:
            attempts.append(model_name)
            raise AssertionError(f"Chargement du vrai modèle d'embeddings interdit dans les tests : {model_name}")

    monkeypatch.setattr(embeddings_module, "TextEmbedding", ForbiddenTextEmbedding)
    embeddings_module._cached_model.cache_clear()
    yield
    embeddings_module._cached_model.cache_clear()
    if attempts:
        pytest.fail(
            f"Le test a tenté de charger le vrai modèle d'embeddings ({attempts[0]}) : "
            "utilisez create_test_app (embeddings factices) ou un faux modèle."
        )


class FakeStructuredChatModel(BaseChatModel):
    """Faux modèle de chat : with_structured_output renvoie, appel après appel, les
    réponses prévues (un objet, ou une exception à lever) et enregistre les messages reçus.
    Avec `responder`, la réponse est calculée à partir des messages (appels concurrents)."""

    responses: list[Any] = Field(default_factory=list)
    responder: Callable[[list[BaseMessage]], Any] | None = None
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    structured_kwargs: dict[str, Any] = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "fake-structured"

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError("Ce faux modèle ne sert qu'à with_structured_output.")

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:  # type: ignore[override]
        self.structured_kwargs = {"schema": schema, **kwargs}

        def respond(messages: Sequence[BaseMessage]) -> Any:
            self.calls.append(list(messages))
            response = self.responder(list(messages)) if self.responder else self.responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            return response

        return RunnableLambda(respond)


class KeywordEmbeddings(Embeddings):
    """Embeddings factices et déterministes : chaque mot (minuscules, sans accents) active
    une dimension fixe. Deux textes qui partagent des mots sont donc proches."""

    DIM = 512

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        plain = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
        vector = [0.0] * self.DIM
        for word in re.findall(r"[a-z0-9]+", plain):
            vector[zlib.crc32(word.encode()) % self.DIM] += 1.0
        return vector


# CV fictif : nom, coordonnées et employeurs inventés.
GENERIC_CV_NAME = "CAMILLE ROUX"
GENERIC_CV_PARAGRAPHS = [
    GENERIC_CV_NAME,
    "Lyon, France\n+33 6 12 34 56 78 · camille.roux@example.com\ngithub.com/camille-roux",
    "Développeuse backend, 5 ans d'expérience en Python et en cloud.",
    "EXPÉRIENCE",
    "Développeuse backend · Globex\n2021 - 2024",
    "- Conception d'API REST en Python avec FastAPI.\n- Déploiement cloud sur AWS avec Kubernetes et Terraform.",
    "Python · FastAPI · Kubernetes · AWS",
    "Développeuse front-end · Initech\n2019 - 2021",
    "- Interfaces React et TypeScript, tests automatisés avec Jest.",
    "FORMATION",
    "Master informatique · Université de Lyon\n2017 - 2019",
    "COMPÉTENCES",
    "Langages\nPython, TypeScript, SQL",
    "LANGUES",
    "Français natif · Anglais C1",
]
GENERIC_CV_TEXT = "\n\n".join(GENERIC_CV_PARAGRAPHS)

SIMPLE_CV_LINES = [
    "Jean Dupont",
    "Développeur full-",
    "stack, 6 ans d'expérience",
    "",
    "COMPÉTENCES",
    "· Python, FastAPI, Docker",
    "· React, TypeScript",
    "",
    "EXPÉRIENCE",
    "Acme Corp - Lead développeur front-",
    "end (2020-2024)",
    "Conception d'une plateforme de traitement de données pour 2 millions d'utilisateurs.",
]

TWO_COLUMN_HEADER = "Marie Martin - Data Scientist senior - Paris - marie.martin@example.com"
TWO_COLUMN_LEFT = ["EXPÉRIENCE", "Data Scientist chez Globex", "Modèles de prévision de la demande"]
TWO_COLUMN_RIGHT = ["COMPÉTENCES", "Python, SQL, PyTorch", "LANGUES", "Anglais courant"]


BULLET_ITEMS = [
    "Intégration continue avec GitHub Actions sur tous les projets du pôle web",
    "Tests unitaires et de bout en bout sur les parcours critiques de l'application",
    "Revue de code systématique et documentation technique des modules partagés",
]
BULLET_NEXT_SECTION = "Stage développement front-end chez Initech, refonte du site vitrine."
RIGHT_COLUMN_SKILLS = ["Python", "TypeScript", "Docker"]

# Écarts entre mots sans caractère espace, à 11 pt : 1,5 pt = 0,136 em, au-dessus de
# _WORD_GAP_EM mais en dessous du seuil de MuPDF, qui colle alors les mots.
TIGHT_GAP = 1.5
TIGHT_WORDS = [("Apache", "Kafka"), ("en", "équipe"), ("et", "CI")]
# Interlettrage d'un titre espacé tel que mesuré sur un vrai CV (~0,093 em à 11 pt).
LETTER_SPACING = 0.093 * 11
SPACED_HEADING = "EXPÉRIENCE"


# Vrais caractères espace rendus étroits par un crénage TJ négatif (en millièmes d'em,
# l'espace Helvetica fait 278) : l'écart visible restant est bien sous _WORD_GAP_EM.
NARROW_SPACE_LINES = [
    (b"[(avant la) ( ) 200 (mise en ligne)] TJ", "avant la mise en ligne"),
    (b"[(Redis) ( ) 260 (/) ( ) 260 (Celery)] TJ", "Redis / Celery"),
    (b"[(Go) ( ) 278 (\\267) ( ) 278 (gRPC)] TJ", "Go · gRPC"),
]
BULLET_NUMBER_ITEM = "2023 : migration de 4 services Python vers Kubernetes"


def _insert_runs(page: pymupdf.Page, origin: tuple[float, float], runs: list[tuple[str, str]], gap: float) -> None:
    """Écrit chaque (texte, police) dans un objet texte séparé, espacé de `gap` points."""
    x, y = origin
    for text, font in runs:
        page.insert_text((x, y), text, fontname=font, fontsize=11)
        x += pymupdf.get_text_length(text, fontname=font, fontsize=11) + gap


def _build_pdf(fill: Callable[[pymupdf.Document], None], **save_options: object) -> bytes:
    doc = pymupdf.open()
    try:
        fill(doc)
        return doc.tobytes(**save_options)
    finally:
        doc.close()


def make_settings(**overrides: object) -> Settings:
    """Settings validés (contrairement à model_copy : les clés deviennent des SecretStr).
    _env_file=None : les tests ne dépendent pas du .env local. Limiteur sans espacement
    (llm_rps=0) sauf demande contraire, pour des tests rapides."""
    return Settings(_env_file=None, **{"llm_rps": 0, **overrides})  # type: ignore[call-arg, arg-type]


@pytest.fixture
def settings() -> Settings:
    return make_settings()


def create_test_app(settings: Settings, **kwargs: Any) -> FastAPI:
    """create_app avec des embeddings factices par défaut : le préchargement du modèle local
    (lifespan) ne touche jamais au vrai modèle. Les tests du préchargement passent leur faux
    modèle par `embeddings_factory`."""
    kwargs.setdefault("embeddings_factory", lambda _: KeywordEmbeddings())
    return create_app(settings, **kwargs)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_test_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def sectioned_cv_pdf() -> bytes:
    """GENERIC_CV_PARAGRAPHS, un objet texte par paragraphe (donc un bloc chacun)."""

    def fill(doc: pymupdf.Document) -> None:
        page = doc.new_page()
        y = 60.0
        for paragraph in GENERIC_CV_PARAGRAPHS:
            page.insert_text((50, y), paragraph, fontsize=10)
            y += 13 * (paragraph.count("\n") + 1) + 12

    return _build_pdf(fill)


@pytest.fixture
def simple_cv_pdf() -> bytes:
    def fill(doc: pymupdf.Document) -> None:
        doc.new_page().insert_text((50, 72), "\n".join(SIMPLE_CV_LINES), fontsize=11)

    return _build_pdf(fill)


@pytest.fixture
def two_column_cv_pdf() -> bytes:
    def fill(doc: pymupdf.Document) -> None:
        page = doc.new_page()
        # Ordre d'écriture volontairement inverse de l'ordre de lecture attendu.
        page.insert_text((320, 140), "\n".join(TWO_COLUMN_RIGHT), fontsize=11)
        page.insert_text((50, 140), "\n".join(TWO_COLUMN_LEFT), fontsize=11)
        page.insert_text((50, 72), TWO_COLUMN_HEADER, fontsize=12)

    return _build_pdf(fill)


@pytest.fixture
def bullet_column_pdf() -> bytes:
    """Colonne de puces dessinée d'un bloc, séparée d'items plus larges que la demi-page."""

    def fill(doc: pymupdf.Document) -> None:
        page = doc.new_page()
        page.insert_text((50, 72), "Mise en place d'environnements de développement et de production.")
        page.insert_text((72, 100), "\n".join(BULLET_ITEMS), fontsize=11)
        # La police de base rend "•" par "·" ; reste une puce pour le parseur.
        page.insert_text((56, 100), "\n".join("•" * len(BULLET_ITEMS)), fontsize=11)
        page.insert_text((50, 160), BULLET_NEXT_SECTION, fontsize=11)

    return _build_pdf(fill)


@pytest.fixture
def bullet_blocks_pdf() -> bytes:
    """Une puce par bloc, collée au texte (comme un export Word), dont une dans la
    colonne de droite d'un CV sur deux colonnes, et une puce sans texte associé."""

    def fill(doc: pymupdf.Document) -> None:
        page = doc.new_page()
        page.insert_text((50, 72), TWO_COLUMN_HEADER, fontsize=12)
        page.insert_text((50, 100), "\n".join(TWO_COLUMN_LEFT), fontsize=11)
        for i, skill in enumerate(RIGHT_COLUMN_SKILLS):
            y = 100 + i * 20
            page.insert_text((340, y), skill, fontsize=11)
            page.insert_text((328, y), "•", fontsize=8)
        page.insert_text((328, 200), "•", fontsize=8)  # orpheline : rien à sa droite

    return _build_pdf(fill)


@pytest.fixture
def tight_spacing_pdf() -> bytes:
    """Mots séparés par un écart sans caractère espace : même police (MuPDF les fusionne
    en un seul span) puis changement de graisse (deux spans), et un titre espacé."""

    def fill(doc: pymupdf.Document) -> None:
        page = doc.new_page()
        for i, (left, right) in enumerate(TIGHT_WORDS):
            _insert_runs(page, (50, 72 + i * 16), [(left, "helv"), (right, "helv")], TIGHT_GAP)
            _insert_runs(page, (300, 72 + i * 16), [(left, "helv"), (right, "hebo")], TIGHT_GAP)
        _insert_runs(page, (50, 140), [(letter, "helv") for letter in SPACED_HEADING], LETTER_SPACING)

    return _build_pdf(fill)


@pytest.fixture
def narrow_space_pdf() -> bytes:
    def fill(doc: pymupdf.Document) -> None:
        page = doc.new_page()
        page.insert_text((0, -100), "x", fontname="helv")  # déclare la ressource /helv
        operations = b" ".join(
            b"BT /helv 11 Tf 50 %d Td %s ET" % (72 + i * 16, tj) for i, (tj, _) in enumerate(NARROW_SPACE_LINES)
        )
        doc.update_stream(page.get_contents()[0], operations)

    return _build_pdf(fill)


@pytest.fixture
def bullet_number_pdf() -> bytes:
    """Puce dans son propre bloc devant un item qui commence par un nombre, et puce
    collée au nombre sur la même ligne."""

    def fill(doc: pymupdf.Document) -> None:
        page = doc.new_page()
        page.insert_text((62, 72), BULLET_NUMBER_ITEM, fontsize=11)
        page.insert_text((50, 72), "•", fontsize=8)
        page.insert_text((50, 100), f"•{BULLET_NUMBER_ITEM}", fontsize=11)

    return _build_pdf(fill)


@pytest.fixture
def image_only_pdf() -> bytes:
    def fill(doc: pymupdf.Document) -> None:
        page = doc.new_page()
        page.draw_rect(pymupdf.Rect(50, 50, 300, 300), color=(0, 0, 0), fill=(0.8, 0.8, 0.8))

    return _build_pdf(fill)


@pytest.fixture
def encrypted_pdf() -> bytes:
    def fill(doc: pymupdf.Document) -> None:
        doc.new_page().insert_text((50, 72), "\n".join(SIMPLE_CV_LINES), fontsize=11)

    return _build_pdf(
        fill,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw="secret",
        owner_pw="owner-secret",
    )


@pytest.fixture
def too_many_pages_pdf(settings: Settings) -> bytes:
    def fill(doc: pymupdf.Document) -> None:
        for number in range(settings.max_pdf_pages + 1):
            doc.new_page().insert_text((50, 72), f"Page {number + 1}", fontsize=11)

    return _build_pdf(fill)


@pytest.fixture
def invalid_bytes() -> bytes:
    return b"Ceci n'est pas un PDF, juste du texte."
