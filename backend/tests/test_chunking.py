from app.chunking import HEADER_SECTION, NAME_PLACEHOLDER, chunk_cv
from app.schemas import Chunk
from tests.conftest import GENERIC_CV_NAME, GENERIC_CV_TEXT


def _chunk_with(chunks: list[Chunk], text: str) -> Chunk:
    [chunk] = [c for c in chunks if text in c.text]
    return chunk


def test_sections_are_detected_in_order() -> None:
    chunks = chunk_cv(GENERIC_CV_TEXT)

    sections = list(dict.fromkeys(c.section for c in chunks))
    assert sections == [HEADER_SECTION, "EXPÉRIENCE", "FORMATION", "COMPÉTENCES", "LANGUES"]
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_header_is_its_own_anonymized_chunk() -> None:
    header = chunk_cv(GENERIC_CV_TEXT)[0]

    assert header.section == HEADER_SECTION
    assert header.text.split("\n")[:2] == [NAME_PLACEHOLDER, "Lyon, France"]
    assert "[PHONE] · [EMAIL]\n[URL]" in header.text
    assert "Développeuse backend, 5 ans" in header.text


def test_uppercase_name_is_not_taken_for_a_section() -> None:
    chunks = chunk_cv(GENERIC_CV_TEXT)

    assert all(c.section != GENERIC_CV_NAME for c in chunks)


def test_no_personal_data_in_any_chunk() -> None:
    text = GENERIC_CV_TEXT + "\n\nENGAGEMENT\n\nAteliers animés par Camille Roux (camille@example.org)."

    joined = "\n".join(c.text for c in chunk_cv(text))

    for leaked in ("Camille", "CAMILLE", "Roux", "@", "+33", "github.com"):
        assert leaked not in joined
    assert f"Ateliers animés par {NAME_PLACEHOLDER} ([EMAIL])." in joined


def test_name_line_with_digits_is_not_replaced() -> None:
    chunks = chunk_cv("Lyon 69000\n\nEXPÉRIENCE\n\nDéveloppeuse · Globex\n2021 - 2024")

    assert chunks[0].text == "Lyon 69000"


def test_title_date_bullets_and_tags_stay_together() -> None:
    chunks = chunk_cv(GENERIC_CV_TEXT)

    job = _chunk_with(chunks, "Développeuse backend · Globex")
    assert job.section == "EXPÉRIENCE"
    assert job.text == (
        "Développeuse backend · Globex\n2021 - 2024\n"
        "- Conception d'API REST en Python avec FastAPI.\n"
        "- Déploiement cloud sur AWS avec Kubernetes et Terraform.\n"
        "Python · FastAPI · Kubernetes · AWS"
    )
    assert _chunk_with(chunks, "Initech").text == (
        "Développeuse front-end · Initech\n2019 - 2021\n- Interfaces React et TypeScript, tests automatisés avec Jest."
    )


def test_date_in_its_own_paragraph_is_attached_to_title() -> None:
    chunks = chunk_cv("EXPÉRIENCE\n\nDéveloppeuse · Globex\n\nJanv. 2021 – En cours\n\n- API REST")

    assert [c.text for c in chunks] == ["Développeuse · Globex\nJanv. 2021 – En cours\n- API REST"]


def test_heading_glued_to_its_first_item_is_detected() -> None:
    chunks = chunk_cv("PROFIL\nDéveloppeuse backend\n\nFORMATION\nMaster · Université de Lyon\n2017 - 2019")

    assert [(c.section, c.text) for c in chunks] == [
        ("PROFIL", "Développeuse backend"),
        ("FORMATION", "Master · Université de Lyon\n2017 - 2019"),
    ]


def test_section_without_bullets_has_one_item_per_paragraph() -> None:
    text = "COMPÉTENCES\n\nLangages\nPython, SQL\n\nCloud\nAWS, Kubernetes\n\nOutils\nGit · Docker"

    assert [c.text for c in chunk_cv(text)] == [
        "Langages\nPython, SQL",
        "Cloud\nAWS, Kubernetes",
        "Outils\nGit · Docker",
    ]


def test_long_item_is_split_with_title_repeated() -> None:
    title = "Lead développeuse · Globex\n2018 - 2024"
    bullets = [f"- Réalisation numéro {i} : " + "conception et maintenance d'API " * 3 for i in range(12)]
    text = "EXPÉRIENCE\n\n" + title + "\n\n" + "\n".join(bullets)

    chunks = chunk_cv(text, max_chars=400, overlap=40)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.startswith(title + "\n- ")
        assert len(chunk.text) <= 400
    joined = "\n".join(c.text for c in chunks)
    for i in range(12):
        assert f"Réalisation numéro {i} :" in joined
