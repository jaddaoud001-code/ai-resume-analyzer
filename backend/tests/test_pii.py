import pytest

from app.pii import redact_pii


@pytest.mark.parametrize(
    "phone",
    [
        "06 12 34 56 78",
        "06.12.34.56.78",
        "06-12-34-56-78",
        "0612345678",
        "01 23 45 67 89",
        "+33 6 12 34 56 78",
        "+33612345678",
        "+33 (0)6 12 34 56 78",
        "0033 6 12 34 56 78",
        "+1 (415) 555-2671",
        "+44 20 7946 0958",
        "+961 3 123 456",
        "+49 30 12345678",
    ],
)
def test_phones_are_redacted(phone: str) -> None:
    assert redact_pii(f"Tél. : {phone}.") == "Tél. : [PHONE]."


@pytest.mark.parametrize(
    "email",
    [
        "camille.roux@example.com",
        "c.roux+cv@mail.example.fr",
        "CAMILLE_ROUX@Example.ORG",
        "dev-42@sub.domain.co.uk",
    ],
)
def test_emails_are_redacted(email: str) -> None:
    assert redact_pii(f"Contact : {email}, merci") == "Contact : [EMAIL], merci"


@pytest.mark.parametrize(
    "url",
    [
        "https://www.linkedin.com/in/camille-roux",
        "http://linkedin.com/in/camille-roux/",
        "linkedin.com/in/camille-roux",
        "LinkedIn.com/in/camille-roux",
        "www.linkedin.com/in/camille-roux",
        "github.com/camille-roux",
        "https://github.com/camille-roux/projet?tab=readme",
        "camille-roux.dev",
        "camilleroux.github.io",
        "HTTPS://Portfolio.Example.FR/projets",
    ],
)
def test_urls_are_redacted(url: str) -> None:
    assert redact_pii(f"Profil : {url}.") == "Profil : [URL]."


def test_email_is_not_split_into_url() -> None:
    assert redact_pii("camille@github.com") == "[EMAIL]"


@pytest.mark.parametrize(
    "text",
    [
        "Mars 2019 – Juin 2022",
        "2025 – 2027",
        "150+ tests automatisés",
        "TOEIC 905/990",
        "8 migrations de schéma, 3 profils",
        "Python 3.12, Java 17",
        "Node.js, ASP.NET, Socket.IO, Vue.js",
        "README.md et main.py",
        "Bac+5, 06/2024",
        "Score de 0.85 sur 1 000 000 requêtes",
    ],
)
def test_non_personal_data_is_kept(text: str) -> None:
    assert redact_pii(text) == text


def test_all_kinds_in_one_header() -> None:
    header = "Lyon · +33 7 80 00 00 00 · camille.roux@example.com · linkedin.com/in/camille"

    assert redact_pii(header) == "Lyon · [PHONE] · [EMAIL] · [URL]"
