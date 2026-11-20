"""Implications directes entre technologies et compétences : source unique dans le code.

Une technologie implique directement une compétence quand on ne peut pas utiliser la première
sans pratiquer la seconde : un framework implique son langage, un SGBD relationnel implique
SQL, un pipeline CI hébergé sur une forge Git implique Git. Rien au-delà : un outil ne prouve
pas une compétence voisine (Docker ne prouve pas Kubernetes, React ne prouve pas Node.js), et
l'implication ne vaut que dans un sens (Python ne prouve pas Django).

Le prompt du matching garde ses propres exemples, au caractère près (MATCH_PROMPT_VERSION) ;
un test vérifie qu'ils figurent tous dans cette table, et que ses contre-exemples n'y sont pas.
L'agent de suggestions n'autorise qu'une compétence ajoutée qui découle d'une de ces lignes."""

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Implication:
    source: str  # technologie citée dans le CV
    implies: str  # compétence qu'elle prouve
    rule: str  # famille de règle, affichée à l'utilisateur


_FRAMEWORK = "un framework prouve son langage"
_RDBMS = "un SGBD relationnel prouve SQL"
_GIT_CI = "un pipeline CI hébergé sur une forge Git prouve Git"

IMPLICATIONS: tuple[Implication, ...] = (
    # Frameworks -> langage.
    *(Implication(s, "Python", _FRAMEWORK) for s in ("Django", "Flask", "FastAPI")),
    *(Implication(s, "PHP", _FRAMEWORK) for s in ("Laravel", "Symfony")),
    Implication("Ruby on Rails", "Ruby", _FRAMEWORK),
    *(Implication(s, "Java", _FRAMEWORK) for s in ("Spring Boot", "Spring", "Java EE", "Jakarta EE")),
    Implication("ASP.NET", "C#", _FRAMEWORK),
    Implication("Angular", "TypeScript", _FRAMEWORK),
    # SGBD relationnels -> SQL.
    *(
        Implication(s, "SQL", _RDBMS)
        for s in ("PostgreSQL", "MySQL", "MariaDB", "Oracle Database", "SQL Server", "SQLite")
    ),
    # CI sur une forge Git -> Git.
    *(Implication(s, "Git", _GIT_CI) for s in ("GitLab CI", "GitHub Actions", "Bitbucket Pipelines")),
)


def normalize(text: str) -> str:
    """Minuscules, sans accents, espaces fusionnés (pour comparer des termes)."""
    plain = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", plain).strip()


def contains_term(text: str, term: str) -> bool:
    """Le terme figure comme mot (ou suite de mots) entier dans le texte, après normalisation :
    « Spring » dans « Spring Boot » oui, « Git » dans « GitHub » non."""
    pattern = rf"(?<![\w+#]){re.escape(normalize(term))}(?![\w+#])"
    return re.search(pattern, normalize(text)) is not None


def implied_by(text: str) -> dict[str, Implication]:
    """Compétences impliquées par les technologies citées dans le texte (compétence normalisée
    -> première implication trouvée)."""
    found: dict[str, Implication] = {}
    for implication in IMPLICATIONS:
        if contains_term(text, implication.source):
            found.setdefault(normalize(implication.implies), implication)
    return found


def implication_for(skill: str, source_text: str) -> Implication | None:
    """L'implication qui autorise `skill` d'après `source_text`, ou None."""
    return implied_by(source_text).get(normalize(skill))
