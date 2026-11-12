"""Les exemples des prompts sont génériques : aucun passage ne vient des cas de référence
(offres d'évaluation, CV de démonstration). Un prompt ajusté sur un cas de test donnerait
une évaluation trompeusement bonne sur ce cas."""

import re
import unicodedata
from pathlib import Path

import pymupdf
import pytest

from app.matcher import _SYSTEM_PROMPT as MATCH_PROMPT
from app.requirements import _SYSTEM_PROMPT as EXTRACTION_PROMPT

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
# Des tournures courtes (« à l'aise en », « Bac+5 en informatique ») sont des formulations
# usuelles ; une suite de 5 mots identiques trahit un passage recopié.
MIN_SHARED_WORDS = 5


def _words(text: str) -> list[str]:
    plain = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return re.findall(r"[a-z0-9]+", plain)


def _sequences(text: str, n: int = MIN_SHARED_WORDS) -> set[str]:
    words = _words(text)
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _reference_cases() -> list[tuple[str, str]]:
    cases = [(path.name, path.read_text(encoding="utf-8")) for path in sorted((EVAL_DIR / "offers").glob("*.txt"))]
    for pdf in sorted((EVAL_DIR / "cvs").glob("*.pdf")):
        with pymupdf.open(pdf) as document:
            cases.append((pdf.name, "\n".join(page.get_text() for page in document)))
    return cases


def test_reference_cases_exist() -> None:
    names = [name for name, _ in _reference_cases()]
    assert {"hager.txt", "sii.txt"} <= set(names)


@pytest.mark.parametrize("prompt_name, prompt", [("extraction", EXTRACTION_PROMPT), ("évaluation", MATCH_PROMPT)])
def test_prompt_examples_do_not_come_from_reference_cases(prompt_name: str, prompt: str) -> None:
    prompt_sequences = _sequences(prompt)
    for case, text in _reference_cases():
        shared = prompt_sequences & _sequences(text)
        assert not shared, f"prompt d'{prompt_name} : passage repris de {case} : {sorted(shared)}"
