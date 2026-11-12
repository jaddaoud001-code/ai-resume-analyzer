import pytest

from app.config import Settings
from app.pdf_parser import (
    EncryptedPDFError,
    InvalidPDFError,
    NoExtractableTextError,
    PDFTooLargeError,
    TooManyPagesError,
    clean_text,
    extract_text,
)
from tests.conftest import (
    BULLET_ITEMS,
    BULLET_NEXT_SECTION,
    BULLET_NUMBER_ITEM,
    NARROW_SPACE_LINES,
    RIGHT_COLUMN_SKILLS,
    SPACED_HEADING,
    TIGHT_WORDS,
    TWO_COLUMN_HEADER,
    TWO_COLUMN_LEFT,
    TWO_COLUMN_RIGHT,
)


def _has_empty_bullet(text: str) -> bool:
    return any(line.strip() == "-" for line in text.split("\n"))


class TestExtractText:
    def test_simple_cv(self, simple_cv_pdf: bytes, settings: Settings) -> None:
        parsed = extract_text(simple_cv_pdf, settings)

        assert parsed.page_count == 1
        assert parsed.char_count == len(parsed.text)
        assert parsed.text.startswith("Jean Dupont")
        assert "Développeur full-stack, 6 ans d'expérience" in parsed.text
        assert "Lead développeur front-end (2020-2024)" in parsed.text
        assert "- Python, FastAPI, Docker\n- React, TypeScript" in parsed.text

    def test_two_columns_are_read_one_after_the_other(self, two_column_cv_pdf: bytes, settings: Settings) -> None:
        text = extract_text(two_column_cv_pdf, settings).text

        positions = [text.index(line) for line in [TWO_COLUMN_HEADER, *TWO_COLUMN_LEFT, *TWO_COLUMN_RIGHT]]
        assert positions == sorted(positions)

    def test_bullet_column_in_separate_block_is_attached_to_items(
        self, bullet_column_pdf: bytes, settings: Settings
    ) -> None:
        text = extract_text(bullet_column_pdf, settings).text

        lines = text.split("\n")
        for item in BULLET_ITEMS:
            assert f"- {item}" in lines
        assert BULLET_NEXT_SECTION in lines
        assert not _has_empty_bullet(text)

    def test_bullet_blocks_attach_within_their_column(self, bullet_blocks_pdf: bytes, settings: Settings) -> None:
        text = extract_text(bullet_blocks_pdf, settings).text

        lines = text.split("\n")
        for skill in RIGHT_COLUMN_SKILLS:
            assert f"- {skill}" in lines
        assert not _has_empty_bullet(text)
        positions = [text.index(line) for line in [TWO_COLUMN_HEADER, *TWO_COLUMN_LEFT, "- Python"]]
        assert positions == sorted(positions)

    def test_missing_spaces_between_words_are_restored(self, tight_spacing_pdf: bytes, settings: Settings) -> None:
        lenient = settings.model_copy(update={"min_text_chars": 0})

        lines = extract_text(tight_spacing_pdf, lenient).text.split("\n")

        for left, right in TIGHT_WORDS:
            # Une fois dans un seul span (même police), une fois sur deux spans (gras).
            assert lines.count(f"{left} {right}") == 2

    def test_real_narrow_spaces_are_always_kept(self, narrow_space_pdf: bytes, settings: Settings) -> None:
        lenient = settings.model_copy(update={"min_text_chars": 0})

        lines = extract_text(narrow_space_pdf, lenient).text.split("\n")

        assert lines == [expected for _, expected in NARROW_SPACE_LINES]

    def test_bullet_before_number_keeps_its_space(self, bullet_number_pdf: bytes, settings: Settings) -> None:
        lenient = settings.model_copy(update={"min_text_chars": 0})

        text = extract_text(bullet_number_pdf, lenient).text

        assert text.count(f"- {BULLET_NUMBER_ITEM}") == 2
        assert "-2025" not in text

    def test_letter_spaced_heading_is_not_split(self, tight_spacing_pdf: bytes, settings: Settings) -> None:
        lenient = settings.model_copy(update={"min_text_chars": 0})

        assert SPACED_HEADING in extract_text(tight_spacing_pdf, lenient).text.split("\n")

    def test_image_only_pdf(self, image_only_pdf: bytes, settings: Settings) -> None:
        with pytest.raises(NoExtractableTextError):
            extract_text(image_only_pdf, settings)

    def test_text_below_threshold(self, simple_cv_pdf: bytes, settings: Settings) -> None:
        strict = settings.model_copy(update={"min_text_chars": 10_000})

        with pytest.raises(NoExtractableTextError):
            extract_text(simple_cv_pdf, strict)

    def test_encrypted_pdf(self, encrypted_pdf: bytes, settings: Settings) -> None:
        with pytest.raises(EncryptedPDFError):
            extract_text(encrypted_pdf, settings)

    @pytest.mark.parametrize("data", [b"", b"Ceci n'est pas un PDF.", b"%PDF-1.7\n%garbage"])
    def test_invalid_bytes(self, data: bytes, settings: Settings) -> None:
        with pytest.raises(InvalidPDFError):
            extract_text(data, settings)

    def test_truncated_pdf(self, simple_cv_pdf: bytes, settings: Settings) -> None:
        with pytest.raises(InvalidPDFError):
            extract_text(simple_cv_pdf[: len(simple_cv_pdf) // 3], settings)

    def test_too_many_pages(self, too_many_pages_pdf: bytes, settings: Settings) -> None:
        with pytest.raises(TooManyPagesError, match="11 pages"):
            extract_text(too_many_pages_pdf, settings)

    def test_file_too_large(self, simple_cv_pdf: bytes, settings: Settings) -> None:
        tiny = settings.model_copy(update={"max_pdf_size_bytes": len(simple_cv_pdf) - 1})

        with pytest.raises(PDFTooLargeError):
            extract_text(simple_cv_pdf, tiny)


class TestCleanText:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("full-\nstack", "full-stack"),
            ("front-\nend", "front-end"),
            ("Jean-\nPierre", "Jean-Pierre"),
            ("2019-\n2023", "2019-2023"),
            ("full-  \n  stack", "full-stack"),
        ],
    )
    def test_line_break_after_hyphen_is_joined_keeping_hyphen(self, raw: str, expected: str) -> None:
        assert clean_text(raw) == expected

    def test_hyphen_not_attached_to_word_is_kept(self) -> None:
        assert clean_text("Python -\nDocker") == "Python -\nDocker"

    def test_control_and_invisible_characters_are_removed(self) -> None:
        assert clean_text("Py\x00th​on­\x07 ﻿dev") == "Python dev"

    def test_spaces_are_normalized(self) -> None:
        assert clean_text("  Lead\t\tdév  senior  \nParis ") == "Lead dév senior\nParis"

    def test_blank_lines_are_collapsed(self) -> None:
        assert clean_text("A\r\n\r\n\r\n\n  \nB\fC") == "A\n\nB\nC"

    def test_line_breaks_are_preserved(self) -> None:
        text = "EXPÉRIENCE\nAcme Corp\n\nFORMATION\nMaster"
        assert clean_text(text) == text

    @pytest.mark.parametrize("bullet", ["•", "●", "▪", "◦", "·", "–", "*", ""])
    def test_bullets_are_normalized(self, bullet: str) -> None:
        assert clean_text(f"{bullet} Python\n{bullet}Docker") == "- Python\n- Docker"

    def test_bullet_alone_on_its_line_is_merged_with_item(self) -> None:
        assert clean_text("•\nPython\n•\nDocker") == "- Python\n- Docker"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Docker\n\n•\n•\n•\n\nStage", "Docker\n\nStage"),
            ("•\n\nPython", "Python"),
            ("Python\n•", "Python"),
            ("•\n•\nPython", "- Python"),
        ],
    )
    def test_orphan_bullets_are_removed(self, raw: str, expected: str) -> None:
        assert clean_text(raw) == expected
