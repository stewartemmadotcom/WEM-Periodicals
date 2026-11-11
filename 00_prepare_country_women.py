#!/usr/bin/env python3
"""
Step 0: Prepare the Country Women corpus
========================================

Lesbian Connection and Maize come into corpus/ already cleaned (by
00_prepare_corpus.py in lc-word-embeddings and maize-word-embeddings). Country
Women still needs cleaning. This script turns the Gemini OCR copied into
raw/country-women/ into the same layout:

    raw/country-women/CountryWomen-1974-01-no9.txt
        ->  corpus/country-women/CountryWomen-1974-01-no9.txt

For each issue it:
  - drops the page markers, [MISSING TRANSCRIPTION], and the [Source: ...] line
    on pages filled with Tesseract
  - drops the transcriber's notes that describe the page rather than quote it:
    [illegible], [stamp], [image ...], [drawing ...], music notes, arrows, ...
  - keeps the words inside notes that wrap printed or handwritten words, without
    the label: "[handwritten: politely modeled by Petunia]" -> "politely modeled
    by Petunia" (same for sideways text, vertical text, label, underline, ...)
  - drops lines that are only a page number, like "-2-" or "14"
  - rejoins words split across a line break ("con-\\nference" -> "conference")
  - unwraps each paragraph onto one line, with a blank line between paragraphs
    (01_train_model.py splits on blank lines)

Spelling, case, and punctuation are left alone: 01_train_model.py lowercases
and splits the words itself. Rerun this script whenever raw/country-women
changes; it rebuilds corpus/country-women.
"""

import re
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# --- Settings ---------------------------------------------------------------

RAW_FOLDER = SCRIPT_DIR / "raw" / "country-women"
OUTPUT_FOLDER = SCRIPT_DIR / "corpus" / "country-women"

# Notes whose words are on the page: keep the words, drop the label
KEEP_WORDS = ["handwritten", "sideways text", "text sideways", "sideways", "vertical text",
              "vertical", "label", "underline", "italicized", "speech bubble"]

# Notes that only describe the page: drop them (matched on the start of the label)
DROP = ["illegible", "stamp", "image", "illustration", "drawing", "photo", "quarter note",
        "eighth note", "half note", "whole note", "treble clef", "bass clef", "sharp",
        "flat key", "arrow", "uparrow", "downarrow", "box outline", "asterisk",
        "female symbol", "number", "artist signature", "logo",
        "source"]   # [Source: Tesseract OCR, not Gemini ...] on pages fill_with_tesseract.py filled

# Warn about issues where more than this share of pages has no transcription.
MISSING_WARNING = 0.10

# ----------------------------------------------------------------------------

PAGE_MARKER = re.compile(r"^===== Page \d+ =====$", re.MULTILINE)
MISSING = "[MISSING TRANSCRIPTION]"
# [label] or [label: words]; a closing tag like [/underline] has a leading slash
NOTE = re.compile(r"\[(/?)([A-Za-z][A-Za-z .'-]*?)\s*(?::\s*([^\]]*))?\]")
PAGE_NUMBER_LINE = re.compile(r"^\s*-?\s*\d{1,3}\s*-?\s*$")
BROKEN_WORD = re.compile(r"(?<=[A-Za-z])(?<!-)-\n\s*(?=[a-z])")
WORD = re.compile(r"[a-z]+(?:['-][a-z]+)*")


def replace_note(match):
    closing, label, words = match.group(1), match.group(2).lower(), match.group(3)
    if label in KEEP_WORDS:
        return "" if closing else (words or "")
    if any(label.startswith(d) for d in DROP):
        return ""
    return match.group(0)   # something else in brackets: probably printed, keep it


def unwrap(page):
    """Join the lines of each paragraph into one line.
    A new paragraph starts after a blank line or at an indented line."""
    paragraphs, current = [], []
    for line in page.split("\n"):
        if PAGE_NUMBER_LINE.match(line):
            continue
        starts_paragraph = not line.strip() or line.startswith((" ", "\t"))
        if starts_paragraph and current:
            paragraphs.append(" ".join(current))
            current = []
        if line.strip():
            current.append(line.strip())
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def clean_issue(text):
    """Return (cleaned text, number of pages, number of missing pages)."""
    pages = [p.strip() for p in PAGE_MARKER.split(text) if p.strip()]
    missing = 0
    paragraphs = []
    for page in pages:
        if page == MISSING:
            missing += 1
            continue
        page = NOTE.sub(replace_note, page)
        page = BROKEN_WORD.sub("", page)
        paragraphs.extend(unwrap(page))
    paragraphs = [re.sub(r"[ \t]+", " ", p).strip() for p in paragraphs]
    return "\n\n".join(p for p in paragraphs if p) + "\n", len(pages), missing


def main():
    files = sorted(RAW_FOLDER.glob("*.txt"))
    if not files:
        print(f"No .txt files in {RAW_FOLDER}")
        return
    if OUTPUT_FOLDER.exists():
        shutil.rmtree(OUTPUT_FOLDER)
    OUTPUT_FOLDER.mkdir(parents=True)

    total_words, warnings = 0, []
    for path in files:
        text, pages, missing = clean_issue(path.read_text(encoding="utf-8"))
        if pages and missing / pages > MISSING_WARNING:
            warnings.append(f"{path.stem}: {missing} of {pages} pages have no transcription")
        (OUTPUT_FOLDER / path.name).write_text(text, encoding="utf-8")
        total_words += len(WORD.findall(text.lower()))

    print(f"Wrote {len(files)} issues, {total_words:,} words, to "
          f"{OUTPUT_FOLDER.relative_to(SCRIPT_DIR)}/")
    if warnings:
        print("\nIssues with many missing pages:")
        for warning in warnings:
            print(f"  {warning}")

    # Everything in corpus/, so you can see how the three sources compare in size
    print("\nThe whole corpus:")
    for folder in sorted(p for p in (SCRIPT_DIR / "corpus").iterdir() if p.is_dir()):
        texts = sorted(folder.glob("*.txt"))
        words = sum(len(WORD.findall(t.read_text(encoding="utf-8").lower())) for t in texts)
        print(f"  {folder.name:20} {len(texts):3} issues  {words:>10,} words")


if __name__ == "__main__":
    main()
