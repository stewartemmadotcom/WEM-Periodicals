#!/usr/bin/env python3
"""
Step 1: Train a word2vec model
==============================

Reads every .txt file in CORPUS_FOLDER, splits the text into words, and
trains a word embedding model with gensim's Word2Vec. The model is saved
to models/<MODEL_NAME>.model.

corpus/ has one folder per periodical (lesbian-connection, maize,
country-women). Run 00_prepare_country_women.py first to build
corpus/country-women. Then:

    corpus                     ->  all       (one model on all three, pooled)
    corpus/lesbian-connection  ->  lesbian-connection   (one per periodical,
    corpus/maize               ->  maize      for 03 and 04 to compare)
    corpus/country-women       ->  country-women

You may see "Exception ignored in: 'gensim.models.word2vec_inner.our_dot_float'"
a few times while it trains. It's a known, harmless gensim bug (issue #3665):
the model still trains correctly.
"""

import os
import re
import time
from pathlib import Path

from gensim.models import Word2Vec

SCRIPT_DIR = Path(__file__).parent

# --- Change these two lines for each model -------------------------------------

CORPUS_FOLDER = SCRIPT_DIR / "corpus"
MODEL_NAME = "all"

# --- Training settings (keep them the same for every model you'll compare) -----

VECTOR_SIZE = 100   # how many numbers (dimensions) describe each word
WINDOW = 5          # how many words on each side count as "context"
MIN_COUNT = 10      # ignore words that appear fewer times than this
SG = 1              # 1 = skip-gram (predict the neighbors from the word); 0 = CBOW
EPOCHS = 15          # how many times to read through the whole corpus

# -------------------------------------------------------------------------------

MODEL_FOLDER = SCRIPT_DIR / "models"

# A "word" is a run of letters, optionally joined by an apostrophe or hyphen
# (so "women's" and "self-control" stay whole)
WORD = re.compile(r"[a-z]+(?:['-][a-z]+)*")


def read_corpus(folder):
    """Turn every text file into lists of lowercased words.
    Each paragraph becomes one list (split into chunks of 1,000 words,
    since word2vec reads at most 10,000 words at a time)."""
    sentences = []
    files = sorted(folder.rglob("*.txt"))
    for path in files:
        for paragraph in path.read_text(encoding="utf-8").split("\n\n"):
            words = WORD.findall(paragraph.lower())
            for i in range(0, len(words), 1000):
                sentences.append(words[i:i + 1000])
    return files, sentences


def main():
    if not CORPUS_FOLDER.exists():
        print(f"Can't find {CORPUS_FOLDER}")
        print("Check CORPUS_FOLDER (and run 00_prepare_country_women.py).")
        return

    print(f"Reading {CORPUS_FOLDER} ...")
    files, sentences = read_corpus(CORPUS_FOLDER)
    total_words = sum(len(s) for s in sentences)
    print(f"  {len(files)} files, {total_words:,} words")

    print("Training (this can take a minute or two) ...")
    start = time.time()
    model = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=SG,
        epochs=EPOCHS,
        workers=os.cpu_count(),   # use every processor core to speed it up
        seed=42,
    )
    print(f"  done in {time.time() - start:.0f} seconds")
    print(f"  vocabulary: {len(model.wv):,} words (each appears at least {MIN_COUNT} times)")

    MODEL_FOLDER.mkdir(exist_ok=True)
    model_path = MODEL_FOLDER / f"{MODEL_NAME}.model"
    model.save(str(model_path))
    print(f"\nSaved {model_path.relative_to(SCRIPT_DIR)}")

    # A quick check that the model learned something sensible
    print("\nQuick check, the 8 words closest to 'lesbian':")
    for word, score in model.wv.most_similar("lesbian", topn=8):
        print(f"  {word:20} {score:.3f}")


if __name__ == "__main__":
    main()
