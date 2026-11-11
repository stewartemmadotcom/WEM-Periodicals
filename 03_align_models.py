#!/usr/bin/env python3
"""
Step 3: Align the per-periodical models (orthogonal Procrustes)
================================================================

Each model you trained has its own random orientation: the numbers for
"physical" in the 1896-1919 model can't be compared directly with the
numbers for "physical" in the 1940-1959 model. It's like having three
maps of the same city, each rotated a different amount.

Procrustes alignment finds the rotation that lines one model up with
another as closely as possible, using words that appear often in both
as anchor points. Rotation doesn't stretch or squash anything, so the
distances *within* each model stay exactly the same.

Every model is rotated to line up with the LAST model in MODEL_NAMES.
The aligned versions are saved to models/aligned/.

Method: Hamilton, Leskovec & Jurafsky (2016), "Diachronic Word Embeddings
Reveal Statistical Laws of Semantic Change."
"""

from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors, Word2Vec

SCRIPT_DIR = Path(__file__).parent

# --- Settings ------------------------------------------------------------------

# Each must be a model you trained with 01_train_model.py. Roughly chronological:
# Country Women 1972-79, LC 1974-89, Maize 1990-98.
MODEL_NAMES = ["country-women", "lesbian-connection", "maize"]   # one model per periodical, trained with 01

# Anchor words must appear at least this many times in EVERY model.
ANCHOR_MIN_COUNT = 50

# -------------------------------------------------------------------------------

MODEL_FOLDER = SCRIPT_DIR / "models"
ALIGNED_FOLDER = MODEL_FOLDER / "aligned"


def unit(vectors):
    """Scale every vector to length 1, so only its direction matters."""
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def procrustes_rotation(source, target):
    """The rotation that best turns `source` vectors into `target` vectors.
    Both are (number of anchor words) x (dimensions), rows in the same word order.
    The math: a singular value decomposition (SVD) of source-transposed x target."""
    u, _, vt = np.linalg.svd(source.T @ target)
    return u @ vt


def main():
    models = {}
    for name in MODEL_NAMES:
        path = MODEL_FOLDER / f"{name}.model"
        if not path.exists():
            print(f"Can't find {path.relative_to(SCRIPT_DIR)}. Train it first with 01_train_model.py.")
            return
        models[name] = Word2Vec.load(str(path)).wv

    # Anchor words: in every model, and frequent enough to have stable vectors
    target_name = MODEL_NAMES[-1]
    anchors = [w for w in models[target_name].index_to_key
               if all(w in kv and kv.get_vecattr(w, "count") >= ANCHOR_MIN_COUNT
                      for kv in models.values())]
    print(f"Anchor words (in every period, {ANCHOR_MIN_COUNT}+ times each): {len(anchors):,}")
    print(f"Lining every model up with {target_name}\n")

    target_anchors = unit(models[target_name][anchors])
    ALIGNED_FOLDER.mkdir(parents=True, exist_ok=True)

    for name, kv in models.items():
        words = kv.index_to_key
        vectors = unit(kv[words])

        if name != target_name:
            rotation = procrustes_rotation(unit(kv[anchors]), target_anchors)

            # Show that it worked: how similar is the same word in the two
            # models, before and after rotating? (1.0 = identical direction)
            before = np.mean(np.sum(unit(kv[anchors]) * target_anchors, axis=1))
            vectors = vectors @ rotation          # rotate EVERY word, not just anchors
            after = np.mean(np.sum(unit(kv[anchors]) @ rotation * target_anchors, axis=1))
            print(f"{name} -> {target_name}: average similarity of the same anchor word "
                  f"before {before:.2f}, after {after:.2f}")

        aligned = KeyedVectors(vector_size=kv.vector_size)
        aligned.add_vectors(words, vectors)
        for word in words:
            aligned.set_vecattr(word, "count", kv.get_vecattr(word, "count"))
        aligned.save(str(ALIGNED_FOLDER / f"{name}.kv"))

    print(f"\nSaved aligned models to {ALIGNED_FOLDER.relative_to(SCRIPT_DIR)}/")
    print("Next: python 04_explore_change.py")


if __name__ == "__main__":
    main()
