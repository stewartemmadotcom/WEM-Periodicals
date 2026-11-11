# Word Embeddings: Lesbian and Rural Periodicals

Train word2vec models on three periodicals together, using the scripts from
`lc-word-embeddings` as a template:

| Folder | Periodical | Issues | Words |
|---|---|---|---|
| `corpus/country-women` | *Country Women* (1972–79) | 31 | ~1.29 million |
| `corpus/lesbian-connection` | *Lesbian Connection* (1974–89) | 78 | ~1.49 million |
| `corpus/maize` | *Maize* (1990–98) | 29 | ~0.61 million |

| Script | What it does | What you change |
|---|---|---|
| `00_prepare_country_women.py` | Cleans `raw/country-women/` into `corpus/country-women/` | `KEEP_WORDS`, `DROP` |
| `01_train_model.py` | Trains a model on one folder of text | `CORPUS_FOLDER`, `MODEL_NAME` |
| `02_explore_model.py` | Nearest words, a semantic field, a 2D plot | `MODEL_NAME`, `WORD`, `SEED_WORDS` |
| `03_align_models.py` | Aligns the per-periodical models so they can be compared | `MODEL_NAMES` |
| `04_explore_change.py` | Traces one word across the periodicals | `WORD`, `MODEL_NAMES` |
| `05_gender_and_a_word.py` | Which words near a target word lean toward women vs. men | `MODEL_NAME`, `TARGET` |

## Where the texts came from

- `corpus/lesbian-connection/`: copies of `lc-word-embeddings/lc-clean/1974-1989/`, already cleaned
- `corpus/maize/`: copies of `maize-word-embeddings/maize-clean/`, already cleaned
- `raw/country-women/`: copies of `country-women-ocr/combined/`, the Gemini OCR. `00` cleans
  them into `corpus/country-women/` in the same layout as the other two (one paragraph per
  line, blank lines between).

They're copies: re-OCRing or re-cleaning in the original folders doesn't change them. Copy the
new files over and rerun `00` if that happens.

## Setup (done once)

Needs Python 3.13 (gensim doesn't support 3.14 yet). The `venv` is already set up:

```bash
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running it

1. `python 00_prepare_country_women.py`. It also prints the size of each periodical.
2. `python 01_train_model.py`. The default trains one pooled model, `all`, on everything in `corpus/`.
3. `python 02_explore_model.py` and `05_gender_and_a_word.py` on `all`.
4. To compare the periodicals, train one model per folder with 01 (`CORPUS_FOLDER` =
   `corpus/maize`, `MODEL_NAME` = `maize`, and so on), then run `03_align_models.py` and
   `04_explore_change.py`.

## Things to keep in mind

- **The sources differ in size.** Maize is under half the size of the other two, so the pooled
  model leans toward LC's and Country Women's usage. The per-periodical Maize model will have
  the smallest vocabulary. Use the same settings for every model you compare.
- **They differ in time too.** Country Women (1972–79) and LC (1974–89) overlap in the 1970s,
  but Maize (1990–98) comes after both. A difference between Maize and the other two may be a
  difference between decades, not between magazines.
- **They differ in OCR.** LC and Country Women were OCR'd with Gemini, Maize with Claude (and
  Tesseract for a few pages). Country Women has many `[illegible]` spots (about 9,400), which
  00 drops; check the neighbors for OCR fragments.
- **Repeated material** (mastheads, subscription blurbs, LC's contact listings) can pull the
  neighbors of words in it. The text reuse scripts can find it if it gets in the way.
- **41 Country Women pages are Tesseract OCR**, not Gemini (pages Gemini refused, 22 of them in
  `CountryWomen-undated-v1-no3`). They keep printed line breaks and are rougher; see
  `country-women-ocr/ocr-results/<issue>/tesseract-log.csv`.
- **Results change a little every time you train.**
