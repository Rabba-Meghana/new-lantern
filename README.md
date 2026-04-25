# Relevant Priors — New Lantern Challenge

Predict whether each prior radiology exam is relevant for a radiologist reading the current exam.

## Architecture

4-stage pipeline (no external API calls, fully deterministic):

1. **In-memory cache** — MD5-keyed on (current_desc, prior_desc, prior_date)
2. **Empirical lookup table** — 906 always-true + 3,907 always-false pairs mined from public labels
3. **Targeted clinical rules** — 3 high-confidence rules derived from data analysis (>95% accuracy each)
4. **Logistic regression** — TF-IDF bigrams + 15 structured features, 92.7% CV accuracy

## Quickstart

```bash
pip install -r requirements.txt
python train.py --data relevant_priors_public.json
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Test

```bash
python test_sanity.py   # 24 sanity checks
```

## Files

| File | Purpose |
|------|---------|
| `app.py` | FastAPI server — exactly what is deployed |
| `train.py` | Reproduces `model.pkl` and `lookup.json` from public eval JSON |
| `test_sanity.py` | 24 unit/sanity tests |
| `requirements.txt` | Pinned dependencies |
| `experiments.md` | Full write-up |
| `model.pkl` | Trained logistic regression (built by `train.py`) |
| `lookup.json` | Empirical lookup table (built by `train.py`) |
