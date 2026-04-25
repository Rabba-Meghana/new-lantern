# Relevant Priors — New Lantern Challenge

Predict whether each prior radiology exam is relevant for a radiologist reading the current exam.

## Architecture

5-stage hybrid pipeline:

1. **In-memory cache** — MD5 keyed on (current_desc, prior_desc, prior_date)
2. **Empirical lookup table** — 906 always-true + 3,907 always-false pairs from public labels
3. **Targeted clinical rules** — mammography/CXR, CT chest/abdomen, bilateral/unilateral mam
4. **Logistic regression** — TF-IDF bigrams + 15 structured features (92.7% CV accuracy)
5. **ONNX BiomedBERT ensemble** — for uncertain sklearn predictions only (97.4% val accuracy)

Server starts instantly on sklearn. ONNX loads in background thread and activates automatically.

## Quickstart

```bash
pip install -r requirements.txt
python train.py --data relevant_priors_public.json
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Test

```bash
python test_sanity.py   # 24 sanity checks, all must pass
```

## Files

| File | Purpose |
|------|---------|
| `app.py` | FastAPI server — exactly what is deployed |
| `train.py` | Reproduces `model.pkl` and `lookup.json` |
| `test_sanity.py` | 24 unit/sanity tests |
| `requirements.txt` | Pinned dependencies |
| `experiments.md` | Full write-up |
| `model.pkl` | Trained logistic regression |
| `lookup.json` | Empirical lookup table |
| `onnx_int8/` | Quantised BiomedBERT (built in Colab, see experiments.md) |
