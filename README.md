# Relevant Priors — New Lantern Challenge

Predict whether each prior radiology exam is relevant for a radiologist reading the current exam.

## Architecture

5-stage hybrid pipeline. All thresholds in `config.json`:

1. **In-memory cache** — MD5(cur_desc + cur_date + pri_desc + pri_date)
2. **Empirical lookup table** — 906 always-true + 3,907 always-false pairs from public labels
3. **Targeted clinical rules** — mammography/CXR, CT chest/abdomen, bilateral/unilateral mam
4. **Logistic regression** — TF-IDF bigrams + 15 structured features (92.7% CV, public split)
5. **ONNX BiomedBERT ensemble** — for uncertain sklearn predictions (97.4% val, public split)

Server starts instantly on sklearn. ONNX loads in background thread and activates automatically. Degrades gracefully to sklearn-only if `onnx_int8/` is absent.

## Quickstart

```bash
pip install -r requirements.txt
python train.py --data relevant_priors_public.json   # builds model.pkl + lookup.json
python test_sanity.py                                 # 32 sanity checks, all must pass
uvicorn app:app --host 0.0.0.0 --port 8000
```

## ONNX Export (requires GPU + fine-tuned BiomedBERT)

```bash
# After fine-tuning BiomedBERT (see experiments.md for Colab procedure)
python train.py --data relevant_priors_public.json --export-onnx --bert-dir ./biomedbert_priors
```

## Configuration

All thresholds and hyperparameters are in `config.json`. No hard-coded values in inference logic.

## Files

| File | Purpose |
|------|---------|
| `app.py` | FastAPI server — exactly what is deployed |
| `train.py` | Builds `model.pkl` + `lookup.json`; optionally exports ONNX |
| `test_sanity.py` | 32 sanity checks covering all helpers and edge cases |
| `config.json` | All thresholds and hyperparameters |
| `requirements.txt` | Pinned dependencies |
| `experiments.md` | Full write-up with ablation |
| `model.pkl` | Trained logistic regression (built by `train.py`) |
| `lookup.json` | Empirical lookup table (built by `train.py`) |
| `onnx_int8/` | Quantised BiomedBERT (built via `--export-onnx`, requires GPU) |
