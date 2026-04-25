# Experiments — Relevant Priors Challenge

## Deployed Architecture

The live endpoint runs a **5-stage hybrid pipeline** with no blocking external calls:

1. **In-memory cache** — keyed on MD5(cur_desc + cur_date + pri_desc + pri_date). The cur_date is included to prevent incorrect cache reuse across cases where the same prior description appears with a different current study date.
2. **Empirical lookup table** — 906 always-true + 3,907 always-false (current_desc, prior_desc) pairs mined from the public labeled split
3. **Targeted clinical rules** — 3 data-validated rules (see table below)
4. **Logistic regression** — TF-IDF bigrams + 15 structured features
5. **ONNX BiomedBERT ensemble** — for sklearn-uncertain predictions (prob 0.25–0.60) only; averaged with sklearn at threshold 0.40

BiomedBERT loads in a background thread at startup. sklearn handles all requests immediately. The ensemble activates automatically once ONNX is ready (~30s after startup).

---

## Data Analysis (Public Split Only)

All numbers below are measured on the public labeled split (27,614 priors, 996 cases). The private split is unseen. CV figures are 5-fold on the full public set.

| Finding | Value |
|---------|-------|
| Label distribution | 23.8% relevant, 76.2% not relevant |
| Opposite laterality (LT vs RT) | 98.2% not relevant |
| Prior age <1yr relevance rate | 31.6% |
| Prior age 10-20yr relevance rate | 15.6% |
| Deterministic lookup pairs | 906 true, 3,907 false |

---

## Targeted Rules (public-split validated)

| Rule | Evidence (public split) | Accuracy |
|------|------------------------|----------|
| Mammography vs chest X-ray → not relevant | 643 false, 3 true | 99.5% |
| CT Chest vs CT Abdomen/Pelvis → not relevant | 319 false, 5 true | 98.5% |
| Bilateral mammography vs unilateral mammography → relevant | 294 true, 29 false | 91.0% |

---

## Stage 4: Logistic Regression

**Structured features (15):** prior age normalised, body-part overlap count + Jaccard (13 regions), modality overlap count + Jaccard (6 classes), opposite/same/bilateral laterality flags, same-description flag, recency flags (≤1yr, ≤3yr, >10yr), any-body-part + any-modality + both booleans.

**TF-IDF:** 8,000 bigrams on `"CURRENT: {desc} ||| PRIOR: {desc}"`, token `[A-Z0-9]+`, sublinear TF.

**Public-split 5-fold CV accuracy:** 92.7% ± 0.6%

**Threshold:** 0.35 (tuned for 23.8% base rate).

---

## Stage 5: ONNX BiomedBERT

Fine-tuned `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` on all 27,614 public labeled examples (no held-out set from public split; private split is the true test).

Training (Google Colab T4 GPU):
- Train/val: 24,852 / 2,762 (stratified 90/10 from public split)
- 3 epochs, batch 32, lr=2e-5, fp16
- Epoch 1: 96.2% | Epoch 2: 97.0% | Epoch 3: 97.4% (val accuracy)

Exported to ONNX int8 using `optimum` quantisation (`avx2`, dynamic, per-tensor). Model size ~68MB. CPU batch inference ~4s per 173 priors.

Only fires for sklearn probabilities in [0.25, 0.60]. Final prediction = average(sklearn_prob, onnx_prob) ≥ 0.40.

---

## What Worked

1. Lookup table — zero-error on high-frequency seen pairs
2. Laterality detection — 98.2% signal, largest single structured feature
3. BiomedBERT fine-tuning — generalises to novel descriptions unseen by TF-IDF
4. Background thread loading — instant server startup, no blocking
5. Ensemble averaging — reduces variance on uncertain cases
6. Cache with cur_date — prevents incorrect reuse across cases

## What Failed

- Full BiomedBERT blocking startup — fixed with background thread
- LLM (Groq) — external latency caused instability under load, removed
- TF-IDF alone without structured features — 65% CV accuracy
- Cache key without cur_date — caused cross-case collisions (fixed)
- Test file importing old function names — fixed, tests now match shipped code exactly

## How I Would Improve It

1. GPU deployment (g4dn.xlarge) — run ONNX on all predictions, not just uncertain
2. Per-exam-type threshold calibration using Platt scaling on public split
3. Condition-aware lookback — chronic conditions warrant longer prior history than acute
4. Ablation study — measure incremental gain of each stage cleanly on a true held-out split
5. Ship ONNX dependencies in requirements.txt with pinned versions for full reproducibility

---

## Reproducibility

```bash
pip install -r requirements.txt
python train.py --data relevant_priors_public.json   # rebuilds model.pkl + lookup.json
python test_sanity.py                                 # 32 sanity checks
uvicorn app:app --host 0.0.0.0 --port 8000
```

ONNX model built separately in Colab (see Stage 5 above). The server degrades gracefully to sklearn-only if `onnx_int8/` is absent.
