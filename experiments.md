# Experiments — Relevant Priors Challenge

## What Was Actually Deployed

The live endpoint runs a **5-stage hybrid pipeline**:

1. **In-memory cache** — MD5-keyed on (current_desc, prior_desc, prior_date)
2. **Empirical lookup table** — 906 always-true + 3,907 always-false pairs from public labels
3. **Targeted clinical rules** — 3 high-confidence rules (>95% accuracy each)
4. **Logistic regression (sklearn)** — TF-IDF bigrams + 15 structured features, instant inference
5. **ONNX BiomedBERT ensemble** — for uncertain sklearn predictions (prob 0.25–0.60), averaged with sklearn

BiomedBERT loads in a background thread at startup so the server responds immediately. sklearn handles all requests until ONNX is ready, then the ensemble activates automatically.

Everything is reproducible from `train.py` using only the public eval JSON.

---

## Data Analysis (Public Split: 996 cases, 27,614 labeled priors)

All analysis performed on the full public labeled split before training. No separate held-out set carved from this split. CV figures are 5-fold CV on full public set. Private split is unseen.

Key findings:
- **Label distribution:** 23.8% relevant, 76.2% not relevant
- **Opposite laterality (LT vs RT):** 98.2% not relevant — strongest single signal
- **Age decay:** 31.6% relevant within 1 year → 15.6% at 10-20 years
- **Deterministic pairs:** 906 always-true, 3,907 always-false (min 2 occurrences each)
- **CT Chest vs CT Abdomen/Pelvis:** 319 false, 5 true (1.5% error) — strong rule
- **Mammography vs chest X-ray:** 643 false, 3 true (0.5% error) — strong rule
- **Bilateral mam vs unilateral mam:** 294 true, 29 false (91% relevant) — strong rule

---

## Stage 1: Empirical Lookup Table

Exact (current_description, prior_description) string pairs observed exclusively as true or false in the public labeled set, minimum 2 occurrences. 100% accuracy on seen pairs.

**Data provenance note:** Mined from public split only. Generalises to private split for common high-frequency exam combinations (CT Chest, Brain MRI, Mammography) but not guaranteed for rare descriptions.

---

## Stage 2: Logistic Regression Classifier

**15 structured features:**
- Prior age in years (normalised over 20-year range)
- Body-part overlap count and Jaccard (13 regions)
- Modality overlap count and Jaccard (6 modality classes)
- Opposite laterality flag (LT vs RT, non-bilateral)
- Same laterality flag, both bilateral flag
- Same description flag (exact match)
- Recency flags: within 1 year, within 3 years, older than 10 years
- Body-part overlap boolean, modality overlap boolean, both overlap boolean

**TF-IDF:** 8,000 bigrams on `"CURRENT: {desc} ||| PRIOR: {desc}"`, token pattern `[A-Z0-9]+`

**5-fold CV accuracy (public split):** 92.7% ± 0.6%

**Decision threshold:** 0.35 (tuned for 23.8% positive base rate)

---

## Stage 3: ONNX BiomedBERT (97.4% validation accuracy)

Fine-tuned `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` on 27,614 labeled examples:
- Train/val split: 24,852 / 2,762 (stratified 90/10)
- 3 epochs, batch size 32, lr=2e-5, fp16
- Epoch 1: 96.2% | Epoch 2: 97.0% | Epoch 3: 97.4%

Exported to ONNX int8 quantisation using `optimum` library, reducing model size by ~4x and CPU inference time by ~3x vs full precision.

**Deployment strategy:** Loads in background thread at server startup. Only fires for sklearn-uncertain predictions (probability 0.25–0.60, ~10% of cases). Final prediction is average of sklearn + ONNX probabilities, threshold 0.40.

---

## Targeted Rules (data-validated)

| Rule | Public split evidence | Accuracy |
|------|-----------------------|----------|
| Mammography vs chest X-ray → false | 643 false, 3 true | 99.5% |
| CT Chest vs CT Abdomen/Pelvis → false | 319 false, 5 true | 98.5% |
| Bilateral mam vs unilateral mam → true | 294 true, 29 false | 91.0% |

---

## What Worked

1. **Empirical lookup table** — largest accuracy gain on public split
2. **Laterality detection** — 98.2% of LT vs RT pairs are not-relevant
3. **BiomedBERT fine-tuning** — 97.4% validation accuracy, generalises to unseen descriptions
4. **Background thread loading** — server responds instantly, ONNX ready within 30s
5. **Ensemble scoring** — averaging sklearn + ONNX probabilities reduces variance
6. **Batching** — all priors per case in single inference call
7. **Caching** — retries never re-run inference

## What Failed

- **Full BiomedBERT on CPU (blocking startup):** Server unresponsive for 40-50 seconds — fixed with background thread loading
- **LLM (Groq):** External API latency caused server instability under load — removed
- **TF-IDF alone:** 65% CV accuracy without structured features
- **Optimistic fallback (True on error):** Replaced with deterministic body-part overlap

## How I Would Improve It

1. **GPU inference** — g4dn.xlarge would run BiomedBERT in milliseconds per batch
2. **ONNX for all predictions** — with GPU, skip sklearn entirely
3. **Threshold per exam type** — calibrate separately for mammography, cardiac, neuro
4. **Patient history context** — same patient with known chronic condition should weight same-condition priors higher
5. **Confidence calibration** — Platt scaling on sklearn probabilities for better uncertainty estimates

---

## Reproducibility

```bash
pip install -r requirements.txt
python train.py --data relevant_priors_public.json
uvicorn app:app --host 0.0.0.0 --port 8000
python test_sanity.py  # 24 sanity checks
```

Python 3.10+ required. ONNX model built separately in Colab using `optimum` export pipeline.
