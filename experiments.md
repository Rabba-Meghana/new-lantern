# Experiments — Relevant Priors Challenge

## Deployed Architecture

The live endpoint runs a **5-stage hybrid pipeline**. All thresholds and hyperparameters are in `config.json` — nothing is hard-coded in inference logic.

1. **In-memory cache** — keyed on MD5(cur_desc + cur_date + pri_desc + pri_date). cur_date is included to prevent collisions across cases sharing the same prior description with different current study dates.
2. **Empirical lookup table** — 906 always-true + 3,907 always-false (current_desc, prior_desc) pairs from public labeled split (min 2 occurrences, no counter-examples)
3. **Targeted clinical rules** — 3 data-validated rules (see table below)
4. **Logistic regression** — TF-IDF bigrams + 15 structured features (threshold: `config.sklearn_threshold`)
5. **ONNX BiomedBERT ensemble** — for sklearn predictions in [`config.onnx_uncertainty_low`, `config.onnx_uncertainty_high`]; averaged with sklearn at `config.onnx_threshold`

BiomedBERT loads in a background thread. sklearn handles all requests immediately and ONNX activates automatically once ready (~30s after startup). If `onnx_int8/` is absent the server degrades gracefully to sklearn-only.

---

## Data (Public Split Only)

All measurements are on the public labeled split (27,614 priors, 996 cases). The private split is unseen. CV figures are 5-fold on the full public set — not a true held-out result.

| Statistic | Value |
|-----------|-------|
| Label distribution | 23.8% relevant, 76.2% not relevant |
| Opposite laterality pairs | 98.2% not relevant (n=346) |
| Age <1yr relevance rate | 31.6% |
| Age 10-20yr relevance rate | 15.6% |

---

## Targeted Rules (public-split validated)

| Rule | Evidence | Accuracy |
|------|----------|----------|
| Mammography vs chest X-ray → not relevant | 643 false, 3 true | 99.5% |
| CT Chest vs CT Abdomen/Pelvis → not relevant | 319 false, 5 true | 98.5% |
| Bilateral mam vs unilateral mam → relevant | 294 true, 29 false | 91.0% |

---

## Stage 4: Logistic Regression

**Structured features (15):** prior age normalised, body-part overlap count + Jaccard (13 regions), modality overlap count + Jaccard (6 classes), opposite/same/bilateral laterality, same-description flag, recency flags (≤1yr, ≤3yr, >10yr), any-body-part + any-modality + both booleans.

**TF-IDF:** 8,000 bigrams on `"CURRENT: {desc} ||| PRIOR: {desc}"`, token `[A-Z0-9]+`.

**Public-split 5-fold CV:** 92.7% ± 0.6% (NOTE: this is public-split CV, not a true held-out result)

---

## Per-Category Ablation (public split, sklearn + lookup, no ONNX)

Measured on all 27,614 public priors using the full pipeline minus ONNX:

| Category | Accuracy | n |
|----------|----------|---|
| age_0_1yr | 90.4% | 6,359 |
| age_1_3yr | 95.9% | 6,493 |
| age_3_10yr | 96.8% | 8,854 |
| age_10plus_yr | 97.0% | 5,908 |
| body_part_overlap | 93.1% | 6,169 |
| no_body_part_overlap | 95.7% | 21,445 |
| modality_overlap | 91.3% | 6,880 |
| no_modality_overlap | 96.4% | 20,734 |
| opposite_laterality | 96.2% | 346 |

**Key insight from ablation:** The hardest cases are recent priors (age <1yr, 90.4%) with body-part overlap (93.1%) and modality overlap (91.3%). These are exactly the cases where description-only models fail — the same exam type at the same body part, but clinical context determines relevance. This is where BiomedBERT adds value.

---

## Stage 5: ONNX BiomedBERT (97.4% val accuracy)

Fine-tuned `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` on 27,614 public examples.

- Train/val split: 24,852 / 2,762 (stratified 90/10 from public split — not a true held-out result)
- 3 epochs, batch 32, lr=2e-5, fp16, Google Colab T4 GPU
- Epoch 1: 96.2% | Epoch 2: 97.0% | Epoch 3: 97.4% (val accuracy on public split)

**Reproducibility note:** BiomedBERT fine-tuning is not in `train.py` (requires GPU). The `--export-onnx` flag in `train.py` handles only the ONNX export step, given a pre-trained model directory. See experiments.md for the full Colab training procedure.

Exported to ONNX int8 using `optimum` (`avx2`, dynamic quantisation). Model size ~68MB. Only fires for sklearn-uncertain predictions (~10% of cases). Final prediction = average(sklearn_prob, onnx_prob) ≥ `config.onnx_threshold`.

---

## What Worked

1. Lookup table — zero error on high-frequency seen pairs
2. Laterality — 98.2% of LT vs RT pairs are not-relevant (confirmed by ablation)
3. BiomedBERT — handles novel descriptions TF-IDF cannot generalise to
4. Background thread loading — instant startup, no blocking
5. Config-driven thresholds — all decision boundaries in `config.json`, not hard-coded
6. Cache with cur_date — prevents cross-case collisions

## What Failed

- Full BiomedBERT blocking startup — fixed with background thread
- LLM (Groq) — external latency caused instability, removed
- TF-IDF alone — 65% CV accuracy without structured features
- Cache key without cur_date — caused cross-case collisions (fixed)
- Hard-coded thresholds — replaced with `config.json`
- Duplicate entrypoint in train.py — fixed to single `if __name__ == "__main__"` block

## How I Would Improve It

1. **True held-out ablation** — reserve 20% of public split before training for clean error analysis
2. **GPU deployment** — g4dn.xlarge for full BiomedBERT on all predictions
3. **Per-exam-type threshold** — calibrate separately for mammography, cardiac, neuro using public split
4. **Condition-aware lookback** — chronic conditions warrant longer prior history than acute
5. **Ship BiomedBERT fine-tuning in train.py** — currently requires Colab/GPU separately

---

## Reproducibility

```bash
# Install dependencies
pip install -r requirements.txt

# Rebuild model.pkl and lookup.json (also runs ablation)
python train.py --data relevant_priors_public.json

# Run 32 sanity checks
python test_sanity.py

# Start server
uvicorn app:app --host 0.0.0.0 --port 8000

# Export ONNX (requires fine-tuned BiomedBERT in ./biomedbert_priors)
python train.py --data relevant_priors_public.json --export-onnx --bert-dir ./biomedbert_priors
```

Python 3.10+. All hyperparameters in `config.json`.
