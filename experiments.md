# Experiments — Relevant Priors Challenge

## What Was Actually Deployed

The live endpoint runs a **3-stage pipeline** with no external LLM calls:

1. **Empirical lookup table** (instant, zero model inference)
2. **Logistic regression classifier** (TF-IDF bigrams + 15 structured features)
3. **Deterministic fallback** (body-part overlap) if sklearn inference fails

Everything is reproducible from `train.py` using only the public eval JSON.

---

## Data Analysis (Public Split: 996 cases, 27,614 labeled priors)

All analysis was performed on the full public labeled split **before** training.
No separate held-out set was carved out from this split; CV figures are 5-fold CV on the full public set.
The private split is unseen.

Key findings:
- **Label distribution:** 23.8% relevant, 76.2% not relevant
- **Opposite laterality (LT vs RT):** 98.2% of pairs labeled not-relevant — strongest single signal
- **Age decay:** 31.6% relevant within 1 year → 15.6% at 10-20 years
- **Deterministic pairs:** 906 (current, prior) description pairs appear exclusively as relevant across all public examples; 3,907 appear exclusively as not-relevant (min 2 occurrences each)

---

## Stage 1: Empirical Lookup Table

**What it is:** A dictionary of (current\_description, prior\_description) string pairs observed exclusively as `true` or exclusively as `false` in the public labeled set, with a minimum frequency threshold of 2.

**Data provenance note:** This table is mined entirely from the public labeled split (`relevant_priors_public.json`). It will generalise only to the extent that the private split contains the same high-frequency exam-pair combinations, which is likely for common modalities (CT Chest, Brain MRI, Mammography) but not guaranteed for rare or novel descriptions.

**Result:** Zero classification error on seen pairs. Covers the majority of high-frequency combinations instantly.

---

## Stage 2: Logistic Regression Classifier

**Features (15 structured + 8,000 TF-IDF bigrams):**

Structured features:
- Prior age in years (normalised to [0, 1] over 20-year range)
- Body-part overlap count and Jaccard similarity (13 regions)
- Modality overlap count and Jaccard similarity (6 modality classes)
- Opposite laterality flag (LT vs RT, non-bilateral)
- Same laterality flag
- Both bilateral flag
- Same description flag (exact string match)
- Recency flags: within 1 year, within 3 years, older than 10 years
- Any body-part overlap (boolean)
- Any modality overlap (boolean)
- Both body-part and modality overlap (boolean)

TF-IDF: 8,000 character bigrams on concatenated `"CURRENT: {desc} ||| PRIOR: {desc}"` string, uppercase, token pattern `[A-Z0-9]+`.

**Training:** Fit on all 27,614 public labeled examples. No data held out from the public split.

**Cross-validated accuracy (5-fold, public split):** 92.7% ± 0.6%

**Decision threshold:** 0.35 (lower than default 0.5 to account for 23.8% positive base rate).

---

## Stage 3: Deterministic Fallback

If sklearn inference raises any exception (malformed input, missing keys, etc.), the fallback predicts `True` if and only if the current and prior descriptions share at least one body-part keyword match. This is deterministic and never silently returns an optimistic default.

---

## What Worked

1. **Lookup table** — largest single accuracy gain; common exam pairs resolved with 100% public-split accuracy
2. **Laterality detection** — opposite LT/RT flag is the strongest individual structured feature; 98.2% of such pairs are not-relevant
3. **Batching all priors per case** — single sklearn inference call per case, well within 360s timeout
4. **In-memory MD5 cache** — retries and repeated (current, prior, date) triples never re-run inference
5. **Conservative unresolved default** — unresolved cases default to `False` rather than `True`, reducing false-positive noise

## What Failed

- **LLM (Groq llama-3.3-70b-versatile):** Added for uncertain cases but caused server instability under load due to external network latency; removed entirely in favour of the deterministic fallback
- **TF-IDF alone (no structured features):** 65% CV accuracy — radiology abbreviations are too sparse for text matching alone
- **BiomedBERT fine-tuning:** Trained to 97.2% validation accuracy in Colab (3 epochs, 24,852 train / 2,762 val, stratified split) but CPU inference on t2.medium was too slow for the 360s evaluator timeout; not shipped in the final endpoint

## How I Would Improve It

1. **Ship BiomedBERT with ONNX quantisation** — export the fine-tuned model to ONNX int8, reducing CPU inference time by ~4x and making it viable within the timeout
2. **Larger instance with GPU** — a g4dn.xlarge would run BiomedBERT inference in milliseconds per batch
3. **Threshold per exam-type** — tune the 0.35 threshold separately for high-volume exam types (Mammography, Chest CT, Brain MRI) using calibration curves on the public split
4. **Condition-aware lookback window** — chronic conditions (MS, cancer staging) warrant longer history than acute presentations (pneumonia, fracture)
5. **Ensemble** — weighted combination of BiomedBERT probabilities + logistic regression + lookup, with the lookup table as a hard override

---

## Reproducibility

```bash
# Install dependencies
pip install -r requirements.txt

# Rebuild model.pkl and lookup.json from the public eval JSON
python train.py --data relevant_priors_public.json

# Run the server
uvicorn app:app --host 0.0.0.0 --port 8000
```

Pinned dependencies are in `requirements.txt`. Python 3.10+ required.

---

## Clinical Validation

To sanity-check predictions, I manually reviewed 30 random predictions against radiology domain knowledge:

- **Brain MRI Stroke vs prior Brain MRI Stroke (6 years ago):** Predicted relevant. Correct — radiologists need baseline comparison for infarct evolution.
- **CT Chest vs CT Abdomen/Pelvis:** Predicted not relevant. Correct — different anatomical regions, no clinical overlap in routine reads.
- **Mammography screening bilateral vs prior diagnostic unilateral (same side):** Predicted relevant. Correct — prior unilateral study provides lesion-level comparison.
- **Mammography screening bilateral vs prior chest X-ray:** Predicted not relevant. Correct — chest X-ray shows lungs/mediastinum, not breast parenchyma.
- **Echo TTE vs prior CT Chest:** Context-dependent. Our model predicts relevant ~29% of the time for this pair (data-driven), which aligns with clinical practice: cardiac CT (CTA coronary) is relevant to Echo, but general chest CT is not. This is the hardest class — a condition-aware model would outperform a description-only model here.

The main failure mode we identified: **context-dependent pairs** where the same two modality/region combinations are clinically relevant in one patient context (oncology staging) but not in another (routine screening). Description-only models cannot resolve this without patient history.
