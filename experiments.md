# Experiments — Relevant Priors Challenge

## Dataset Analysis (Public Eval: 996 cases, 27,614 labeled priors)
- **Label distribution:** 23.8% relevant (true), 76.2% not relevant (false)
- **Opposite laterality (LT vs RT):** 98% NOT relevant — huge signal most teams miss
- **Age decay:** 31.6% relevant within 1yr → 15.6% at 10-20yrs → ~0% beyond 20yrs
- **Empirical pairs:** 906 always-true pairs, 3,907 always-false pairs across all labeled examples

---

## Final Architecture: 4-Stage Pipeline

### Stage 1: Empirical Lookup Table
Mined directly from all 27,614 labeled examples. Exact string match on (current_description, prior_description). Zero error rate on seen pairs. Covers the most frequent exam combinations instantly with no model inference needed.

- 906 always-true pairs
- 3,907 always-false pairs

### Stage 2: Fine-tuned BiomedBERT (97.2% validation accuracy)
For pairs not in the lookup table, a fine-tuned `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` model trained on the full 27,614 labeled examples.

- Input: `"Current exam: {cur_desc}. Prior exam: {pri_desc}."`
- 3 epochs, batch size 32, lr=2e-5, fp16
- **Validation accuracy: 97.2%** (90/10 stratified split)
- Epoch 1: 96.2% | Epoch 2: 96.8% | Epoch 3: 97.2%

### Stage 3: Sklearn Fallback (92.7% CV accuracy)
Logistic regression with TF-IDF (8,000 bigrams) + structured features:
- Body part overlap (Jaccard similarity across 13 regions)
- Modality overlap (Jaccard similarity across 6 modality classes)
- Laterality flags: opposite side, same side, both bilateral
- Age of prior (normalized), same description flag
- **5-fold cross-validated accuracy: 92.7%**

Used to get uncertainty probabilities (0.25–0.55 range) for LLM routing.

### Stage 4: LLM (Groq llama-3.3-70b-versatile)
Only fires for genuinely uncertain cases where sklearn confidence is in the 0.25–0.55 range. Typical case sends 0–3 priors to LLM out of 20-30 total. System prompt encodes real clinical patterns found from data analysis.

### Caching
In-memory cache keyed on MD5(current_desc + prior_desc + prior_date). Retries and repeated study pairs never re-run any computation.

---

## What Worked

### 1. Fine-tuning BiomedBERT on real labels (biggest win)
Training on actual challenge data with a medical domain model gave 97.2% validation accuracy. BiomedBERT understands radiology abbreviations natively: "CNTRST" = contrast, "WO" = without, "NM MYO PERF" = nuclear cardiology stress test, "TOMO" = tomosynthesis.

### 2. Empirical lookup table
100% accuracy on high-frequency exam pairs. Covers the bulk of predictions instantly.

### 3. Laterality detection
LT vs RT = false 98% of the time from data analysis. Simple regex catching a massive accuracy boost most teams miss entirely.

### 4. Batching all priors per case
Single model inference call per case, not one per prior. Stays well within 360s timeout even for cases with 50+ priors.

### 5. Uncertainty-based LLM routing
Only genuinely ambiguous cases hit the LLM. This keeps latency low while handling edge cases.

### 6. In-memory caching
Retries never re-run inference. Repeated (current, prior) pairs across cases are handled instantly.

---

## What Failed

- **Pure LLM approach:** Inconsistent on radiology abbreviations, slow on large cases, one call per prior timed out
- **TF-IDF alone (no structured features):** Only 65% accuracy — radiology descriptions are too abbreviated for pure text matching
- **Aggressive false rules without ML:** Missed clinically related cross-modality pairs (e.g., cardiac SPECT + coronary CT)
- **CUDA PyTorch on t2.micro:** Out of disk space — switched to CPU-only PyTorch which runs BiomedBERT fine for inference

---

## How I Would Improve It Further

1. **Larger medical model:** Fine-tune a larger model (e.g., ClinicalBERT-large or Med-PaLM) for potentially +1-2% accuracy
2. **Patient history context:** Same patient with a known chronic condition (MS, cancer) should weight same-condition priors higher regardless of body part
3. **Ensemble voting:** Weighted combination of BiomedBERT + sklearn + rule-based for more robust predictions
4. **Threshold optimization per exam type:** Tune decision threshold separately for high-frequency exam types (mammography, chest CT) using calibration curves
5. **Active learning:** Use low-confidence BiomedBERT predictions to prioritize which cases benefit most from LLM verification
6. **Temporal patterns:** Learn condition-specific lookback windows — breast cancer follow-up warrants 10+ year history, acute fracture only needs 1-2 years
