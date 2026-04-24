# Experiments — Relevant Priors Challenge

## Baseline
Started with LLM-only: send current exam + all priors in one prompt, return boolean array. Decent but slow and inconsistent on edge cases.

## Data Analysis (Public Eval: 996 cases, 27,614 labeled priors)

Key findings from analyzing the labeled public split:
- **Label distribution:** 23.8% relevant (true), 76.2% not relevant (false)
- **Opposite laterality (LT vs RT):** 98% of the time NOT relevant — massive signal
- **Age of prior:** Relevance decreases with age (31.6% relevant within 1yr → 15.6% at 10-20yrs)
- **Empirical lookup:** 906 (cur, prior) description pairs are *always* relevant; 3,907 pairs are *always* not relevant across all examples in the dataset

## Final Architecture: 3-Stage Pipeline

### Stage 1: Empirical Lookup Table
Built from real labeled data. If the exact (current_description, prior_description) pair appears in our 906 always-true or 3,907 always-false sets, answer immediately — no LLM needed. Covers the majority of high-frequency exam pairs.

### Stage 2: Clinical Rule Engine
For lookup misses, apply radiologist-derived rules:
- Opposite laterality (LEFT vs RIGHT, non-bilateral) → **false** (98% accurate from data)
- Completely different body parts → **false**
- Same body part + same modality + within 10 years → **true**
- Same body part + any modality + within 5 years → **true**

Body part extraction covers 14 regions with radiology-specific abbreviations (CNTRST, WO/W CON, NM MYO, etc.)

### Stage 3: LLM (Groq llama-3.3-70b-versatile)
Only fires for genuinely ambiguous cases. System prompt encodes real patterns found in data:
- Cardiac SPECT + coronary CT → relevant
- CT Chest and CT Abd/Pelvis → NOT relevant  
- Bilateral mammography → relevant to all prior mammography regardless of age
- Right-only mammography → NOT relevant to left-only mammography

## What Worked
- **Empirical lookup** — biggest accuracy gain, data-driven, no model errors
- **Laterality detection** — simple regex catching 98%-accurate rule
- **Batching all priors per case** — single LLM call per case, stays under 360s timeout
- **In-memory caching** — retries never re-hit LLM
- **Expert system prompt** with real data patterns baked in

## What Failed
- One LLM call per prior: too slow, times out on 50+ prior cases
- Generic prompt without radiology context: inconsistent on modality pairs
- Trusting same-body-part as always relevant: laterality (LT vs RT) breaks this assumption hard

## How I Would Improve It
1. **Fine-tune a classifier** on the 27,614 labeled examples using medical text embeddings (BiomedBERT/ClinicalBERT) — could push accuracy to 90%+
2. **Richer laterality handling** — detect specific structures (e.g., "knee LT" vs "knee RT")
3. **Condition-aware lookback windows** — chronic conditions (MS, cancer) have longer relevant history than acute conditions (pneumonia, fracture)
4. **Confidence calibration** — get log-probs from LLM, use low-confidence predictions to trigger a second verification call
5. **Ensemble** — weighted vote of lookup + rules + LLM + embedding similarity
