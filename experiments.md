# Experiments — Relevant Priors Challenge

## Baseline
Started with a naive LLM-only approach: send current exam + all priors to the LLM in one prompt, ask for a boolean array. Accuracy was decent but slow on large cases and inconsistent on ambiguous modality pairs.

## What Worked

### 1. Rule-Based Pre-filter (biggest accuracy gain)
Built a two-stage pipeline:
- **Stage 1:** Rule-based classifier using body part and modality extraction from study descriptions
- **Stage 2:** LLM only for ambiguous cases rules can't confidently decide

Rules:
- Same body part + same modality + within 10 years → **always true** (no LLM needed)
- Same body part + any modality + within 5 years → **always true**
- Completely different body parts (no overlap) → **always false**
- Everything else → send to LLM

This cut LLM calls by ~60% on the public eval, massively reducing latency and cost.

### 2. Body Part & Modality Keyword Extraction
Built a comprehensive keyword dictionary covering:
- 12 body regions (brain, cervical spine, thoracic spine, lumbar spine, chest, abdomen, pelvis, upper extremity, lower extremity, breast, vascular, whole body)
- 6 modality classes (MRI, CT, X-ray, Ultrasound, PET/Nuclear, Mammography)

Normalized radiology shorthand: "CNTRST" → CT contrast, "CXR" → chest X-ray, etc.

### 3. Expert System Prompt
Gave the LLM a detailed radiologist persona with explicit clinical relevance rules rather than a generic prompt. This reduced hallucinations and edge case errors significantly.

### 4. In-Memory Caching
Keyed on (current_description, prior_description, prior_date). Identical study pairs across cases or retries never hit the LLM twice.

### 5. Batch All Priors Per Case
All priors for a given case sent in one LLM call. One call per case, not one per prior.

## What Failed
- **One LLM call per prior:** Too slow, timed out on cases with 50+ priors
- **Pure LLM approach:** Inconsistent on edge cases like "CT CHEST" vs "CXR" (both chest, different modality)
- **Overly aggressive false rules:** Early version flagged CT head vs MRI brain as irrelevant — wrong, they're the same body part

## How I Would Improve It
1. **Train on public eval labels:** Fine-tune a small classifier on the 27,614 labeled examples — this alone would likely push accuracy to 90%+
2. **Embeddings similarity:** Use a medical text embedding model (e.g., BiomedBERT) to compute semantic similarity between study descriptions as an additional signal
3. **Date decay scoring:** Weight relevance by recency — a prior from 6 months ago is more valuable than one from 8 years ago for most conditions
4. **Condition-aware rules:** Chronic conditions (MS, cancer follow-up) should have longer lookback windows than acute conditions (stroke)
5. **Confidence thresholding:** Get log-probabilities from the LLM and use confidence to decide when to abstain vs. predict
6. **Ensemble:** Combine rule-based + LLM + embedding similarity with a learned weighted vote
