# Experiments — Relevant Priors Challenge

## Dataset Analysis (Public Eval: 996 cases, 27,614 labeled priors)
- **Label distribution:** 23.8% relevant, 76.2% not relevant
- **Opposite laterality (LT vs RT):** 98% NOT relevant — massive signal
- **Age decay:** 31.6% relevant within 1yr → 15.6% at 10-20yrs
- **Empirical pairs:** 906 always-true, 3,907 always-false across all labeled examples

## Final Architecture: 4-Stage Pipeline

### Stage 1: Empirical Lookup Table
Mined directly from 27,614 labeled examples. Exact (current_desc, prior_desc) string matches from training data. Zero error rate on seen pairs. Covers the most frequent exam combinations.

### Stage 2: ML Classifier (Logistic Regression + TF-IDF)
For unseen pairs, a trained classifier with:
- **TF-IDF features** (8,000 bigrams) on concatenated "CURRENT: X ||| PRIOR: Y" text
- **Structured features:** body part overlap (Jaccard), modality overlap (Jaccard), laterality flags (opposite side, same side, bilateral), age of prior, same description flag
- **Cross-validated accuracy: 92.7%** on the full public labeled set
- Threshold tuned to 0.35 (accounting for 23.8% base rate)

### Stage 3: LLM (Groq llama-3.3-70b-versatile)
Only fires for ML predictions in the uncertain zone (probability 0.25–0.55). Typical case sends 0–3 priors to the LLM out of 20-30 total. Well within 360s timeout.

### Stage 4: Caching
In-memory cache on (current_desc, prior_desc, prior_date) hash. Retries and repeated study pairs never re-run any computation.

## What Worked
- **ML classifier** — 92.7% CV accuracy, the biggest accuracy gain
- **Rich feature engineering** — body part Jaccard + modality Jaccard + laterality were the most predictive features
- **Empirical lookup** — 100% accuracy on high-frequency seen pairs
- **Uncertainty-based LLM routing** — only genuinely ambiguous cases hit the LLM
- **Laterality detection** — LT vs RT = false 98% of the time, caught by both ML and rules

## What Failed
- Pure LLM: inconsistent, slow on large cases
- TF-IDF alone (no structured features): 65% accuracy
- Aggressive false rules without ML: missed clinically related cross-modality pairs

## How I Would Improve It
1. **BiomedBERT/ClinicalBERT embeddings** instead of TF-IDF — semantic understanding of radiology abbreviations
2. **Gradient boosting** (XGBoost/LightGBM) on structured features — likely +2-3% over logistic regression
3. **Patient history context** — same patient having a known condition should weight prior exams of that condition higher
4. **Threshold optimization** — tune decision threshold per exam type using calibration curves
5. **Active learning** — use low-confidence predictions to prioritize LLM calls most effectively
