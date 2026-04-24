# Experiments

## Baseline
- Rule-based matching on study_description keywords (body part + modality)
- Returned true if current and prior shared keywords like "BRAIN", "CHEST", "MRI", "CT"

## What Worked
- Switched to LLM-based (Groq llama3-70b) batch inference per case
- Sending all priors for a case in a single LLM call reduced latency significantly
- In-memory caching keyed on (current_description, prior_description, prior_date) avoids redundant LLM calls on retries
- Temperature=0 for deterministic outputs
- Prompting the model to return a simple JSON boolean array made parsing reliable

## What Failed
- One LLM call per prior study: too slow, hit timeout on large cases
- GPT-style verbose prompts: model would return explanations instead of clean JSON
- Trusting raw model output without stripping/parsing: broke on edge cases

## How I Would Improve It
- Fine-tune on the labeled public eval data for domain-specific accuracy
- Add body-part extraction as a structured pre-processing step before LLM call
- Use study dates more aggressively: heavily penalize priors older than 5 years unless same rare condition
- Ensemble: combine LLM judgment with rule-based modality/body-part matching
- Use embeddings to find semantically similar study descriptions as a fast pre-filter
- Add confidence scores and threshold tuning on the public eval set
