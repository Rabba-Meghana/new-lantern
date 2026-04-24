import os
import json
import hashlib
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from groq import Groq
import uvicorn

app = FastAPI()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
cache = {}

# ─── Body part + modality extraction ─────────────────────────────────────────

BODY_PARTS = {
    "brain": ["brain", "head", "cranial", "cranium", "intracranial", "cerebr", "neuro", "skull", "orbit", "sella", "pituitary"],
    "spine_cervical": ["cervical", "c-spine", "c spine", "neck"],
    "spine_thoracic": ["thoracic", "t-spine", "t spine"],
    "spine_lumbar": ["lumbar", "l-spine", "l spine", "lumbosacral", "sacral"],
    "chest": ["chest", "thorax", "lung", "pulmon", "pleural", "mediastin", "rib", "cardiac", "heart", "coronary"],
    "abdomen": ["abdomen", "abdominal", "liver", "hepat", "pancrea", "spleen", "kidney", "renal", "adrenal", "bowel", "colon", "rectum", "gallbladder", "biliary"],
    "pelvis": ["pelvis", "pelvic", "bladder", "prostate", "uterus", "ovary"],
    "extremity_upper": ["shoulder", "humerus", "elbow", "forearm", "wrist", "hand", "finger", "upper extremity", "upper ext"],
    "extremity_lower": ["hip", "femur", "knee", "tibia", "fibula", "ankle", "foot", "toe", "lower extremity", "lower ext"],
    "breast": ["breast", "mammogram", "mammo"],
    "vascular": ["angio", "vascular", "venous", "arterial", "carotid"],
    "whole_body": ["whole body", "pet", "bone scan", "nuclear", "spect"],
}

MODALITIES = {
    "mri": ["mri", "mr ", "magnetic", "flair", "dwi", "dti", "mrcp", "mra"],
    "ct": ["ct ", "cta", "computed tom", "cntrst", "contrast"],
    "xray": ["xray", "x-ray", "radiograph", "plain film", "cxr"],
    "ultrasound": ["ultrasound", "us ", "sonograph", "echo", "doppler"],
    "pet": ["pet", "nuclear", "bone scan", "spect"],
    "mammo": ["mammo", "breast"],
}

def extract_body_parts(desc: str) -> set:
    d = desc.lower()
    return {part for part, kws in BODY_PARTS.items() if any(k in d for k in kws)}

def extract_modality(desc: str) -> set:
    d = desc.lower()
    return {mod for mod, kws in MODALITIES.items() if any(k in d for k in kws)}

def years_apart(date1: str, date2: str) -> float:
    try:
        d1 = datetime.strptime(date1[:10], "%Y-%m-%d")
        d2 = datetime.strptime(date2[:10], "%Y-%m-%d")
        return abs((d1 - d2).days) / 365.25
    except Exception:
        return 0.0

def rule_based_prefilter(current: dict, prior: dict) -> str:
    cur_desc = current.get("study_description", "")
    pri_desc = prior.get("study_description", "")
    cur_date = current.get("study_date", "")
    pri_date = prior.get("study_date", "")

    cur_parts = extract_body_parts(cur_desc)
    pri_parts = extract_body_parts(pri_desc)
    cur_mods = extract_modality(cur_desc)
    pri_mods = extract_modality(pri_desc)
    years = years_apart(cur_date, pri_date)

    # Whole body exams are always potentially relevant
    if "whole_body" in cur_parts or "whole_body" in pri_parts:
        return "llm"

    # Completely different body parts = not relevant
    if cur_parts and pri_parts and cur_parts.isdisjoint(pri_parts):
        return "false"

    # Same body part + same modality within 10 years = relevant
    if cur_parts & pri_parts and cur_mods & pri_mods and years <= 10:
        return "true"

    # Same body part + any modality within 5 years = relevant
    if cur_parts & pri_parts and years <= 5:
        return "true"

    # Same body part but old or unknown date = let LLM decide
    if cur_parts & pri_parts:
        return "llm"

    return "llm"

# ─── LLM prediction ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert radiologist AI with 20+ years of clinical experience. Determine which prior radiology exams are relevant for a radiologist reading a current exam.

RELEVANCE CRITERIA:
1. Same body part + same modality → relevant (direct comparison)
2. Same body part + different modality → usually relevant (complementary info)
3. Clinically related conditions across body parts → relevant (e.g., metastasis workup)
4. Exams showing same known pathology → relevant
5. Completely unrelated body part with no clinical link → not relevant
6. Exams >10 years old with no body part overlap → usually not relevant

Return ONLY a JSON boolean array. No explanation. No markdown. Just the array like [true, false, true]."""

def make_cache_key(current_study: dict, prior_study: dict) -> str:
    key = f"{current_study.get('study_description','')}|{prior_study.get('study_description','')}|{prior_study.get('study_date','')}"
    return hashlib.md5(key.encode()).hexdigest()

def predict_with_llm(current_study: dict, priors: list) -> list:
    if not priors:
        return []

    priors_text = "\n".join([
        f"{j+1}. desc={p.get('study_description','N/A')} | date={p.get('study_date','N/A')}"
        for j, p in enumerate(priors)
    ])

    prompt = f"""CURRENT EXAM:
- Description: {current_study.get('study_description', 'N/A')}
- Date: {current_study.get('study_date', 'N/A')}

PRIOR EXAMS ({len(priors)} total):
{priors_text}

Return a JSON array of {len(priors)} booleans (true=relevant, false=not relevant), one per prior in order.
ONLY the array, nothing else."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=512,
    )

    raw = response.choices[0].message.content.strip()
    try:
        start = raw.index('[')
        end = raw.rindex(']') + 1
        predictions = json.loads(raw[start:end])
        # Pad or trim to correct length
        while len(predictions) < len(priors):
            predictions.append(True)
        return [bool(p) for p in predictions[:len(priors)]]
    except Exception:
        # Fallback: body part overlap
        cur_parts = extract_body_parts(current_study.get("study_description", ""))
        return [bool(cur_parts & extract_body_parts(p.get("study_description", ""))) for p in priors]

def predict_relevance_batch(current_study: dict, prior_studies: list) -> list:
    final_results = [None] * len(prior_studies)
    llm_indices = []
    llm_priors = []

    for i, prior in enumerate(prior_studies):
        ck = make_cache_key(current_study, prior)
        if ck in cache:
            final_results[i] = cache[ck]
            continue

        decision = rule_based_prefilter(current_study, prior)
        if decision == "true":
            final_results[i] = True
            cache[ck] = True
        elif decision == "false":
            final_results[i] = False
            cache[ck] = False
        else:
            llm_indices.append(i)
            llm_priors.append(prior)

    if llm_priors:
        llm_preds = predict_with_llm(current_study, llm_priors)
        for idx, pred in zip(llm_indices, llm_preds):
            final_results[idx] = pred
            cache[make_cache_key(current_study, prior_studies[idx])] = pred

    return [r if r is not None else True for r in final_results]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    cases = body.get("cases", [])
    predictions = []

    for case in cases:
        case_id = case["case_id"]
        current_study = case["current_study"]
        prior_studies = case.get("prior_studies", [])
        relevances = predict_relevance_batch(current_study, prior_studies)

        for prior, is_relevant in zip(prior_studies, relevances):
            predictions.append({
                "case_id": case_id,
                "study_id": prior["study_id"],
                "predicted_is_relevant": is_relevant
            })

    return JSONResponse(content={"predictions": predictions})


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
