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

# ─── Load empirical lookup table from real labeled data ───────────────────────
_lookup_path = os.path.join(os.path.dirname(__file__), "lookup.json")
with open(_lookup_path) as f:
    _lookup_raw = json.load(f)

LOOKUP_TRUE  = {(a, b) for a, b in _lookup_raw["true"]}
LOOKUP_FALSE = {(a, b) for a, b in _lookup_raw["false"]}

# ─── Body part + modality extraction ─────────────────────────────────────────

BODY_PARTS = {
    "brain":            ["brain", "head", "cranial", "cranium", "intracranial", "cerebr", "neuro", "skull", "orbit", "sella", "pituitary"],
    "spine_cervical":   ["cervical", "c-spine", "c spine"],
    "spine_thoracic":   ["thoracic spine", "t-spine", "t spine"],
    "spine_lumbar":     ["lumbar", "l-spine", "l spine", "lumbosacral", "sacral"],
    "chest":            ["chest", "thorax", "lung", "pulmon", "pleural", "mediastin", "rib", "heart", "coronary", "cardiac", "aorta", "spect", "nm myo", "nmmyo", "myocard"],
    "abdomen":          ["abdomen", "abdominal", "liver", "hepat", "pancrea", "spleen", "kidney", "renal", "adrenal", "bowel", "colon", "rectum", "gallbladder", "biliary", "aaa"],
    "pelvis":           ["pelvis", "pelvic", "bladder", "prostate", "uterus", "ovary", "abd/pel", "abd pel", "abdom/pel"],
    "extremity_upper":  ["shoulder", "humerus", "elbow", "forearm", "wrist", "hand", "finger", "upper extremity", "clavicle"],
    "extremity_lower":  ["hip", "femur", "knee", "tibia", "fibula", "ankle", "foot", "toe", "lower extremity"],
    "breast":           ["breast", "mammograph", "mammo", "mam "],
    "neck":             ["neck", "thyroid", "soft tissue neck", "parotid", "salivary"],
    "vascular":         ["angio", "vascular", "venous", "arterial", "carotid", "doppler"],
    "bone":             ["bone density", "dxa", "dexa", "osteo"],
    "whole_body":       ["whole body", "pet", "bone scan", "nuclear", "spect"],
}

MODALITIES = {
    "mri":        ["mri", "mr ", "magnetic", "flair", "dwi", "dti", "mrcp", "mra"],
    "ct":         ["ct ", "cta", "computed tom", "cntrst", "angiogram"],
    "xray":       ["xray", "x-ray", "radiograph", "plain film", "cxr", "xr ", " view", "frontal", "pa/lat", "pa lat"],
    "ultrasound": ["ultrasound", "us ", " us ", "sonograph", "echo", "doppler"],
    "pet":        ["pet", "nuclear", "bone scan", "spect", "nm "],
    "mammo":      ["mammo", "mammograph", "mam "],
    "nuclear":    ["nm ", "nuclear", "spect", "myocard", "myo perf"],
}

def extract_body_parts(desc: str) -> set:
    d = desc.lower()
    return {part for part, kws in BODY_PARTS.items() if any(k in d for k in kws)}

def extract_modality(desc: str) -> set:
    d = desc.lower()
    return {mod for mod, kws in MODALITIES.items() if any(k in d for k in kws)}

def extract_laterality(desc: str):
    d = desc.upper()
    has_lt = ' LT' in d or 'LEFT' in d or ' LT ' in d
    has_rt = ' RT' in d or 'RIGHT' in d or ' RT ' in d
    has_bi = 'BILAT' in d or 'BILATERAL' in d or ' BI ' in d or d.endswith(' BI')
    if has_bi:
        return 'bilateral'
    if has_lt and not has_rt:
        return 'left'
    if has_rt and not has_lt:
        return 'right'
    return 'unknown'

def years_apart(date1: str, date2: str) -> float:
    try:
        d1 = datetime.strptime(date1[:10], "%Y-%m-%d")
        d2 = datetime.strptime(date2[:10], "%Y-%m-%d")
        return abs((d1 - d2).days) / 365.25
    except Exception:
        return 0.0

# ─── Three-stage pipeline ─────────────────────────────────────────────────────

def lookup_decision(cur_desc: str, pri_desc: str) -> str:
    """Stage 1: exact empirical lookup from labeled data."""
    key = (cur_desc.upper(), pri_desc.upper())
    if key in LOOKUP_TRUE:
        return "true"
    if key in LOOKUP_FALSE:
        return "false"
    return "miss"

def rule_based_decision(current: dict, prior: dict) -> str:
    """Stage 2: clinical rules."""
    cur_desc = current.get("study_description", "")
    pri_desc = prior.get("study_description", "")
    cur_date = current.get("study_date", "")
    pri_date = prior.get("study_date", "")

    cur_parts = extract_body_parts(cur_desc)
    pri_parts = extract_body_parts(pri_desc)
    cur_mods  = extract_modality(cur_desc)
    pri_mods  = extract_modality(pri_desc)
    cur_side  = extract_laterality(cur_desc)
    pri_side  = extract_laterality(pri_desc)
    years     = years_apart(cur_date, pri_date)

    # Whole body / PET / nuclear: let LLM handle
    if "whole_body" in cur_parts or "whole_body" in pri_parts:
        return "llm"

    # Opposite laterality (left vs right, non-bilateral) = almost always false
    sides = {cur_side, pri_side}
    if 'left' in sides and 'right' in sides:
        # 98% false from data - call it false unless bilateral
        if cur_side != 'bilateral' and pri_side != 'bilateral':
            return "false"

    # Completely different body parts = not relevant
    if cur_parts and pri_parts and cur_parts.isdisjoint(pri_parts):
        return "false"

    # Same body part + same modality within 10 years = relevant
    if cur_parts & pri_parts and cur_mods & pri_mods and years <= 10:
        return "true"

    # Same body part + any modality within 5 years = relevant
    if cur_parts & pri_parts and years <= 5:
        return "true"

    # Same body part, older or unclear = LLM
    if cur_parts & pri_parts:
        return "llm"

    return "llm"

# ─── LLM stage ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert radiologist with 20+ years of experience. Determine which prior radiology exams are relevant for a radiologist to review when reading a current exam.

RELEVANCE RULES:
1. Same body part + same modality → relevant (direct comparison baseline)
2. Same body part + different modality → usually relevant (complementary tissue info)
3. Clinically related exams across body parts → relevant (e.g., cardiac SPECT + coronary CT, cancer staging across regions)
4. Opposite laterality (left vs right of same structure) → NOT relevant unless bilateral disease
5. Completely different body part with no clinical link → NOT relevant
6. Very old exams (>15 years) with different body part → NOT relevant
7. Bone density / DXA exams are only relevant to other DXA or bone-related exams

IMPORTANT PATTERNS FROM REAL DATA:
- Chest CT and chest X-ray: RELEVANT to each other
- Cardiac SPECT (NM MYO PERF) and coronary CT/CTA: RELEVANT
- Mammography bilateral: relevant to all prior mammography regardless of age
- Mammography RIGHT only: NOT relevant to LEFT-only mammography
- CT Chest and CT Abdomen/Pelvis: NOT relevant (different body regions)
- CT Head/Brain and Chest X-ray: NOT relevant

Return ONLY a JSON boolean array. No explanation. No markdown."""

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

Return a JSON array of exactly {len(priors)} booleans (true=relevant, false=not relevant).
ONLY the array."""

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
        while len(predictions) < len(priors):
            predictions.append(True)
        return [bool(p) for p in predictions[:len(priors)]]
    except Exception:
        # Fallback: body part overlap heuristic
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

        # Stage 1: empirical lookup
        decision = lookup_decision(
            current_study.get("study_description", ""),
            prior.get("study_description", "")
        )

        # Stage 2: rules (if lookup missed)
        if decision == "miss":
            decision = rule_based_decision(current_study, prior)

        if decision == "true":
            final_results[i] = True
            cache[ck] = True
        elif decision == "false":
            final_results[i] = False
            cache[ck] = False
        else:
            llm_indices.append(i)
            llm_priors.append(prior)

    # Stage 3: LLM for remaining ambiguous cases
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
