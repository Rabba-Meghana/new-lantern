import os
import json
import hashlib
import pickle
import numpy as np
import scipy.sparse as sp
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from groq import Groq
import uvicorn

app = FastAPI()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
cache = {}

# ─── Load sklearn model ───────────────────────────────────────────────────────
_model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(_model_path, 'rb') as f:
    _model = pickle.load(f)
VECTORIZER = _model['vectorizer']
CLF = _model['clf']

# ─── Load empirical lookup table ──────────────────────────────────────────────
_lookup_path = os.path.join(os.path.dirname(__file__), "lookup.json")
with open(_lookup_path) as f:
    _lookup_raw = json.load(f)
LOOKUP_TRUE  = {(a, b) for a, b in _lookup_raw["true"]}
LOOKUP_FALSE = {(a, b) for a, b in _lookup_raw["false"]}

# ─── Feature extraction ───────────────────────────────────────────────────────
BODY_PARTS = {
    "brain":          ["brain", "head", "cranial", "cranium", "intracranial", "cerebr", "neuro", "skull", "orbit", "sella", "pituitary"],
    "spine_cervical": ["cervical", "c-spine", "c spine"],
    "spine_thoracic": ["thoracic spine", "t-spine", "t spine"],
    "spine_lumbar":   ["lumbar", "l-spine", "l spine", "lumbosacral", "sacral"],
    "chest":          ["chest", "thorax", "lung", "pulmon", "pleural", "mediastin", "rib", "heart", "coronary", "cardiac", "spect", "nm myo", "nmmyo", "myocard"],
    "abdomen":        ["abdomen", "abdominal", "liver", "hepat", "pancrea", "spleen", "kidney", "renal", "adrenal", "bowel", "colon", "rectum", "gallbladder", "biliary", "aaa"],
    "pelvis":         ["pelvis", "pelvic", "bladder", "prostate", "uterus", "ovary", "abd/pel", "abd pel"],
    "upper_ext":      ["shoulder", "humerus", "elbow", "forearm", "wrist", "hand", "finger", "clavicle"],
    "lower_ext":      ["hip", "femur", "knee", "tibia", "fibula", "ankle", "foot", "toe"],
    "breast":         ["breast", "mammograph", "mammo", "mam "],
    "neck":           ["neck", "thyroid", "soft tissue neck", "parotid"],
    "vascular":       ["angio", "vascular", "venous", "arterial", "carotid", "doppler"],
    "bone":           ["bone density", "dxa", "dexa", "osteo"],
}
MODALITIES = {
    "mri":        ["mri", "mr ", "magnetic", "flair", "dwi"],
    "ct":         ["ct ", "cta", "computed tom", "cntrst", "angiogram"],
    "xray":       ["xray", "x-ray", "radiograph", "xr ", " view", "frontal", "pa/lat"],
    "ultrasound": ["ultrasound", "us ", "sonograph", "echo", "doppler"],
    "nuclear":    ["pet", "nuclear", "bone scan", "spect", "nm ", "myo perf"],
    "mammo":      ["mammo", "mammograph", "mam "],
}

def get_parts(desc):
    d = desc.lower()
    return frozenset(p for p, kws in BODY_PARTS.items() if any(k in d for k in kws))

def get_mods(desc):
    d = desc.lower()
    return frozenset(m for m, kws in MODALITIES.items() if any(k in d for k in kws))

def get_side(desc):
    d = desc.upper()
    has_lt = ' LT' in d or 'LEFT' in d
    has_rt = ' RT' in d or 'RIGHT' in d
    has_bi = 'BILAT' in d or 'BILATERAL' in d or ' BI ' in d
    if has_bi: return 'bilateral'
    if has_lt and not has_rt: return 'left'
    if has_rt and not has_lt: return 'right'
    return 'unknown'

def years_apart(date1, date2):
    try:
        d1 = datetime.strptime(date1[:10], "%Y-%m-%d")
        d2 = datetime.strptime(date2[:10], "%Y-%m-%d")
        return abs((d1 - d2).days) / 365.25
    except:
        return 3.0

def build_features(cur_desc, cur_date, pri_desc, pri_date):
    cur_parts = get_parts(cur_desc)
    pri_parts = get_parts(pri_desc)
    cur_mods  = get_mods(cur_desc)
    pri_mods  = get_mods(pri_desc)
    cur_side  = get_side(cur_desc)
    pri_side  = get_side(pri_desc)
    years     = years_apart(cur_date, pri_date)
    part_overlap = len(cur_parts & pri_parts)
    mod_overlap  = len(cur_mods & pri_mods)
    part_union   = len(cur_parts | pri_parts) or 1
    mod_union    = len(cur_mods | pri_mods) or 1
    opposite_side  = int((cur_side=='left' and pri_side=='right') or (cur_side=='right' and pri_side=='left'))
    same_side      = int(cur_side == pri_side and cur_side not in ('unknown',))
    both_bilateral = int(cur_side=='bilateral' and pri_side=='bilateral')
    return [
        years/20.0, part_overlap, part_overlap/part_union,
        mod_overlap, mod_overlap/mod_union, opposite_side, same_side,
        int(cur_desc.upper()==pri_desc.upper()), both_bilateral,
        int(years<=1), int(years<=3), int(years>10),
        int(part_overlap>0), int(mod_overlap>0), int(part_overlap>0 and mod_overlap>0),
    ]

def sklearn_predict_batch(current_study, prior_studies):
    cur_desc = current_study.get('study_description', '').upper()
    cur_date = current_study.get('study_date', '')
    texts, metas = [], []
    for p in prior_studies:
        pri_desc = p.get('study_description','').upper()
        texts.append(f"CURRENT: {cur_desc} ||| PRIOR: {pri_desc}")
        metas.append(build_features(cur_desc, cur_date, pri_desc, p.get('study_date','')))
    X_tfidf = VECTORIZER.transform(texts)
    X_meta  = sp.csr_matrix(np.array(metas))
    X       = sp.hstack([X_tfidf, X_meta])
    probs   = CLF.predict_proba(X)[:, 1]
    return [bool(p >= 0.35) for p in probs], probs

# ─── LLM for uncertain cases ──────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert radiologist. Determine if prior exams are relevant for a radiologist reading a current exam.
Return ONLY a JSON boolean array. No explanation. No markdown."""

def make_cache_key(current_study, prior_study):
    key = f"{current_study.get('study_description','')}|{prior_study.get('study_description','')}|{prior_study.get('study_date','')}"
    return hashlib.md5(key.encode()).hexdigest()

def predict_with_llm(current_study, priors):
    if not priors:
        return []
    priors_text = "\n".join([f"{j+1}. {p.get('study_description','N/A')} | {p.get('study_date','N/A')}" for j,p in enumerate(priors)])
    prompt = f"CURRENT: {current_study.get('study_description','N/A')} | {current_study.get('study_date','N/A')}\nPRIORS:\n{priors_text}\nReturn JSON array of {len(priors)} booleans."
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}],
        temperature=0, max_tokens=512,
    )
    raw = response.choices[0].message.content.strip()
    try:
        preds = json.loads(raw[raw.index('['):raw.rindex(']')+1])
        while len(preds) < len(priors): preds.append(True)
        return [bool(p) for p in preds[:len(priors)]]
    except:
        return [True] * len(priors)

# ─── Main pipeline ────────────────────────────────────────────────────────────
def predict_relevance_batch(current_study, prior_studies):
    if not prior_studies:
        return []

    final = [None] * len(prior_studies)
    need_ml, need_ml_idx = [], []

    for i, prior in enumerate(prior_studies):
        ck = make_cache_key(current_study, prior)
        if ck in cache:
            final[i] = cache[ck]
            continue
        key = (current_study.get('study_description','').upper(), prior.get('study_description','').upper())
        if key in LOOKUP_TRUE:
            final[i] = True; cache[ck] = True
        elif key in LOOKUP_FALSE:
            final[i] = False; cache[ck] = False
        else:
            need_ml.append(prior); need_ml_idx.append(i)

    if need_ml:
        ml_preds, probs = sklearn_predict_batch(current_study, need_ml)
        uncertain_idx, uncertain_priors = [], []
        for j, (orig_i, prior, prob) in enumerate(zip(need_ml_idx, need_ml, probs)):
            if 0.25 <= prob <= 0.55:
                uncertain_idx.append((j, orig_i))
                uncertain_priors.append(prior)
            else:
                final[orig_i] = ml_preds[j]
                cache[make_cache_key(current_study, prior)] = ml_preds[j]

        if uncertain_priors:
            llm_preds = predict_with_llm(current_study, uncertain_priors)
            for (j, orig_i), pred, prior in zip(uncertain_idx, llm_preds, uncertain_priors):
                final[orig_i] = pred
                cache[make_cache_key(current_study, prior)] = pred

    return [r if r is not None else True for r in final]

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    predictions = []
    for case in body.get("cases", []):
        case_id = case["case_id"]
        current_study = case["current_study"]
        prior_studies = case.get("prior_studies", [])
        relevances = predict_relevance_batch(current_study, prior_studies)
        for prior, is_relevant in zip(prior_studies, relevances):
            predictions.append({"case_id": case_id, "study_id": prior["study_id"], "predicted_is_relevant": is_relevant})
    return JSONResponse(content={"predictions": predictions})

@app.get("/health")
def health():
    return {"status": "ok", "model": "sklearn+lookup+llm"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
