import os
import json
import hashlib
import pickle
import numpy as np
import scipy.sparse as sp
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()
cache = {}

_model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(_model_path, 'rb') as f:
    _model = pickle.load(f)
VECTORIZER = _model['vectorizer']
CLF = _model['clf']

_lookup_path = os.path.join(os.path.dirname(__file__), "lookup.json")
with open(_lookup_path) as f:
    _lookup_raw = json.load(f)
LOOKUP_TRUE  = {(a, b) for a, b in _lookup_raw["true"]}
LOOKUP_FALSE = {(a, b) for a, b in _lookup_raw["false"]}

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

def make_cache_key(current_study, prior_study):
    key = f"{current_study.get('study_description','')}|{prior_study.get('study_description','')}|{prior_study.get('study_date','')}"
    return hashlib.md5(key.encode()).hexdigest()

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
        cur_desc = current_study.get('study_description', '').upper()
        cur_date = current_study.get('study_date', '')
        texts, metas = [], []
        for p in need_ml:
            pri_desc = p.get('study_description','').upper()
            texts.append(f"CURRENT: {cur_desc} ||| PRIOR: {pri_desc}")
            metas.append(build_features(cur_desc, cur_date, pri_desc, p.get('study_date','')))
        X_tfidf = VECTORIZER.transform(texts)
        X_meta  = sp.csr_matrix(np.array(metas))
        X       = sp.hstack([X_tfidf, X_meta])
        probs   = CLF.predict_proba(X)[:, 1]
        preds   = [bool(p >= 0.35) for p in probs]
        for j, (orig_i, prior) in enumerate(zip(need_ml_idx, need_ml)):
            final[orig_i] = preds[j]
            cache[make_cache_key(current_study, prior)] = preds[j]

    return [r if r is not None else True for r in final]

@app.post("/predict")
async def predict(request: Request):
    try:
        body = await request.json()
        predictions = []
        for case in body.get("cases", []):
            case_id = case["case_id"]
            current_study = case["current_study"]
            prior_studies = case.get("prior_studies", [])
            try:
                relevances = predict_relevance_batch(current_study, prior_studies)
            except Exception as e:
                print(f"Error in case {case_id}: {e}")
                relevances = [True] * len(prior_studies)
            for prior, is_relevant in zip(prior_studies, relevances):
                predictions.append({"case_id": case_id, "study_id": prior["study_id"], "predicted_is_relevant": is_relevant})
        return JSONResponse(content={"predictions": predictions})
    except Exception as e:
        print(f"Fatal error: {e}")
        return JSONResponse(content={"predictions": []}, status_code=500)

@app.get("/health")
def health():
    return {"status": "ok", "model": "sklearn+lookup"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
