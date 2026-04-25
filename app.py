import os
import json
import hashlib
import pickle
import logging
import numpy as np
import scipy.sparse as sp
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
cache = {}

# ── Load sklearn model ────────────────────────────────────────────────────────
_model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(_model_path, "rb") as f:
    _model = pickle.load(f)
VECTORIZER = _model["vectorizer"]
CLF        = _model["clf"]
logger.info("Model loaded OK")

# ── Load empirical lookup table ───────────────────────────────────────────────
_lookup_path = os.path.join(os.path.dirname(__file__), "lookup.json")
with open(_lookup_path) as f:
    _lookup_raw = json.load(f)
LOOKUP_TRUE  = {(a, b) for a, b in _lookup_raw["true"]}
LOOKUP_FALSE = {(a, b) for a, b in _lookup_raw["false"]}
logger.info("Lookup: %d true, %d false pairs", len(LOOKUP_TRUE), len(LOOKUP_FALSE))

# ── Feature extraction ────────────────────────────────────────────────────────
BODY_PARTS = {
    "brain":          ["brain","head","cranial","cranium","intracranial","cerebr","neuro","skull","orbit","sella","pituitary"],
    "spine_cervical": ["cervical","c-spine","c spine"],
    "spine_thoracic": ["thoracic spine","t-spine","t spine"],
    "spine_lumbar":   ["lumbar","l-spine","l spine","lumbosacral","sacral"],
    "chest":          ["chest","thorax","lung","pulmon","pleural","mediastin","rib","heart","coronary","cardiac","spect","nm myo","nmmyo","myocard"],
    "abdomen":        ["abdomen","abdominal","liver","hepat","pancrea","spleen","kidney","renal","adrenal","bowel","colon","rectum","gallbladder","biliary","aaa"],
    "pelvis":         ["pelvis","pelvic","bladder","prostate","uterus","ovary","abd/pel","abd pel"],
    "upper_ext":      ["shoulder","humerus","elbow","forearm","wrist","hand","finger","clavicle"],
    "lower_ext":      ["hip","femur","knee","tibia","fibula","ankle","foot","toe"],
    "breast":         ["breast","mammograph","mammo","mam "],
    "neck":           ["neck","thyroid","soft tissue neck","parotid"],
    "vascular":       ["angio","vascular","venous","arterial","carotid","doppler"],
    "bone":           ["bone density","dxa","dexa","osteo"],
}
MODALITIES = {
    "mri":        ["mri","mr ","magnetic","flair","dwi"],
    "ct":         ["ct ","cta","computed tom","cntrst","angiogram"],
    "xray":       ["xray","x-ray","radiograph","xr "," view","frontal","pa/lat"],
    "ultrasound": ["ultrasound","us ","sonograph","echo","doppler"],
    "nuclear":    ["pet","nuclear","bone scan","spect","nm ","myo perf"],
    "mammo":      ["mammo","mammograph","mam "],
}

def _get_parts(desc): d=desc.lower(); return frozenset(p for p,kws in BODY_PARTS.items() if any(k in d for k in kws))
def _get_mods(desc):  d=desc.lower(); return frozenset(m for m,kws in MODALITIES.items() if any(k in d for k in kws))

def _get_side(desc):
    d=desc.upper()
    if "BILAT" in d or "BILATERAL" in d: return "bilateral"
    if (" LT" in d or "LEFT" in d) and not (" RT" in d or "RIGHT" in d): return "left"
    if (" RT" in d or "RIGHT" in d) and not (" LT" in d or "LEFT" in d): return "right"
    return "unknown"

def _years_apart(d1, d2):
    try:
        return abs((datetime.strptime(d1[:10],"%Y-%m-%d")-datetime.strptime(d2[:10],"%Y-%m-%d")).days)/365.25
    except (ValueError, TypeError):
        return 3.0

def _is_mam(desc):    d=desc.lower(); return any(k in d for k in ["mam ","mammo","mammograph","breast"])
def _is_cxr(desc):    d=desc.lower(); return any(k in d for k in ["chest","cxr","frontal","pa/lat","thorax"]) and not _is_mam(desc)
def _is_echo(desc):   d=desc.lower(); return any(k in d for k in ["echo","tte","tee","transesophag","transthorac"])
def _is_ctchest(desc):d=desc.lower(); return "ct " in d and "chest" in d and "abd" not in d and "pelv" not in d
def _is_ctabd(desc):  d=desc.lower(); return "ct " in d and ("abdomen" in d or "pelvis" in d or "abd" in d or "abd/pel" in d)
def _is_mam_bilateral(desc): d=desc.lower(); return _is_mam(desc) and ("bilat" in d or " bi " in d or "screen" in d or "3d" in d)
def _is_mam_unilateral(desc):
    d=desc.lower()
    return _is_mam(desc) and not ("bilat" in d or " bi " in d) and (" lt" in d or " rt" in d or "left" in d or "right" in d)

# ── High-confidence targeted rules from data analysis ────────────────────────
def _targeted_rule(cur_desc, pri_desc):
    """
    Returns True, False, or None (= let ML decide).
    Rules are derived from public-split analysis with >95% confidence.
    """
    # Mammography vs chest X-ray: 0% relevant (643 false, 3 true)
    if _is_mam(cur_desc) and _is_cxr(pri_desc): return False
    if _is_cxr(cur_desc) and _is_mam(pri_desc): return False

    # CT Chest vs CT Abdomen/Pelvis: 2% relevant (5 true, 319 false)
    if _is_ctchest(cur_desc) and _is_ctabd(pri_desc): return False
    if _is_ctabd(cur_desc) and _is_ctchest(pri_desc): return False

    # Mammography bilateral (screening/3D) vs unilateral prior: 91% relevant
    if _is_mam_bilateral(cur_desc) and _is_mam_unilateral(pri_desc): return True
    if _is_mam_unilateral(cur_desc) and _is_mam_bilateral(pri_desc): return True

    return None  # ML decides

def _build_features(cur_desc, cur_date, pri_desc, pri_date):
    cp=_get_parts(cur_desc); pp=_get_parts(pri_desc)
    cm=_get_mods(cur_desc);  pm=_get_mods(pri_desc)
    cs=_get_side(cur_desc);  ps=_get_side(pri_desc)
    years=_years_apart(cur_date, pri_date)
    po=len(cp&pp); mo=len(cm&pm); pu=len(cp|pp) or 1; mu=len(cm|pm) or 1
    opp=int((cs=="left" and ps=="right") or (cs=="right" and ps=="left"))
    same=int(cs==ps and cs!="unknown"); bi=int(cs=="bilateral" and ps=="bilateral")
    return [years/20, po, po/pu, mo, mo/mu, opp, same,
            int(cur_desc==pri_desc), bi,
            int(years<=1), int(years<=3), int(years>10),
            int(po>0), int(mo>0), int(po>0 and mo>0)]

def _safe_desc(study): return str(study.get("study_description") or "").upper()
def _safe_date(study): return str(study.get("study_date") or "")

def _cache_key(cd, pd, pdate):
    return hashlib.md5(f"{cd}|{pd}|{pdate}".encode()).hexdigest()

def _predict_batch(current_study, prior_studies):
    cur_desc = _safe_desc(current_study)
    cur_date = _safe_date(current_study)
    final, need_ml = [None]*len(prior_studies), []

    for i, prior in enumerate(prior_studies):
        pri_desc = _safe_desc(prior)
        pri_date = _safe_date(prior)
        ck = _cache_key(cur_desc, pri_desc, pri_date)

        if ck in cache:
            final[i] = cache[ck]; continue

        key = (cur_desc, pri_desc)

        # Stage 1: lookup table
        if key in LOOKUP_TRUE:
            final[i] = True; cache[ck] = True; continue
        if key in LOOKUP_FALSE:
            final[i] = False; cache[ck] = False; continue

        # Stage 2: targeted rules
        rule = _targeted_rule(cur_desc, pri_desc)
        if rule is not None:
            final[i] = rule; cache[ck] = rule; continue

        need_ml.append((i, pri_desc, pri_date, ck))

    # Stage 3: sklearn batch
    if need_ml:
        texts = [f"CURRENT: {cur_desc} ||| PRIOR: {pd}" for _,pd,_,_ in need_ml]
        metas = [_build_features(cur_desc, cur_date, pd, pdate) for _,pd,pdate,_ in need_ml]
        try:
            X = sp.hstack([VECTORIZER.transform(texts), sp.csr_matrix(np.array(metas, dtype=float))])
            probs = CLF.predict_proba(X)[:, 1]
            preds = [bool(p >= 0.35) for p in probs]
        except Exception as exc:
            logger.error("sklearn failed: %s", exc)
            cur_parts = _get_parts(cur_desc)
            preds = [bool(cur_parts & _get_parts(pd)) for _,pd,_,_ in need_ml]

        for j, (orig_i, _, _, ck) in enumerate(need_ml):
            final[orig_i] = preds[j]; cache[ck] = preds[j]

    return [bool(r) if r is not None else False for r in final]

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    cases = body.get("cases") or []
    if not isinstance(cases, list):
        raise HTTPException(status_code=400, detail="'cases' must be a list")

    logger.info("Request: %d cases", len(cases))
    predictions = []

    for case in cases:
        case_id       = case.get("case_id", "")
        current_study = case.get("current_study") or {}
        prior_studies = case.get("prior_studies") or []
        if not isinstance(prior_studies, list): prior_studies = []

        logger.info("  case=%s priors=%d", case_id, len(prior_studies))
        relevances = _predict_batch(current_study, prior_studies)

        for prior, is_relevant in zip(prior_studies, relevances):
            predictions.append({
                "case_id":               case_id,
                "study_id":              prior.get("study_id", ""),
                "predicted_is_relevant": is_relevant,
            })

    logger.info("Response: %d predictions", len(predictions))
    return JSONResponse(content={"predictions": predictions})

@app.get("/health")
def health():
    return {"status": "ok", "model": "sklearn+lookup+rules", "cache_size": len(cache)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
