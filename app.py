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

# ── Load sklearn model (fallback) ─────────────────────────────────────────────
_model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(_model_path, "rb") as f:
    _model = pickle.load(f)
VECTORIZER = _model["vectorizer"]
CLF        = _model["clf"]
logger.info("sklearn model loaded OK")

# ── Load ONNX BiomedBERT (primary) ────────────────────────────────────────────
ONNX_MODEL     = None
ONNX_TOKENIZER = None
ONNX_LOADED    = False

try:
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer
    _onnx_dir = os.path.join(os.path.dirname(__file__), "onnx_int8")
    ONNX_TOKENIZER = AutoTokenizer.from_pretrained(_onnx_dir)
    ONNX_MODEL     = ORTModelForSequenceClassification.from_pretrained(_onnx_dir)
    ONNX_LOADED    = True
    logger.info("ONNX BiomedBERT loaded OK")
except Exception as e:
    logger.warning("ONNX load failed (%s) — using sklearn only", e)

# ── Load empirical lookup table ───────────────────────────────────────────────
_lookup_path = os.path.join(os.path.dirname(__file__), "lookup.json")
with open(_lookup_path) as f:
    _lookup_raw = json.load(f)
LOOKUP_TRUE  = {(a, b) for a, b in _lookup_raw["true"]}
LOOKUP_FALSE = {(a, b) for a, b in _lookup_raw["false"]}
logger.info("Lookup: %d true, %d false pairs", len(LOOKUP_TRUE), len(LOOKUP_FALSE))

# ── Keyword dictionaries ──────────────────────────────────────────────────────
BODY_PARTS = {
    "brain":          ["brain","head","cranial","cranium","intracranial","cerebr",
                       "neuro","skull","orbit","sella","pituitary"],
    "spine_cervical": ["cervical","c-spine","c spine"],
    "spine_thoracic": ["thoracic spine","t-spine","t spine"],
    "spine_lumbar":   ["lumbar","l-spine","l spine","lumbosacral","sacral"],
    "chest":          ["chest","thorax","lung","pulmon","pleural","mediastin",
                       "rib","heart","coronary","cardiac","spect","nm myo",
                       "nmmyo","myocard"],
    "abdomen":        ["abdomen","abdominal","liver","hepat","pancrea","spleen",
                       "kidney","renal","adrenal","bowel","colon","rectum",
                       "gallbladder","biliary","aaa"],
    "pelvis":         ["pelvis","pelvic","bladder","prostate","uterus","ovary",
                       "abd/pel","abd pel"],
    "upper_ext":      ["shoulder","humerus","elbow","forearm","wrist","hand",
                       "finger","clavicle"],
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

def _get_parts(desc):
    d = desc.lower()
    return frozenset(p for p, kws in BODY_PARTS.items() if any(k in d for k in kws))

def _get_mods(desc):
    d = desc.lower()
    return frozenset(m for m, kws in MODALITIES.items() if any(k in d for k in kws))

def _get_side(desc):
    d = desc.upper()
    if "BILAT" in d or "BILATERAL" in d: return "bilateral"
    if (" LT" in d or "LEFT" in d) and not (" RT" in d or "RIGHT" in d): return "left"
    if (" RT" in d or "RIGHT" in d) and not (" LT" in d or "LEFT" in d): return "right"
    return "unknown"

def _years_apart(d1, d2):
    try:
        t1 = datetime.strptime(d1[:10], "%Y-%m-%d")
        t2 = datetime.strptime(d2[:10], "%Y-%m-%d")
        return abs((t1 - t2).days) / 365.25
    except (ValueError, TypeError):
        return 3.0

def _safe_desc(study): return str(study.get("study_description") or "").upper()
def _safe_date(study): return str(study.get("study_date") or "")
def _cache_key(cd, pd, pdate): return hashlib.md5(f"{cd}|{pd}|{pdate}".encode()).hexdigest()

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

# ── Targeted rules ────────────────────────────────────────────────────────────
def _is_mam(d):     d=d.lower(); return any(k in d for k in ["mam ","mammo","mammograph","breast"])
def _is_cxr(d):     d=d.lower(); return any(k in d for k in ["chest","cxr","frontal","pa/lat","thorax"]) and not _is_mam(d)
def _is_ctchest(d): d=d.lower(); return "ct " in d and "chest" in d and "abd" not in d and "pelv" not in d
def _is_ctabd(d):   d=d.lower(); return "ct " in d and any(k in d for k in ["abdomen","pelvis","abd/pel","abd pel"])
def _is_mam_bi(d):  d=d.lower(); return _is_mam(d) and any(k in d for k in ["bilat","bilateral"," bi ","screen","3d"])
def _is_mam_uni(d): d=d.lower(); return _is_mam(d) and not any(k in d for k in ["bilat","bilateral"," bi "]) and any(k in d for k in [" lt"," rt","left","right"])

def _targeted_rule(cur_desc, pri_desc):
    if _is_mam(cur_desc) and _is_cxr(pri_desc): return False
    if _is_cxr(cur_desc) and _is_mam(pri_desc): return False
    if _is_ctchest(cur_desc) and _is_ctabd(pri_desc): return False
    if _is_ctabd(cur_desc) and _is_ctchest(pri_desc): return False
    if _is_mam_bi(cur_desc) and _is_mam_uni(pri_desc): return True
    if _is_mam_uni(cur_desc) and _is_mam_bi(pri_desc): return True
    return None

# ── ONNX inference ────────────────────────────────────────────────────────────
def _onnx_predict(cur_desc, prior_descs, threshold=0.35):
    texts = [
        f"Current exam: {cur_desc}. Prior exam: {pd}."
        for pd in prior_descs
    ]
    all_probs = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = ONNX_TOKENIZER(
            batch, truncation=True, padding=True,
            max_length=128, return_tensors="pt"
        )
        import torch
        with torch.no_grad():
            logits = ONNX_MODEL(**enc).logits
            probs  = torch.softmax(logits, dim=-1)[:, 1].numpy()
        all_probs.extend(probs.tolist())
    return [bool(p >= threshold) for p in all_probs]

# ── sklearn inference ─────────────────────────────────────────────────────────
def _sklearn_predict(cur_desc, cur_date, need_ml):
    texts = [f"CURRENT: {cur_desc} ||| PRIOR: {pd}" for _, pd, _, _ in need_ml]
    metas = [_build_features(cur_desc, cur_date, pd, pdate) for _, pd, pdate, _ in need_ml]
    try:
        X = sp.hstack([VECTORIZER.transform(texts),
                       sp.csr_matrix(np.array(metas, dtype=float))])
        probs = CLF.predict_proba(X)[:, 1]
        return [bool(p >= 0.35) for p in probs]
    except Exception as exc:
        logger.error("sklearn failed: %s", exc)
        cur_parts = _get_parts(cur_desc)
        return [bool(cur_parts & _get_parts(pd)) for _, pd, _, _ in need_ml]

# ── Core pipeline ─────────────────────────────────────────────────────────────
def _predict_batch(current_study, prior_studies):
    cur_desc = _safe_desc(current_study)
    cur_date = _safe_date(current_study)
    final    = [None] * len(prior_studies)
    need_ml  = []

    for i, prior in enumerate(prior_studies):
        pri_desc = _safe_desc(prior)
        pri_date = _safe_date(prior)
        ck       = _cache_key(cur_desc, pri_desc, pri_date)

        if ck in cache:
            final[i] = cache[ck]; continue

        key = (cur_desc, pri_desc)
        if key in LOOKUP_TRUE:
            final[i] = True;  cache[ck] = True;  continue
        if key in LOOKUP_FALSE:
            final[i] = False; cache[ck] = False; continue

        rule = _targeted_rule(cur_desc, pri_desc)
        if rule is not None:
            final[i] = rule; cache[ck] = rule; continue

        need_ml.append((i, pri_desc, pri_date, ck))

    if need_ml:
        pri_descs = [pd for _, pd, _, _ in need_ml]

        if ONNX_LOADED:
            try:
                preds = _onnx_predict(cur_desc, pri_descs)
            except Exception as exc:
                logger.error("ONNX failed: %s — falling back to sklearn", exc)
                preds = _sklearn_predict(cur_desc, cur_date, need_ml)
        else:
            preds = _sklearn_predict(cur_desc, cur_date, need_ml)

        for j, (orig_i, _, _, ck) in enumerate(need_ml):
            final[orig_i] = preds[j]
            cache[ck]     = preds[j]

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
        if not isinstance(prior_studies, list):
            prior_studies = []

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
    return {
        "status":     "ok",
        "model":      "onnx_biomedbert" if ONNX_LOADED else "sklearn",
        "cache_size": len(cache),
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
