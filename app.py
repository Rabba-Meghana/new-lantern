import os
import json
import hashlib
import pickle
import logging
import threading
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

# ── sklearn (loads instantly, always available) ───────────────────────────────
_model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(_model_path, "rb") as f:
    _model = pickle.load(f)
VECTORIZER = _model["vectorizer"]
CLF        = _model["clf"]
logger.info("sklearn loaded OK")

# ── Lookup table ──────────────────────────────────────────────────────────────
_lookup_path = os.path.join(os.path.dirname(__file__), "lookup.json")
with open(_lookup_path) as f:
    _lookup_raw = json.load(f)
LOOKUP_TRUE  = {(a, b) for a, b in _lookup_raw["true"]}
LOOKUP_FALSE = {(a, b) for a, b in _lookup_raw["false"]}
logger.info("Lookup: %d true, %d false", len(LOOKUP_TRUE), len(LOOKUP_FALSE))

# ── ONNX loads in background thread ──────────────────────────────────────────
ONNX_MODEL     = None
ONNX_TOKENIZER = None
ONNX_READY     = False

def _load_onnx_background():
    global ONNX_MODEL, ONNX_TOKENIZER, ONNX_READY
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
        _dir = os.path.join(os.path.dirname(__file__), "onnx_int8")
        ONNX_TOKENIZER = AutoTokenizer.from_pretrained(_dir)
        ONNX_MODEL     = ORTModelForSequenceClassification.from_pretrained(
            _dir, file_name="model_quantized.onnx"
        )
        ONNX_READY = True
        logger.info("ONNX BiomedBERT ready!")
    except Exception as e:
        logger.warning("ONNX load failed: %s — sklearn only", e)

threading.Thread(target=_load_onnx_background, daemon=True).start()

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

def _is_mam(d):     dl=d.lower(); return any(k in dl for k in ["mam ","mammo","mammograph","breast"])
def _is_cxr(d):     dl=d.lower(); return any(k in dl for k in ["chest","cxr","frontal","pa/lat","thorax"]) and not _is_mam(d)
def _is_ctchest(d): dl=d.lower(); return "ct " in dl and "chest" in dl and "abd" not in dl and "pelv" not in dl
def _is_ctabd(d):   dl=d.lower(); return "ct " in dl and any(k in dl for k in ["abdomen","pelvis","abd/pel","abd pel"])
def _is_mam_bi(d):  dl=d.lower(); return _is_mam(d) and any(k in dl for k in ["bilat","bilateral"," bi ","screen","3d"])
def _is_mam_uni(d): dl=d.lower(); return _is_mam(d) and not any(k in dl for k in ["bilat","bilateral"," bi "]) and any(k in dl for k in [" lt"," rt","left","right"])

def _targeted_rule(cur_desc, pri_desc):
    if _is_mam(cur_desc) and _is_cxr(pri_desc): return False
    if _is_cxr(cur_desc) and _is_mam(pri_desc): return False
    if _is_ctchest(cur_desc) and _is_ctabd(pri_desc): return False
    if _is_ctabd(cur_desc) and _is_ctchest(pri_desc): return False
    if _is_mam_bi(cur_desc) and _is_mam_uni(pri_desc): return True
    if _is_mam_uni(cur_desc) and _is_mam_bi(pri_desc): return True
    return None

def _onnx_predict(cur_desc, pri_descs):
    import torch
    texts = [f"Current exam: {cur_desc}. Prior exam: {pd}." for pd in pri_descs]
    all_probs = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i+32]
        enc = ONNX_TOKENIZER(batch, truncation=True, padding=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(ONNX_MODEL(**enc).logits, dim=-1)[:, 1].numpy()
        all_probs.extend(probs.tolist())
    return all_probs

def _sklearn_probs(cur_desc, cur_date, items):
    texts = [f"CURRENT: {cur_desc} ||| PRIOR: {pd}" for _, pd, _, _ in items]
    metas = [_build_features(cur_desc, cur_date, pd, pdate) for _, pd, pdate, _ in items]
    try:
        X = sp.hstack([VECTORIZER.transform(texts), sp.csr_matrix(np.array(metas, dtype=float))])
        return CLF.predict_proba(X)[:, 1].tolist()
    except Exception as e:
        logger.error("sklearn error: %s", e)
        cp = _get_parts(cur_desc)
        return [0.8 if cp & _get_parts(pd) else 0.2 for _, pd, _, _ in items]

def _predict_batch(current_study, prior_studies):
    cur_desc = _safe_desc(current_study)
    cur_date = _safe_date(current_study)
    final    = [None] * len(prior_studies)
    need_ml  = []

    for i, prior in enumerate(prior_studies):
        pri_desc = _safe_desc(prior)
        pri_date = _safe_date(prior)
        ck = _cache_key(cur_desc, pri_desc, pri_date)

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
        # Get sklearn probabilities for all
        sk_probs = _sklearn_probs(cur_desc, cur_date, need_ml)

        if ONNX_READY:
            # Only send uncertain cases (0.25-0.60) to ONNX
            uncertain = [(j, idx, item) for j, (sk_p, item) in enumerate(zip(sk_probs, need_ml))
                        if 0.25 <= sk_p <= 0.60
                        for idx in [item[0]]]
            uncertain_descs = [need_ml[j][1] for j, _, _ in uncertain]

            if uncertain_descs:
                try:
                    onnx_probs = _onnx_predict(cur_desc, uncertain_descs)
                    for k, (j, orig_i, _) in enumerate(uncertain):
                        # Ensemble: average sklearn + onnx
                        final_prob = (sk_probs[j] + onnx_probs[k]) / 2
                        pred = bool(final_prob >= 0.40)
                        final[orig_i] = pred
                        cache[need_ml[j][3]] = pred
                except Exception as e:
                    logger.error("ONNX inference failed: %s", e)
                    uncertain = []  # fall through to sklearn

            # Non-uncertain: use sklearn directly
            for j, (orig_i, _, _, ck) in enumerate(need_ml):
                if final[orig_i] is None:
                    pred = bool(sk_probs[j] >= 0.35)
                    final[orig_i] = pred
                    cache[ck] = pred
        else:
            # ONNX not ready yet — use sklearn for everything
            for j, (orig_i, _, _, ck) in enumerate(need_ml):
                pred = bool(sk_probs[j] >= 0.35)
                final[orig_i] = pred
                cache[ck] = pred

    return [bool(r) if r is not None else False for r in final]

@app.post("/predict")
async def predict(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    cases = body.get("cases") or []
    if not isinstance(cases, list):
        raise HTTPException(status_code=400, detail="'cases' must be a list")

    logger.info("Request: %d cases | ONNX ready: %s", len(cases), ONNX_READY)
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
    return {
        "status":     "ok",
        "onnx_ready": ONNX_READY,
        "model":      "sklearn+onnx_ensemble" if ONNX_READY else "sklearn",
        "cache_size": len(cache),
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
