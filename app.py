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
logger.info("Lookup table loaded: %d true, %d false", len(LOOKUP_TRUE), len(LOOKUP_FALSE))

# ── Feature extraction ────────────────────────────────────────────────────────
BODY_PARTS = {
    "brain":          ["brain", "head", "cranial", "cranium", "intracranial", "cerebr",
                       "neuro", "skull", "orbit", "sella", "pituitary"],
    "spine_cervical": ["cervical", "c-spine", "c spine"],
    "spine_thoracic": ["thoracic spine", "t-spine", "t spine"],
    "spine_lumbar":   ["lumbar", "l-spine", "l spine", "lumbosacral", "sacral"],
    "chest":          ["chest", "thorax", "lung", "pulmon", "pleural", "mediastin",
                       "rib", "heart", "coronary", "cardiac", "spect", "nm myo",
                       "nmmyo", "myocard"],
    "abdomen":        ["abdomen", "abdominal", "liver", "hepat", "pancrea", "spleen",
                       "kidney", "renal", "adrenal", "bowel", "colon", "rectum",
                       "gallbladder", "biliary", "aaa"],
    "pelvis":         ["pelvis", "pelvic", "bladder", "prostate", "uterus", "ovary",
                       "abd/pel", "abd pel"],
    "upper_ext":      ["shoulder", "humerus", "elbow", "forearm", "wrist", "hand",
                       "finger", "clavicle"],
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

def _get_parts(desc: str) -> frozenset:
    d = desc.lower()
    return frozenset(p for p, kws in BODY_PARTS.items() if any(k in d for k in kws))

def _get_mods(desc: str) -> frozenset:
    d = desc.lower()
    return frozenset(m for m, kws in MODALITIES.items() if any(k in d for k in kws))

def _get_side(desc: str) -> str:
    d = desc.upper()
    has_lt = " LT" in d or "LEFT" in d
    has_rt = " RT" in d or "RIGHT" in d
    has_bi = "BILAT" in d or "BILATERAL" in d or " BI " in d
    if has_bi:
        return "bilateral"
    if has_lt and not has_rt:
        return "left"
    if has_rt and not has_lt:
        return "right"
    return "unknown"

def _years_apart(date1: str, date2: str) -> float:
    """Return absolute years between two ISO date strings. Returns 3.0 on any parse error."""
    try:
        d1 = datetime.strptime(date1[:10], "%Y-%m-%d")
        d2 = datetime.strptime(date2[:10], "%Y-%m-%d")
        return abs((d1 - d2).days) / 365.25
    except (ValueError, TypeError):
        return 3.0

def _build_features(cur_desc: str, cur_date: str, pri_desc: str, pri_date: str) -> list:
    cp = _get_parts(cur_desc); pp = _get_parts(pri_desc)
    cm = _get_mods(cur_desc);  pm = _get_mods(pri_desc)
    cs = _get_side(cur_desc);  ps = _get_side(pri_desc)
    years = _years_apart(cur_date, pri_date)
    po = len(cp & pp); mo = len(cm & pm)
    pu = len(cp | pp) or 1; mu = len(cm | pm) or 1
    opp  = int((cs == "left" and ps == "right") or (cs == "right" and ps == "left"))
    same = int(cs == ps and cs != "unknown")
    bi   = int(cs == "bilateral" and ps == "bilateral")
    return [
        years / 20.0, po, po / pu, mo, mo / mu,
        opp, same, int(cur_desc.upper() == pri_desc.upper()), bi,
        int(years <= 1), int(years <= 3), int(years > 10),
        int(po > 0), int(mo > 0), int(po > 0 and mo > 0),
    ]

def _safe_desc(study: dict) -> str:
    """Return uppercased study_description or empty string if missing."""
    return str(study.get("study_description") or "").upper()

def _safe_date(study: dict) -> str:
    """Return study_date string or empty string if missing."""
    return str(study.get("study_date") or "")

def _cache_key(cur_desc: str, pri_desc: str, pri_date: str) -> str:
    raw = f"{cur_desc}|{pri_desc}|{pri_date}"
    return hashlib.md5(raw.encode()).hexdigest()

def _predict_batch(current_study: dict, prior_studies: list) -> list[bool]:
    """
    3-stage pipeline: lookup table → sklearn classifier → deterministic fallback.
    Never raises; always returns a bool for every prior.
    """
    cur_desc = _safe_desc(current_study)
    cur_date = _safe_date(current_study)
    final      = [None] * len(prior_studies)
    need_ml    = []
    need_ml_idx = []

    for i, prior in enumerate(prior_studies):
        pri_desc = _safe_desc(prior)
        pri_date = _safe_date(prior)
        ck = _cache_key(cur_desc, pri_desc, pri_date)

        if ck in cache:
            final[i] = cache[ck]
            continue

        key = (cur_desc, pri_desc)
        if key in LOOKUP_TRUE:
            final[i] = True
            cache[ck] = True
        elif key in LOOKUP_FALSE:
            final[i] = False
            cache[ck] = False
        else:
            need_ml.append((i, prior, pri_desc, pri_date, ck))
            need_ml_idx.append(i)

    if need_ml:
        texts = [
            f"CURRENT: {cur_desc} ||| PRIOR: {pri_desc}"
            for _, _, pri_desc, _, _ in need_ml
        ]
        metas = [
            _build_features(cur_desc, cur_date, pri_desc, pri_date)
            for _, _, pri_desc, pri_date, _ in need_ml
        ]
        try:
            X_tfidf = VECTORIZER.transform(texts)
            X_meta  = sp.csr_matrix(np.array(metas, dtype=float))
            X       = sp.hstack([X_tfidf, X_meta])
            probs   = CLF.predict_proba(X)[:, 1]
            preds   = [bool(p >= 0.35) for p in probs]
        except Exception as exc:
            logger.error("sklearn inference failed: %s", exc)
            # Deterministic fallback: predict relevant if any body-part overlap
            cur_parts = _get_parts(cur_desc)
            preds = [
                bool(cur_parts & _get_parts(pri_desc))
                for _, _, pri_desc, _, _ in need_ml
            ]

        for j, (orig_i, _, _, _, ck) in enumerate(need_ml):
            final[orig_i] = preds[j]
            cache[ck] = preds[j]

    # Safety: any remaining None gets False (conservative: don't show irrelevant priors)
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
            study_id = prior.get("study_id", "")
            predictions.append({
                "case_id":              case_id,
                "study_id":             study_id,
                "predicted_is_relevant": is_relevant,
            })

    logger.info("Response: %d predictions", len(predictions))
    return JSONResponse(content={"predictions": predictions})


@app.get("/health")
def health():
    return {"status": "ok", "model": "sklearn+lookup", "cache_size": len(cache)}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
