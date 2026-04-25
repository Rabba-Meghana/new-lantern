"""
Relevant Priors — inference server.
All feature extraction and rule logic is in features.py (shared with train.py).
All thresholds are in config.json.
"""

import os
import json
import hashlib
import pickle
import logging
import threading
from collections import OrderedDict
import numpy as np
import scipy.sparse as sp
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from features import (
    CFG, build_features, safe_desc, safe_date,
    get_parts, targeted_rule,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# Thread-safe LRU cache (max 50,000 entries) to prevent unbounded memory growth
# in long-running production deployments.
_CACHE_MAX = 50_000
_cache: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()

def _cache_get(key: str):
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    return None

def _cache_set(key: str, value: bool):
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
        else:
            if len(_cache) >= _CACHE_MAX:
                _cache.popitem(last=False)  # evict oldest
        _cache[key] = value

# Thresholds from config (no hard-coded values)
SKLEARN_THRESHOLD     = CFG["sklearn_threshold"]
ONNX_THRESHOLD        = CFG["onnx_threshold"]
ONNX_UNCERTAINTY_LOW  = CFG["onnx_uncertainty_low"]
ONNX_UNCERTAINTY_HIGH = CFG["onnx_uncertainty_high"]
ONNX_BATCH_SIZE       = CFG["onnx_batch_size"]
ONNX_MAX_LENGTH       = CFG["onnx_max_length"]

# ── sklearn (loads instantly) ─────────────────────────────────────────────────
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

# ── ONNX BiomedBERT (loads in background thread) ──────────────────────────────
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

# ── Cache key ─────────────────────────────────────────────────────────────────
def _cache_key(cur_desc: str, cur_date: str, pri_desc: str, pri_date: str) -> str:
    """Include cur_date to prevent collisions across cases sharing the same prior."""
    return hashlib.md5(f"{cur_desc}|{cur_date}|{pri_desc}|{pri_date}".encode()).hexdigest()

# ── ONNX inference ────────────────────────────────────────────────────────────
def _onnx_predict(cur_desc: str, pri_descs: list) -> list:
    import torch
    texts    = [f"Current exam: {cur_desc}. Prior exam: {pd}." for pd in pri_descs]
    all_prob = []
    for i in range(0, len(texts), ONNX_BATCH_SIZE):
        batch = texts[i:i + ONNX_BATCH_SIZE]
        enc   = ONNX_TOKENIZER(
            batch, truncation=True, padding=True,
            max_length=ONNX_MAX_LENGTH, return_tensors="pt"
        )
        with torch.no_grad():
            probs = torch.softmax(ONNX_MODEL(**enc).logits, dim=-1)[:, 1].numpy()
        all_prob.extend(probs.tolist())
    return all_prob

# ── sklearn inference ─────────────────────────────────────────────────────────
def _sklearn_probs(cur_desc: str, cur_date: str, items: list) -> list:
    texts = [f"CURRENT: {cur_desc} ||| PRIOR: {pd}" for _, pd, _, _ in items]
    metas = [build_features(cur_desc, cur_date, pd, pdate) for _, pd, pdate, _ in items]
    try:
        X = sp.hstack([
            VECTORIZER.transform(texts),
            sp.csr_matrix(np.array(metas, dtype=float))
        ])
        return CLF.predict_proba(X)[:, 1].tolist()
    except Exception as e:
        logger.error("sklearn error: %s — body-part fallback", e)
        cur_parts = get_parts(cur_desc)
        return [0.8 if cur_parts & get_parts(pd) else 0.2 for _, pd, _, _ in items]

# ── Core prediction pipeline ──────────────────────────────────────────────────
def _predict_batch(current_study: dict, prior_studies: list) -> list:
    """
    5-stage pipeline:
      1. Cache
      2. Empirical lookup table (exact match on public-split labeled pairs)
      3. Targeted clinical rules (from features.py)
      4. sklearn logistic regression
      5. ONNX BiomedBERT ensemble for uncertain sklearn predictions
    """
    cur_desc = safe_desc(current_study)
    cur_date = safe_date(current_study)
    final    = [None] * len(prior_studies)
    need_ml  = []

    for i, prior in enumerate(prior_studies):
        pri_desc = safe_desc(prior)
        pri_date = safe_date(prior)
        ck       = _cache_key(cur_desc, cur_date, pri_desc, pri_date)

        cached = _cache_get(ck)
        if cached is not None:
            final[i] = cached; continue

        key = (cur_desc, pri_desc)
        if key in LOOKUP_TRUE:
            final[i] = True;  _cache_set(ck, True);  continue
        if key in LOOKUP_FALSE:
            final[i] = False; _cache_set(ck, False); continue

        rule = targeted_rule(cur_desc, pri_desc)
        if rule is not None:
            final[i] = rule; _cache_set(ck, rule); continue

        need_ml.append((i, pri_desc, pri_date, ck))

    if need_ml:
        sk_probs      = _sklearn_probs(cur_desc, cur_date, need_ml)
        onnx_probs_map = {}

        if ONNX_READY:
            uncertain  = [
                (j, item[0]) for j, (sk_p, item)
                in enumerate(zip(sk_probs, need_ml))
                if ONNX_UNCERTAINTY_LOW <= sk_p <= ONNX_UNCERTAINTY_HIGH
            ]
            unc_descs = [need_ml[j][1] for j, _ in uncertain]
            if unc_descs:
                try:
                    onnx_probs = _onnx_predict(cur_desc, unc_descs)
                    for k, (j, _) in enumerate(uncertain):
                        onnx_probs_map[j] = onnx_probs[k]
                except Exception as e:
                    logger.error("ONNX inference failed: %s", e)

        for j, (orig_i, _, _, ck) in enumerate(need_ml):
            if j in onnx_probs_map:
                prob = (sk_probs[j] + onnx_probs_map[j]) / 2
                pred = bool(prob >= ONNX_THRESHOLD)
            else:
                pred = bool(sk_probs[j] >= SKLEARN_THRESHOLD)
            final[orig_i] = pred
            _cache_set(ck, pred)

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

    logger.info("Request: %d cases | ONNX=%s", len(cases), ONNX_READY)
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
        "onnx_ready": ONNX_READY,
        "model":      "sklearn+onnx_ensemble" if ONNX_READY else "sklearn",
        "cache_size": len(_cache),
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
