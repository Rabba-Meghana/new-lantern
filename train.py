"""
Reproducible training script.

Usage:
    python train.py --data relevant_priors_public.json

Outputs:
    model.pkl    Logistic regression classifier (TF-IDF + structured features)
    lookup.json  Empirical lookup table mined from labeled data
"""

import argparse
import json
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

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

def get_parts(desc):
    d = desc.lower()
    return frozenset(p for p, kws in BODY_PARTS.items() if any(k in d for k in kws))

def get_mods(desc):
    d = desc.lower()
    return frozenset(m for m, kws in MODALITIES.items() if any(k in d for k in kws))

def get_side(desc):
    d = desc.upper()
    if "BILAT" in d or "BILATERAL" in d: return "bilateral"
    if (" LT" in d or "LEFT" in d) and not (" RT" in d or "RIGHT" in d): return "left"
    if (" RT" in d or "RIGHT" in d) and not (" LT" in d or "LEFT" in d): return "right"
    return "unknown"

def years_apart(d1, d2):
    try:
        return abs((datetime.strptime(d1[:10], "%Y-%m-%d") -
                    datetime.strptime(d2[:10], "%Y-%m-%d")).days) / 365.25
    except Exception:
        return 3.0

def build_features(cur_desc, cur_date, pri_desc, pri_date):
    cp = get_parts(cur_desc); pp = get_parts(pri_desc)
    cm = get_mods(cur_desc);  pm = get_mods(pri_desc)
    cs = get_side(cur_desc);  ps = get_side(pri_desc)
    years = years_apart(cur_date, pri_date)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to relevant_priors_public.json")
    parser.add_argument("--model-out", default="model.pkl")
    parser.add_argument("--lookup-out", default="lookup.json")
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading {args.data} ...")
    with open(args.data) as f:
        data = json.load(f)

    cases     = data["cases"]
    truth_map = {(t["case_id"], t["study_id"]): t["is_relevant_to_current"]
                 for t in data["truth"]}

    # ── Build training set ────────────────────────────────────────────────────
    texts, metas, labels = [], [], []
    pair_stats = defaultdict(lambda: {"true": 0, "false": 0})

    for case in cases:
        case_id  = case["case_id"]
        cur      = case["current_study"]
        cur_desc = cur["study_description"].upper()
        cur_date = cur["study_date"]

        for prior in case.get("prior_studies", []):
            label = truth_map.get((case_id, prior["study_id"]))
            if label is None:
                continue
            pri_desc = prior["study_description"].upper()
            pri_date = prior["study_date"]

            texts.append(f"CURRENT: {cur_desc} ||| PRIOR: {pri_desc}")
            metas.append(build_features(cur_desc, cur_date, pri_desc, pri_date))
            labels.append(1 if label else 0)

            key = (cur_desc, pri_desc)
            if label:
                pair_stats[key]["true"] += 1
            else:
                pair_stats[key]["false"] += 1

    print(f"Dataset: {len(texts)} samples  positive={sum(labels)/len(labels)*100:.1f}%")

    # ── Build lookup tables (pairs seen ONLY as true or ONLY as false, min 2x) ─
    lookup_true  = [[k[0], k[1]] for k, v in pair_stats.items()
                    if v["true"] >= 2 and v["false"] == 0]
    lookup_false = [[k[0], k[1]] for k, v in pair_stats.items()
                    if v["false"] >= 2 and v["true"] == 0]
    print(f"Lookup table: {len(lookup_true)} always-true, {len(lookup_false)} always-false pairs")

    with open(args.lookup_out, "w") as f:
        json.dump({"true": lookup_true, "false": lookup_false}, f)
    print(f"Saved {args.lookup_out}")

    # ── Train classifier ──────────────────────────────────────────────────────
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), max_features=8000,
        sublinear_tf=True, token_pattern=r"[A-Z0-9]+"
    )
    X_tfidf = vectorizer.fit_transform(texts)
    X_meta  = sp.csr_matrix(np.array(metas, dtype=float))
    X       = sp.hstack([X_tfidf, X_meta])
    y       = np.array(labels)

    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")

    print(f"Running {args.cv_folds}-fold cross-validation ...")
    cv_scores = cross_val_score(clf, X, y, cv=args.cv_folds, scoring="accuracy")
    print(f"CV accuracy: {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")

    clf.fit(X, y)
    train_acc = accuracy_score(y, clf.predict(X))
    print(f"Train accuracy: {train_acc*100:.2f}%")
    print(classification_report(y, clf.predict(X), target_names=["not_relevant", "relevant"]))

    with open(args.model_out, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "clf": clf}, f)
    print(f"Saved {args.model_out}")

if __name__ == "__main__":
    main()
