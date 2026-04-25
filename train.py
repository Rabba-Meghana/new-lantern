"""
Reproducible training script for the Relevant Priors classifier.

Produces:
    model.pkl    Logistic regression (TF-IDF + structured features)
    lookup.json  Empirical lookup table from labeled data

Usage:
    python train.py --data relevant_priors_public.json
    python train.py --data relevant_priors_public.json --export-onnx --bert-dir ./biomedbert_priors

ONNX export requires: pip install optimum[onnxruntime] torch transformers
BiomedBERT fine-tuning is done in Colab (finetune_biomedbert.ipynb). The --export-onnx
flag exports a pre-trained model from --bert-dir to ONNX int8.

Reproducibility:
    Seeds are set via --seed (default 42). CV uses StratifiedKFold with shuffle=True
    and the same seed. Results may vary slightly across platforms due to floating-point.
"""

import argparse
import json
import pickle
import random
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score

from features import (
    CFG, build_features, get_parts, get_mods, get_side, years_apart,
    targeted_rule,
)


def set_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def build_dataset(data: dict):
    """Build texts, metas, labels, and pair statistics from labeled data."""
    cases     = data["cases"]
    truth_map = {
        (t["case_id"], t["study_id"]): t["is_relevant_to_current"]
        for t in data["truth"]
    }

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

    return texts, metas, labels, pair_stats, cases, truth_map


def build_lookup(pair_stats: dict, min_count: int):
    """Build always-true and always-false lookup tables."""
    lookup_true  = [[k[0], k[1]] for k, v in pair_stats.items()
                    if v["true"] >= min_count and v["false"] == 0]
    lookup_false = [[k[0], k[1]] for k, v in pair_stats.items()
                    if v["false"] >= min_count and v["true"] == 0]
    return lookup_true, lookup_false


def run_ablation(cases, truth_map, clf, vectorizer, lookup_true_set, lookup_false_set):
    """Per-category accuracy breakdown on the full public split."""
    results = defaultdict(lambda: {"correct": 0, "total": 0})

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

            key = (cur_desc, pri_desc)
            if key in lookup_true_set:
                pred = True
            elif key in lookup_false_set:
                pred = False
            else:
                rule = targeted_rule(cur_desc, pri_desc)
                if rule is not None:
                    pred = rule
                else:
                    text  = f"CURRENT: {cur_desc} ||| PRIOR: {pri_desc}"
                    feats = build_features(cur_desc, cur_date, pri_desc, pri_date)
                    X     = sp.hstack([
                        vectorizer.transform([text]),
                        sp.csr_matrix([feats])
                    ])
                    pred = bool(clf.predict_proba(X)[0, 1] >= CFG["sklearn_threshold"])

            cs    = get_side(cur_desc); ps = get_side(pri_desc)
            opp   = ((cs == "left" and ps == "right") or (cs == "right" and ps == "left"))
            cp    = get_parts(cur_desc); pp = get_parts(pri_desc)
            yrs   = years_apart(cur_date, pri_date)

            cats = []
            if opp: cats.append("opposite_laterality")
            cats.append("body_part_overlap" if cp & pp else "no_body_part_overlap")
            cats.append("modality_overlap" if get_mods(cur_desc) & get_mods(pri_desc) else "no_modality_overlap")
            if yrs <= 1:   cats.append("age_0_1yr")
            elif yrs <= 3: cats.append("age_1_3yr")
            elif yrs <= 10:cats.append("age_3_10yr")
            else:          cats.append("age_10plus_yr")

            for cat in cats:
                results[cat]["total"] += 1
                if pred == label:
                    results[cat]["correct"] += 1

    return {cat: {"accuracy": v["correct"] / v["total"] * 100, "n": v["total"]}
            for cat, v in results.items()}


def main(args):
    set_seeds(args.seed)
    print(f"Seed: {args.seed}")
    print(f"Loading {args.data} ...")

    with open(args.data) as f:
        data = json.load(f)

    texts, metas, labels, pair_stats, cases, truth_map = build_dataset(data)
    print(f"Dataset: {len(texts)} samples  positive={sum(labels)/len(labels)*100:.1f}%")

    # ── Lookup tables ──────────────────────────────────────────────────────────
    lookup_true, lookup_false = build_lookup(pair_stats, CFG["lookup_min_count"])
    print(f"Lookup: {len(lookup_true)} always-true, {len(lookup_false)} always-false pairs")

    with open(args.lookup_out, "w") as f:
        json.dump({"true": lookup_true, "false": lookup_false}, f)
    print(f"Saved {args.lookup_out}")

    # ── Train ──────────────────────────────────────────────────────────────────
    vectorizer = TfidfVectorizer(
        ngram_range=tuple(CFG["tfidf_ngram_range"]),
        max_features=CFG["tfidf_max_features"],
        sublinear_tf=True,
        token_pattern=r"[A-Z0-9]+"
    )
    X_tfidf = vectorizer.fit_transform(texts)
    X_meta  = sp.csr_matrix(np.array(metas, dtype=float))
    X       = sp.hstack([X_tfidf, X_meta])
    y       = np.array(labels)

    clf = LogisticRegression(
        C=CFG["lr_C"],
        max_iter=CFG["lr_max_iter"],
        class_weight=CFG["lr_class_weight"],
        random_state=args.seed,
    )

    # Explicit StratifiedKFold with shuffle and seed for full reproducibility
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    print(f"Running {args.cv_folds}-fold StratifiedKFold CV (seed={args.seed}) ...")
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    print(f"CV accuracy (public split only): {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")
    print("NOTE: this is measured on the public split — private split is unseen")

    clf.fit(X, y)
    print(f"Train accuracy: {accuracy_score(y, clf.predict(X))*100:.2f}%")
    print(classification_report(y, clf.predict(X), target_names=["not_relevant", "relevant"]))

    with open(args.model_out, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "clf": clf, "seed": args.seed}, f)
    print(f"Saved {args.model_out}")

    # ── Ablation ───────────────────────────────────────────────────────────────
    lt_set = {(a, b) for a, b in lookup_true}
    lf_set = {(a, b) for a, b in lookup_false}
    print("\n=== Ablation by category (public split, full pipeline minus ONNX) ===")
    ablation = run_ablation(cases, truth_map, clf, vectorizer, lt_set, lf_set)
    for cat, v in sorted(ablation.items()):
        print(f"  {cat:30s}: {v['accuracy']:.1f}% (n={v['n']})")

    return clf, vectorizer


def export_onnx(bert_dir: str, output_dir: str):
    """
    Export a fine-tuned BiomedBERT model to quantised ONNX int8.
    The model must be fine-tuned separately (see finetune_biomedbert.ipynb).
    """
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    fp32_dir = output_dir + "_fp32"
    print(f"Exporting to ONNX fp32: {fp32_dir}")
    ort_model = ORTModelForSequenceClassification.from_pretrained(bert_dir, export=True)
    ort_model.save_pretrained(fp32_dir)
    AutoTokenizer.from_pretrained(bert_dir).save_pretrained(fp32_dir)

    print(f"Quantising to int8: {output_dir}")
    quantizer = ORTQuantizer.from_pretrained(fp32_dir)
    qconfig   = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=output_dir, quantization_config=qconfig)
    AutoTokenizer.from_pretrained(bert_dir).save_pretrained(output_dir)
    print(f"ONNX model saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Relevant Priors classifier")
    parser.add_argument("--data",        required=True,          help="Path to relevant_priors_public.json")
    parser.add_argument("--model-out",   default="model.pkl",    help="Output path for sklearn model")
    parser.add_argument("--lookup-out",  default="lookup.json",  help="Output path for lookup table")
    parser.add_argument("--cv-folds",    type=int, default=5,    help="Number of CV folds")
    parser.add_argument("--seed",        type=int, default=42,   help="Random seed for reproducibility")
    parser.add_argument("--export-onnx", action="store_true",    help="Export BiomedBERT to ONNX")
    parser.add_argument("--bert-dir",    default="./biomedbert_priors", help="Fine-tuned BiomedBERT dir")
    parser.add_argument("--onnx-out",    default="./onnx_int8",  help="ONNX output directory")
    args = parser.parse_args()

    main(args)

    if args.export_onnx:
        export_onnx(bert_dir=args.bert_dir, output_dir=args.onnx_out)
