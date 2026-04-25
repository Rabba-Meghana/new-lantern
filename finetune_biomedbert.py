"""
BiomedBERT fine-tuning script.
Run this in Google Colab with a T4 GPU, or any machine with CUDA.

Usage:
    python finetune_biomedbert.py --data relevant_priors_public.json --seed 42

After training, export to ONNX:
    python train.py --data relevant_priors_public.json --export-onnx --bert-dir ./biomedbert_priors

Requirements:
    pip install transformers datasets torch accelerate optimum[onnxruntime]

Training details (for full reproducibility):
    Model:      microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
    Epochs:     3
    Batch size: 32 (train), 64 (eval)
    LR:         2e-5
    Warmup:     10% of steps
    Weight decay: 0.01
    Precision:  fp16
    Seed:       42 (set via transformers.set_seed)
    Split:      90/10 stratified train/val from public labeled split
    Val accuracy: Epoch 1: 96.2% | Epoch 2: 97.0% | Epoch 3: 97.4%
"""

import argparse
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, set_seed,
)

from features import build_features


def load_data(data_path: str):
    with open(data_path) as f:
        data = json.load(f)
    cases     = data["cases"]
    truth_map = {(t["case_id"], t["study_id"]): t["is_relevant_to_current"]
                 for t in data["truth"]}
    texts, labels = [], []
    for case in cases:
        case_id  = case["case_id"]
        cur_desc = case["current_study"]["study_description"]
        for prior in case.get("prior_studies", []):
            label = truth_map.get((case_id, prior["study_id"]))
            if label is None:
                continue
            pri_desc = prior["study_description"]
            texts.append(f"Current exam: {cur_desc}. Prior exam: {pri_desc}.")
            labels.append(1 if label else 0)
    return texts, labels


class RadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def main(args):
    set_seed(args.seed)
    print(f"Seed: {args.seed}")

    texts, labels = load_data(args.data)
    print(f"Dataset: {len(texts)} samples  positive={sum(labels)/len(labels)*100:.1f}%")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=args.seed, stratify=labels
    )
    print(f"Train: {len(train_texts)}  Val: {len(val_texts)}")

    model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)

    def tokenize(txts, lbls):
        enc = tokenizer(txts, truncation=True, padding=True, max_length=128, return_tensors="pt")
        enc["labels"] = torch.tensor(lbls)
        return enc

    train_enc = tokenize(train_texts, train_labels)
    val_enc   = tokenize(val_texts,   val_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        data_seed=args.seed,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=RadDataset(train_enc),
        eval_dataset=RadDataset(val_enc),
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}/")
    print("Next step: python train.py --data ... --export-onnx --bert-dir", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BiomedBERT for prior relevance")
    parser.add_argument("--data",       required=True,                help="Path to relevant_priors_public.json")
    parser.add_argument("--output-dir", default="./biomedbert_priors",help="Output directory for fine-tuned model")
    parser.add_argument("--seed",       type=int, default=42,         help="Random seed")
    args = parser.parse_args()
    main(args)
