"""
BERTimbau Large Fine-tuning for Multi-Label Topic Classification

Fine-tunes neuralmind/bert-large-portuguese-cased on Portuguese municipal
meeting minutes for multi-label topic classification into 22 administrative
categories. Uses the canonical dset.json dataset and split_info.json split.

Key Features:
    - Fine-tuning of BERTimbau Large (neuralmind/bert-large-portuguese-cased)
    - Multi-label classification with BCEWithLogitsLoss
    - Dynamic threshold optimization per label on validation set
    - Early stopping based on validation loss
    - GPU acceleration support

Hyperparameters:
    - max_length: 512
    - batch_size: 16
    - epochs: 10
    - learning_rate: 5e-5
    - weight_decay: 0.01
    - warmup_ratio: 0.1

Date: October 2025
"""

import os
import json
import re
import numpy as np
import torch
import random
from transformers import (
    AutoTokenizer, BertForSequenceClassification,
    TrainingArguments, Trainer
)
from datasets import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    f1_score, accuracy_score, hamming_loss,
    classification_report, average_precision_score
)
from joblib import dump


# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_SEED = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DSET_PATH = os.path.join(SCRIPT_DIR, "dataset_sample", "dset.json")
SPLIT_JSON_PATH = os.path.join(SCRIPT_DIR, "split_info.json")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "results", "pt_results.json")
MODEL_NAME = "Bertimbau_Large_Finetune_512_10ep_LR5e5"
OUTPUT_MODEL_DIR = os.path.join(SCRIPT_DIR, "bertimbau_large_finetuned_multilabel")

# Model configuration
BERT_MODEL_NAME = "neuralmind/bert-large-portuguese-cased"
MAX_LENGTH = 512

# Training configuration
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# Threshold optimization
THRESHOLD_MIN = 0.1
THRESHOLD_MAX = 0.9
THRESHOLD_STEP = 0.05

# Minimum text length filter
MIN_TEXT_LENGTH = 10

# Reproducibility
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# TEXT PREPROCESSING
# ============================================================
def clean_text(text):
    """Basic text cleaning: lowercase, whitespace and punctuation normalization."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


# ============================================================
# DATA LOADING (canonical dset.json + split_info.json)
# ============================================================
def load_data(dset_path, split_files):
    """
    Load text segments from canonical dset.json for the given split files.

    Args:
        dset_path (str): Path to dset.json
        split_files (list): List of filenames belonging to this split

    Returns:
        tuple: (texts, labels) - cleaned texts and their topic label lists
    """
    file_set = set(split_files)
    with open(dset_path, "r", encoding="utf-8") as f:
        dset = json.load(f)

    texts, labels = [], []
    for muni_obj in dset["municipalities"]:
        for minute in muni_obj["minutes"]:
            source_file = minute["minute_id"] + ".json"
            if source_file not in file_set:
                continue
            for item in minute.get("agenda_items", []):
                text = item.get("text", "")
                topics = item.get("topics", [])
                if topics and len(text) > MIN_TEXT_LENGTH:
                    texts.append(clean_text(text))
                    labels.append(topics)

    return texts, labels


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    print("\n" + "=" * 70)
    print("BERTIMBAU LARGE FINE-TUNING - MULTI-LABEL CLASSIFICATION")
    print("=" * 70 + "\n")

    # -------------------- Data Loading --------------------
    print("Loading data from canonical dataset...")
    with open(SPLIT_JSON_PATH, "r", encoding="utf-8") as f:
        split_info = json.load(f)

    train_texts, train_labels = load_data(DSET_PATH, split_info["train_files"])
    val_texts, val_labels = load_data(DSET_PATH, split_info["val_files"])
    test_texts, test_labels = load_data(DSET_PATH, split_info["test_files"])

    all_labels = train_labels + val_labels + test_labels

    print(f"  Train: {len(train_texts)} segments")
    print(f"  Validation: {len(val_texts)} segments")
    print(f"  Test: {len(test_texts)} segments")
    print(f"  Total: {len(train_texts) + len(val_texts) + len(test_texts)} segments")

    # -------------------- Label Encoding --------------------
    print("\nEncoding labels...")
    mlb = MultiLabelBinarizer()
    mlb.fit(all_labels)

    y_train = mlb.transform(train_labels).astype(np.float32)
    y_val = mlb.transform(val_labels).astype(np.float32)
    y_test = mlb.transform(test_labels).astype(np.float32)

    n_labels = len(mlb.classes_)
    print(f"  Number of unique labels: {n_labels}")

    # -------------------- Tokenization --------------------
    print(f"\nTokenizing with {BERT_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    def tokenize_function(texts, labels_list):
        encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        encodings["labels"] = labels_list
        return encodings

    train_dataset = Dataset.from_dict(tokenize_function(train_texts, y_train.tolist()))
    val_dataset = Dataset.from_dict(tokenize_function(val_texts, y_val.tolist()))
    test_dataset = Dataset.from_dict(tokenize_function(test_texts, y_test.tolist()))

    def cast_labels(example):
        example["labels"] = np.array(example["labels"], dtype=np.float32)
        return example

    train_dataset = train_dataset.map(cast_labels)
    val_dataset = val_dataset.map(cast_labels)
    test_dataset = test_dataset.map(cast_labels)

    # -------------------- Model --------------------
    print("\nLoading model...")
    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=n_labels,
        problem_type="multi_label_classification",
    )

    # -------------------- Training --------------------
    print("\nStarting training...")
    training_args = TrainingArguments(
        output_dir=os.path.join(SCRIPT_DIR, "training_output"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(SCRIPT_DIR, "logs"),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # -------------------- Predictions --------------------
    print("\nGenerating predictions...")
    val_preds = trainer.predict(val_dataset)
    test_preds = trainer.predict(test_dataset)

    y_val_proba = torch.sigmoid(torch.tensor(val_preds.predictions)).numpy()
    y_test_proba = torch.sigmoid(torch.tensor(test_preds.predictions)).numpy()

    # -------------------- Threshold Optimization --------------------
    print("\nOptimizing classification thresholds on validation set...")
    optimal_thresholds = []

    for i in range(n_labels):
        best_f1, best_thresh = 0, 0.5
        for thresh in np.arange(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP):
            y_pred_i = (y_val_proba[:, i] >= thresh).astype(int)
            f1 = f1_score(y_val[:, i], y_pred_i, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh
        optimal_thresholds.append(best_thresh)

    print("  Optimal thresholds per label:")
    for label, thresh in zip(mlb.classes_, np.round(optimal_thresholds, 2)):
        print(f"    {label}: {thresh}")

    y_pred_test = np.zeros_like(y_test)
    for i, thresh in enumerate(optimal_thresholds):
        y_pred_test[:, i] = (y_test_proba[:, i] >= thresh).astype(int)

    # -------------------- Evaluation --------------------
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    accuracy = accuracy_score(y_test, y_pred_test)
    hamming = hamming_loss(y_test, y_pred_test)
    f1_macro = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
    f1_micro = f1_score(y_test, y_pred_test, average="micro", zero_division=0)
    avg_precision = average_precision_score(y_test, y_test_proba, average="macro")

    class_report = classification_report(
        y_test, y_pred_test,
        target_names=mlb.classes_,
        zero_division=0,
        output_dict=True,
    )

    print(f"\n  Accuracy: {accuracy:.4f}")
    print(f"  Hamming Loss: {hamming:.4f}")
    print(f"  F1-Macro: {f1_macro:.4f}")
    print(f"  F1-Micro: {f1_micro:.4f}")
    print(f"  Average Precision (Macro): {avg_precision:.4f}")

    # -------------------- Save Results --------------------
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)

    results[MODEL_NAME] = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "hamming_loss": float(hamming),
        "average_precision_macro": float(avg_precision),
        "classification_report": class_report,
        "model_info": {
            "algorithm": "Fine-tuned BERTimbau Large",
            "model": BERT_MODEL_NAME,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "n_labels": n_labels,
            "labels": mlb.classes_.tolist(),
        },
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # -------------------- Save Model & Artifacts --------------------
    trainer.save_model(OUTPUT_MODEL_DIR)
    np.save(os.path.join(SCRIPT_DIR, "optimal_thresholds.npy"), np.array(optimal_thresholds))
    dump(mlb, os.path.join(SCRIPT_DIR, "mlb_encoder.joblib"))

    print("\n  Model, thresholds, and label encoder saved.")
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
