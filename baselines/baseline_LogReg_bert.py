"""
Logistic Regression with BERTimbau Embeddings for Multi-Label Text Classification

This module implements a Logistic Regression classifier using BERTimbau embeddings
(CLS token representation) for Portuguese municipal text classification. Combines
the power of pre-trained BERT with the simplicity and speed of Logistic Regression.

Key Features:
    - BERTimbau-base CLS token embeddings
    - Logistic Regression with One-vs-Rest strategy
    - Dynamic threshold optimization per label
    - Class-balanced training
    - Efficient inference without fine-tuning

Requirements:
    - transformers
    - torch
    - sklearn
    - joblib

Author: Miguel
Date: October 2025
"""

import os
import json
import re
import numpy as np
import torch
import random
from collections import Counter
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
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
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DSET_PATH = os.path.join(ROOT_DIR, "dataset_sample", "dset.json")
SPLIT_JSON_PATH = os.path.join(ROOT_DIR, "split_info.json")
RESULTS_PATH = os.path.join(ROOT_DIR, "results", "pt_results.json")
MODEL_NAME = "LogBERT_DynamicThreshold"

# BERTimbau configuration
BERT_MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
BERT_MAX_LENGTH = 256
BERT_BATCH_SIZE = 16

# Logistic Regression configuration
LR_SOLVER = "liblinear"
LR_CLASS_WEIGHT = "balanced"
LR_MAX_ITER = 1000

# Threshold optimization configuration
THRESHOLD_MIN = 0.1
THRESHOLD_MAX = 0.9
THRESHOLD_STEP = 0.05

# Minimum text length filter
MIN_TEXT_LENGTH = 10

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# TEXT PREPROCESSING
# ============================================================
def clean_text(text):
    """
    Basic text cleaning: lowercase, whitespace and punctuation normalization.

    Args:
        text (str): Raw input text

    Returns:
        str: Cleaned text
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def load_data(dset_path, split_files):
    """
    Load text segments from canonical dset.json for the given split files.

    Args:
        dset_path (str): Path to dset.json
        split_files (list): List of filenames belonging to this split

    Returns:
        tuple: (texts, labels, municipalities)
    """
    file_set = set(split_files)
    with open(dset_path, "r", encoding="utf-8") as f:
        dset = json.load(f)

    texts, labels, municipalities = [], [], []
    for muni_obj in dset["municipalities"]:
        muni_name = muni_obj["municipality"]
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
                    municipalities.append(muni_name)

    return texts, labels, municipalities


# ============================================================
# BERTIMBAU EMBEDDING EXTRACTION
# ============================================================
def get_bertimbau_embeddings(texts, tokenizer, model, batch_size=16, max_length=256):
    """
    Extract BERTimbau CLS token embeddings for classification.

    The CLS token (first token) representation from BERT's last hidden layer
    is used as a fixed-length sentence embedding.

    Args:
        texts (list): List of text strings
        tokenizer: Pretrained BERTimbau tokenizer
        model: Pretrained BERTimbau model
        batch_size (int): Batch size for processing
        max_length (int): Maximum sequence length

    Returns:
        numpy.ndarray: Matrix of CLS embeddings (n_samples, hidden_size)
    """
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            # Tokenize with padding and truncation
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            # Move to device
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Extract CLS token representation (first token of last hidden layer)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)

    return np.vstack(embeddings)


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================
def optimize_thresholds_per_label(y_true, y_proba, threshold_range):
    """
    Optimize classification threshold for each label independently.

    Uses F1-score maximization on validation data to find the optimal
    threshold for each label. This approach handles class imbalance
    more effectively than using a fixed threshold.

    Args:
        y_true (numpy.ndarray): True binary labels (n_samples, n_labels)
        y_proba (numpy.ndarray): Predicted probabilities (n_samples, n_labels)
        threshold_range (numpy.ndarray): Range of thresholds to evaluate

    Returns:
        list: Optimal threshold for each label
    """
    n_labels = y_true.shape[1]
    optimal_thresholds = []

    for label_idx in range(n_labels):
        best_f1 = 0.0
        best_thresh = 0.5

        # Grid search over threshold range
        for thresh in threshold_range:
            y_pred = (y_proba[:, label_idx] >= thresh).astype(int)
            f1 = f1_score(y_true[:, label_idx], y_pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        optimal_thresholds.append(best_thresh)

    return optimal_thresholds


# ============================================================
# DATASET STATISTICS
# ============================================================
def print_dataset_statistics(texts, labels, municipalities):
    """
    Print comprehensive dataset statistics including municipality and label distributions.

    Args:
        texts (list): List of text strings
        labels (list): List of label sets
        municipalities (list): List of municipality names
    """
    # Municipality distribution
    muni_counts = Counter(municipalities)
    print("\nDataset Statistics:")
    print("\n  Segments per municipality:")
    for muni, count in muni_counts.most_common():
        print(f"    {muni}: {count}")

    # Label distribution
    flat_labels = [label for sublist in labels for label in sublist]
    label_counts = Counter(flat_labels)

    print("\n  Top 10 most frequent labels:")
    for label, count in label_counts.most_common(10):
        print(f"    {label}: {count}")

    # Label cardinality
    label_cardinality = [len(label_set) for label_set in labels]
    print(f"\n  Label cardinality - Min: {min(label_cardinality)}, "
          f"Max: {max(label_cardinality)}, Avg: {np.mean(label_cardinality):.2f}")


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    """
    Main training and evaluation pipeline.
    """
    print("\n" + "="*70)
    print("LOGISTIC REGRESSION WITH BERTIMBAU EMBEDDINGS")
    print("="*70 + "\n")

    # -------------------- Data Loading --------------------
    print("Loading data...")
    with open(SPLIT_JSON_PATH, "r", encoding="utf-8") as f:
        split_info = json.load(f)

    train_texts, train_labels, train_munis = load_data(
        DSET_PATH, split_info["train_files"]
    )
    val_texts, val_labels, val_munis = load_data(
        DSET_PATH, split_info["val_files"]
    )
    test_texts, test_labels, test_munis = load_data(
        DSET_PATH, split_info["test_files"]
    )

    print(f"  Train: {len(train_texts)} segments")
    print(f"  Validation: {len(val_texts)} segments")
    print(f"  Test: {len(test_texts)} segments")

    # Print combined dataset statistics
    all_texts = train_texts + val_texts + test_texts
    all_labels = train_labels + val_labels + test_labels
    all_municipalities = train_munis + val_munis + test_munis
    print_dataset_statistics(all_texts, all_labels, all_municipalities)

    # -------------------- Label Encoding --------------------
    print("\nEncoding labels...")
    mlb = MultiLabelBinarizer()
    mlb.fit(all_labels)

    y_train = mlb.transform(train_labels)
    y_val = mlb.transform(val_labels)
    y_test = mlb.transform(test_labels)

    n_labels = len(mlb.classes_)
    print(f"  Number of unique labels: {n_labels}")

    # -------------------- BERTimbau Embeddings --------------------
    print(f"\nExtracting BERTimbau embeddings ({BERT_MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME).to(device)
    bert_model.eval()

    print("  Processing train set...")
    X_train = get_bertimbau_embeddings(
        train_texts, tokenizer, bert_model, BERT_BATCH_SIZE, BERT_MAX_LENGTH
    )

    print("  Processing validation set...")
    X_val = get_bertimbau_embeddings(
        val_texts, tokenizer, bert_model, BERT_BATCH_SIZE, BERT_MAX_LENGTH
    )

    print("  Processing test set...")
    X_test = get_bertimbau_embeddings(
        test_texts, tokenizer, bert_model, BERT_BATCH_SIZE, BERT_MAX_LENGTH
    )

    print(f"\nEmbedding dimensions: {X_train.shape[1]}")
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    # -------------------- Logistic Regression Training --------------------
    print("\nTraining Logistic Regression with One-vs-Rest strategy...")
    clf = OneVsRestClassifier(
        LogisticRegression(
            solver=LR_SOLVER,
            class_weight=LR_CLASS_WEIGHT,
            max_iter=LR_MAX_ITER,
            random_state=RANDOM_SEED
        ),
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    print("  Training complete")

    # -------------------- Threshold Optimization --------------------
    print("\nOptimizing classification thresholds...")

    # Generate probability predictions
    y_val_proba = clf.predict_proba(X_val)
    y_test_proba = clf.predict_proba(X_test)

    # Optimize thresholds on validation set
    threshold_range = np.arange(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP)
    optimal_thresholds = optimize_thresholds_per_label(y_val, y_val_proba, threshold_range)

    print("  Optimal thresholds per label:")
    for label, thresh in zip(mlb.classes_, optimal_thresholds):
        print(f"    {label}: {thresh:.2f}")

    # Apply optimized thresholds to test predictions
    y_pred_test = np.zeros_like(y_test)
    for label_idx, thresh in enumerate(optimal_thresholds):
        y_pred_test[:, label_idx] = (y_test_proba[:, label_idx] >= thresh).astype(int)

    # -------------------- Evaluation --------------------
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)

    accuracy = accuracy_score(y_test, y_pred_test)
    hamming = hamming_loss(y_test, y_pred_test)
    f1_macro = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
    f1_micro = f1_score(y_test, y_pred_test, average="micro", zero_division=0)
    avg_precision = average_precision_score(y_test, y_test_proba, average="macro")

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"F1-Macro: {f1_macro:.4f}")
    print(f"F1-Micro: {f1_micro:.4f}")
    print(f"Average Precision (Macro): {avg_precision:.4f}")

    # -------------------- Model Persistence --------------------
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    dump(clf, "logreg_multilabel_model.joblib")
    print("  ✓ Logistic Regression model")

    np.save("optimal_thresholds.npy", np.array(optimal_thresholds))
    print("  ✓ Optimal thresholds")

    dump(mlb, "mlb_encoder.joblib")
    print("  ✓ MultiLabelBinarizer")

    # -------------------- Save Results --------------------
    class_report = classification_report(
        y_test, y_pred_test,
        target_names=mlb.classes_,
        zero_division=0,
        output_dict=True
    )

    results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            results = json.load(f)

    results[MODEL_NAME] = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "hamming_loss": float(hamming),
        "average_precision_macro": float(avg_precision),
        "classification_report": class_report,
        "model_info": {
            "embedding_model": BERT_MODEL_NAME,
            "classifier": "Logistic Regression",
            "embedding_dim": int(X_train.shape[1]),
            "solver": LR_SOLVER,
            "class_weight": LR_CLASS_WEIGHT,
            "max_iter": LR_MAX_ITER,
            "n_labels": n_labels,
            "labels": mlb.classes_.tolist()
        }
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print("  ✓ Results saved")

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
