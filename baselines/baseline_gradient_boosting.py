"""
Gradient Boosting Ensemble with Active Learning for Multi-Label Text Classification

This module implements a hybrid ensemble approach combining Logistic Regression and
multiple Gradient Boosting classifiers with active learning simulation for Portuguese
municipal text classification of discussion subjects.

Key Features:
    - Enhanced text preprocessing for Portuguese municipal domain
    - TF-IDF + BERT hybrid feature extraction
    - Ensemble of multiple GradientBoosting models with diverse hyperparameters
    - Active learning uncertainty sampling simulation
    - Adaptive ensemble weighting based on label frequency
    - Dynamic threshold optimization per label

Date: October 2025
"""

import os
import json
import re
import numpy as np
import torch
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, hamming_loss,
    classification_report, average_precision_score
)
from transformers import AutoTokenizer, AutoModel
from joblib import dump
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_SEED = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DSET_PATH = os.path.join(ROOT_DIR, "dataset_sample", "dset.json")
SPLIT_JSON_PATH = os.path.join(ROOT_DIR, "split_info.json")
RESULTS_PATH = os.path.join(ROOT_DIR, "results", "pt_results.json")
MODEL_NAME = "GradientBoosting_ActiveLearning"

# Feature extraction parameters
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 3)
BERT_MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
BERT_BATCH_SIZE = 8

# Gradient Boosting ensemble parameters
GB_N_MODELS = 3
GB_CONFIGS = [
    {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, "subsample": 0.8},
    {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 8, "subsample": 0.9},
    {"n_estimators": 80, "learning_rate": 0.15, "max_depth": 4, "subsample": 0.7}
]

# Active learning parameters
ACTIVE_LEARNING_SAMPLES = 100

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# TEXT PREPROCESSING
# ============================================================
def smart_preprocess(text):
    """
    Advanced text preprocessing with Portuguese municipal domain specialization.

    This function normalizes Portuguese municipal terminology, preserves compound terms,
    and performs intelligent tokenization for improved feature extraction.

    Args:
        text (str): Raw input text

    Returns:
        str: Preprocessed and normalized text

    Examples:
        >>> smart_preprocess("A Câmara Municipal aprovou o Decreto Lei n.º 42")
        'camara_municipal aprovou decreto_lei numero_42'
    """
    text = text.lower()

    # Municipal terminology normalization - preserve compound terms
    substitutions = {
        r'câmara\s+municipal': 'camara_municipal',
        r'assembleia\s+municipal': 'assembleia_municipal',
        r'junta\s+de\s+freguesia': 'junta_freguesia',
        r'presidente\s+da\s+câmara': 'presidente_camara',
        r'vereador\s+': 'vereador_',
        r'art\.?\s*(\d+)': r'artigo_\1',
        r'n\.?º\s*(\d+)': r'numero_\1',
        r'decreto\s+lei': 'decreto_lei',
        r'código\s+civil': 'codigo_civil',
        r'€\s*(\d+)': r'\1_euros'
    }

    for pattern, replacement in substitutions.items():
        text = re.sub(pattern, replacement, text)

    # Clean and normalize whitespace
    text = re.sub(r'[^\w\s_]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Filter tokens (minimum length 2 characters)
    words = [w for w in text.split() if len(w) >= 2]

    return ' '.join(words).strip()


def load_data(dset_path, split_files):
    """
    Load and preprocess text segments from canonical dset.json for the given split files.

    Args:
        dset_path (str): Path to dset.json
        split_files (list): List of filenames belonging to this split

    Returns:
        tuple: (texts, labels)
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
                # Quality filters: non-empty labels, minimum text length
                if topics and len(text) > 20:
                    cleaned = smart_preprocess(text)
                    # Additional filter: minimum 5 tokens
                    if len(cleaned.split()) >= 5:
                        texts.append(cleaned)
                        labels.append(topics)

    return texts, labels


# ============================================================
# BERT FEATURE EXTRACTION
# ============================================================
def get_bert_embeddings(texts, tokenizer, model, batch_size=8):
    """
    Extract BERT embeddings using mean pooling with attention mask weighting.

    Args:
        texts (list): List of text strings
        tokenizer: Pretrained BERT tokenizer
        model: Pretrained BERT model
        batch_size (int): Batch size for processing

    Returns:
        numpy.ndarray: Matrix of BERT embeddings (n_samples, embedding_dim)
    """
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            # Tokenize with padding and truncation
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            # Forward pass
            outputs = model(**encoded)

            # Mean pooling with attention mask weighting
            embeddings = outputs.last_hidden_state
            attention_mask = encoded['attention_mask'].unsqueeze(-1)
            pooled = (embeddings * attention_mask).sum(1) / attention_mask.sum(1)

            all_embeddings.append(pooled.cpu().numpy())

    return np.vstack(all_embeddings)


# ============================================================
# GRADIENT BOOSTING ENSEMBLE TRAINING
# ============================================================
def train_gradient_boosting_ensemble(X_train, y_train, configs):
    """
    Train multiple GradientBoosting models with different hyperparameters.

    Each configuration is trained as a separate OneVsRest classifier ensemble,
    providing diverse predictions for robust ensemble aggregation.

    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels (multi-label binary matrix)
        configs (list): List of hyperparameter dictionaries

    Returns:
        list: List of trained model ensembles, one per configuration
    """
    trained_models = []

    for idx, config in enumerate(configs, 1):
        print(f"  Training GB model {idx}/{len(configs)}...")

        # Train one model per label (OneVsRest approach)
        label_models = []

        for label_idx in range(y_train.shape[1]):
            # Create GradientBoosting classifier for this label
            gb_model = GradientBoostingClassifier(
                n_estimators=config["n_estimators"],
                learning_rate=config["learning_rate"],
                max_depth=config["max_depth"],
                subsample=config["subsample"],
                random_state=RANDOM_SEED + label_idx
            )

            # Train on single label column
            gb_model.fit(X_train, y_train[:, label_idx])
            label_models.append(gb_model)

        trained_models.append(label_models)

    return trained_models


def predict_gradient_boosting_ensemble(X, gb_models, n_labels):
    """
    Generate probability predictions from trained GradientBoosting ensemble.

    Args:
        X (numpy.ndarray): Feature matrix
        gb_models (list): List of label-wise model ensembles
        n_labels (int): Number of labels

    Returns:
        list: List of probability matrices, one per model configuration
    """
    predictions = []

    for model_set in gb_models:
        proba = np.zeros((X.shape[0], n_labels))

        for label_idx, label_model in enumerate(model_set):
            # Get probability for positive class
            proba[:, label_idx] = label_model.predict_proba(X)[:, 1]

        predictions.append(proba)

    return predictions


# ============================================================
# ACTIVE LEARNING SIMULATION
# ============================================================
def calculate_uncertainty_scores(predictions_list):
    """
    Calculate uncertainty scores based on prediction variance across models.

    Higher variance indicates higher uncertainty, which in active learning
    scenarios suggests samples that would benefit most from additional labeling.

    Args:
        predictions_list (list): List of prediction matrices from different models

    Returns:
        numpy.ndarray: Uncertainty score per sample
    """
    n_samples = predictions_list[0].shape[0]
    uncertainty_scores = []

    for i in range(n_samples):
        # Collect predictions from all models for this sample
        sample_predictions = [preds[i] for preds in predictions_list]

        # Calculate variance across models as uncertainty measure
        pred_variance = np.var(sample_predictions, axis=0)

        # Average variance across all labels
        uncertainty_scores.append(np.mean(pred_variance))

    return np.array(uncertainty_scores)


# ============================================================
# ADAPTIVE ENSEMBLE WEIGHTING
# ============================================================
def compute_adaptive_weights(label_frequencies, n_models):
    """
    Compute adaptive ensemble weights based on label frequency.

    Strategy:
        - Rare labels (< 10): Favor LogisticRegression (more stable)
        - Medium labels (10-50): Balanced ensemble
        - Common labels (> 50): Favor GradientBoosting (better capacity)

    Args:
        label_frequencies (numpy.ndarray): Frequency count per label
        n_models (int): Number of base models (LogReg + GB models)

    Returns:
        list: Weight vectors, one per label
    """
    weights = []

    for freq in label_frequencies:
        if freq < 10:
            # Very rare labels: favor LogisticRegression
            w = [0.7] + [0.3 / (n_models - 1)] * (n_models - 1)
        elif freq < 50:
            # Medium frequency: balanced ensemble
            w = [1.0 / n_models] * n_models
        else:
            # Common labels: equal weighting
            w = [1.0 / n_models] * n_models

        weights.append(w)

    return weights


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================
def optimize_thresholds(y_true, y_proba, label_frequencies):
    """
    Optimize classification thresholds per label using validation data.

    Uses label-specific threshold ranges based on frequency to handle
    class imbalance effectively.

    Args:
        y_true (numpy.ndarray): True labels (binary matrix)
        y_proba (numpy.ndarray): Predicted probabilities
        label_frequencies (numpy.ndarray): Training set label frequencies

    Returns:
        list: Optimal threshold per label
    """
    n_labels = y_true.shape[1]
    optimal_thresholds = []

    for label_idx in range(n_labels):
        best_f1 = 0.0
        best_thresh = 0.5

        # Adaptive threshold search range based on label frequency
        freq = label_frequencies[label_idx]
        if freq < 5:
            thresh_range = np.arange(0.01, 0.4, 0.01)
        elif freq < 20:
            thresh_range = np.arange(0.05, 0.7, 0.01)
        else:
            thresh_range = np.arange(0.1, 0.9, 0.01)

        # Find threshold that maximizes F1-score
        for thresh in thresh_range:
            y_pred = (y_proba[:, label_idx] >= thresh).astype(int)
            f1 = f1_score(y_true[:, label_idx], y_pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        optimal_thresholds.append(best_thresh)

    return optimal_thresholds


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    """
    Main training and evaluation pipeline.
    """
    print("\n" + "="*70)
    print("GRADIENT BOOSTING ENSEMBLE WITH ACTIVE LEARNING")
    print("="*70 + "\n")

    # -------------------- Data Loading --------------------
    print("Loading data...")
    with open(SPLIT_JSON_PATH, "r", encoding="utf-8") as f:
        split_info = json.load(f)

    train_texts, train_labels = load_data(DSET_PATH, split_info["train_files"])
    val_texts, val_labels = load_data(DSET_PATH, split_info["val_files"])
    test_texts, test_labels = load_data(DSET_PATH, split_info["test_files"])

    print(f"  Train: {len(train_texts)} segments")
    print(f"  Validation: {len(val_texts)} segments")
    print(f"  Test: {len(test_texts)} segments")

    # -------------------- Label Encoding --------------------
    print("\nEncoding labels...")
    all_labels = train_labels + val_labels + test_labels
    mlb = MultiLabelBinarizer()
    mlb.fit(all_labels)

    y_train = mlb.transform(train_labels)
    y_val = mlb.transform(val_labels)
    y_test = mlb.transform(test_labels)

    n_labels = len(mlb.classes_)
    print(f"  Number of unique labels: {n_labels}")

    # -------------------- Feature Extraction --------------------
    print("\nExtracting TF-IDF features...")
    tfidf = TfidfVectorizer(
        ngram_range=TFIDF_NGRAM_RANGE,
        max_features=TFIDF_MAX_FEATURES,
        min_df=2,
        max_df=0.8,
        sublinear_tf=True,
        norm='l2'
    )

    X_tfidf_train = tfidf.fit_transform(train_texts)
    X_tfidf_val = tfidf.transform(val_texts)
    X_tfidf_test = tfidf.transform(test_texts)
    print(f"  TF-IDF features: {X_tfidf_train.shape[1]}")

    print("\nExtracting BERT embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME).to(device)
    bert_model.eval()

    X_bert_train = get_bert_embeddings(train_texts, tokenizer, bert_model, BERT_BATCH_SIZE)
    X_bert_val = get_bert_embeddings(val_texts, tokenizer, bert_model, BERT_BATCH_SIZE)
    X_bert_test = get_bert_embeddings(test_texts, tokenizer, bert_model, BERT_BATCH_SIZE)
    print(f"  BERT features: {X_bert_train.shape[1]}")

    # Combine features (dense matrix for GradientBoosting)
    print("\nCombining features...")
    X_train = np.hstack([X_tfidf_train.toarray(), X_bert_train])
    X_val = np.hstack([X_tfidf_val.toarray(), X_bert_val])
    X_test = np.hstack([X_tfidf_test.toarray(), X_bert_test])
    print(f"  Total feature dimensions: {X_train.shape[1]}")

    # -------------------- Model Training --------------------
    print("\nTraining Logistic Regression baseline...")
    logistic_model = OneVsRestClassifier(
        LogisticRegression(
            solver='liblinear',
            class_weight='balanced',
            C=1.0,
            max_iter=2000,
            random_state=RANDOM_SEED
        ),
        n_jobs=-1
    )
    logistic_model.fit(X_train, y_train)

    print("\nTraining Gradient Boosting ensemble...")
    gb_models = train_gradient_boosting_ensemble(X_train, y_train, GB_CONFIGS)

    # -------------------- Active Learning Simulation --------------------
    print("\nSimulating active learning with uncertainty sampling...")

    # Get validation predictions from all models
    logistic_val_proba = logistic_model.predict_proba(X_val)
    gb_val_predictions = predict_gradient_boosting_ensemble(gb_models, X_val, n_labels)

    # Calculate uncertainty scores
    all_val_predictions = [logistic_val_proba] + gb_val_predictions
    uncertainty_scores = calculate_uncertainty_scores(all_val_predictions)

    # Select high-uncertainty samples
    high_uncertainty_indices = np.argsort(uncertainty_scores)[-ACTIVE_LEARNING_SAMPLES:]
    print(f"  Selected {len(high_uncertainty_indices)} high-uncertainty samples")

    # -------------------- Ensemble Prediction --------------------
    print("\nGenerating ensemble predictions...")

    # Compute adaptive weights based on label frequency
    label_frequencies = np.sum(y_train, axis=0)
    weights = compute_adaptive_weights(label_frequencies, n_models=1 + GB_N_MODELS)

    # Get test predictions
    logistic_test_proba = logistic_model.predict_proba(X_test)
    gb_test_predictions = predict_gradient_boosting_ensemble(gb_models, X_test, n_labels)

    # Weighted ensemble combination
    ensemble_test_proba = np.zeros_like(logistic_test_proba)
    for label_idx in range(n_labels):
        w = weights[label_idx]
        ensemble_test_proba[:, label_idx] = (
            w[0] * logistic_test_proba[:, label_idx] +
            sum(w[i+1] * gb_test_predictions[i][:, label_idx]
                for i in range(GB_N_MODELS))
        )

    # -------------------- Threshold Optimization --------------------
    print("\nOptimizing classification thresholds...")

    # Generate validation ensemble predictions
    ensemble_val_proba = np.zeros_like(logistic_val_proba)
    for label_idx in range(n_labels):
        w = weights[label_idx]
        ensemble_val_proba[:, label_idx] = (
            w[0] * logistic_val_proba[:, label_idx] +
            sum(w[i+1] * gb_val_predictions[i][:, label_idx]
                for i in range(GB_N_MODELS))
        )

    # Optimize thresholds on validation set
    optimal_thresholds = optimize_thresholds(y_val, ensemble_val_proba, label_frequencies)

    # Apply optimized thresholds to test predictions
    y_pred_test = np.zeros_like(y_test)
    for label_idx, thresh in enumerate(optimal_thresholds):
        y_pred_test[:, label_idx] = (ensemble_test_proba[:, label_idx] >= thresh).astype(int)

    # -------------------- Evaluation --------------------
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)

    accuracy = accuracy_score(y_test, y_pred_test)
    hamming = hamming_loss(y_test, y_pred_test)
    f1_macro = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
    f1_micro = f1_score(y_test, y_pred_test, average="micro", zero_division=0)
    avg_precision = average_precision_score(y_test, ensemble_test_proba, average="macro")

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"F1-macro: {f1_macro:.4f}")
    print(f"F1-micro: {f1_micro:.4f}")
    print(f"Average Precision (macro): {avg_precision:.4f}")

    # -------------------- Model Persistence --------------------
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)

    dump(tfidf, 'gradient_boosting_tfidf_vectorizer.joblib')
    print("  ✓ TF-IDF vectorizer")

    dump(mlb, 'gradient_boosting_mlb_encoder.joblib')
    print("  ✓ MultiLabelBinarizer")

    np.save('gradient_boosting_optimal_thresholds.npy', np.array(optimal_thresholds))
    print("  ✓ Optimal thresholds")

    np.save('gradient_boosting_adaptive_weights.npy', np.array(weights))
    print("  ✓ Adaptive weights")

    dump(logistic_model, 'gradient_boosting_logistic_model.joblib')
    print("  ✓ Logistic Regression model")

    dump(gb_models, 'gradient_boosting_gb_models.joblib')
    print("  ✓ Gradient Boosting models")

    # Save model metadata
    model_info = {
        "model_name": MODEL_NAME,
        "description": "Gradient Boosting ensemble with active learning simulation",
        "architecture": {
            "base_models": ["LogisticRegression", "GradientBoosting_x3"],
            "n_labels": n_labels,
            "feature_dimensions": X_train.shape[1],
            "tfidf_features": X_tfidf_train.shape[1],
            "bert_features": X_bert_train.shape[1]
        },
        "training": {
            "random_seed": RANDOM_SEED,
            "gb_configurations": GB_CONFIGS,
            "active_learning_samples": len(high_uncertainty_indices)
        },
        "performance": {
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_micro": float(f1_micro),
            "hamming_loss": float(hamming),
            "average_precision_macro": float(avg_precision)
        },
        "labels": mlb.classes_.tolist()
    }

    with open('gradient_boosting_model_info.json', 'w') as f:
        json.dump(model_info, f, indent=4)
    print("  ✓ Model metadata")

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
        "classification_report": class_report
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
