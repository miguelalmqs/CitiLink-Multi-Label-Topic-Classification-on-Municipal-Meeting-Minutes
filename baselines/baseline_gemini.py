"""
Gemini API Baseline for Multi-Label Text Classification

This module implements a baseline using Google's Gemini API for Portuguese
municipal text classification with few-shot learning. The model learns from
example annotations and applies that knowledge to new texts.

Key Features:
    - Few-shot learning with training examples
    - Gemini 2.5 Pro (best available model)
    - Robust error handling and rate limit management
    - Checkpoint system for resumable execution
    - Zero-shot classification capability

Requirements:
    - google-generativeai library (pip install google-generativeai)
    - Gemini API key (free tier available at https://aistudio.google.com/app/apikey)


Date: October 2025
"""

import os
import json
import time
import re
import numpy as np
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    f1_score, accuracy_score, hamming_loss,
    classification_report, average_precision_score
)

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai not installed")
    print("Install with: pip install google-generativeai")
    exit(1)


# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DSET_PATH = os.path.join(ROOT_DIR, "dataset_sample", "dset.json")
SPLIT_JSON_PATH = os.path.join(ROOT_DIR, "split_info.json")
RESULTS_PATH = os.path.join(ROOT_DIR, "results", "pt_results.json")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "results", "gemini_checkpoints")
PREDICTIONS_PATH = os.path.join(ROOT_DIR, "results", "gemini_fewshot_predictions.json")

# Model configuration
MODEL_NAME = "gemini-2.5-pro"  # Best Gemini model
FALLBACK_MODELS = ["gemini-1.5-pro", "gemini-pro"]

# Few-shot learning configuration
N_FEW_SHOT_EXAMPLES = 5

# API rate limiting (adjust based on your tier)
REQUEST_DELAY = 0.1  # seconds between requests (600 req/min for paid tier)
MAX_RETRIES = 5

# Checkpoint configuration
CHECKPOINT_INTERVAL = 50  # Save every N samples
MIN_TEXT_LENGTH = 30  # Minimum text length for processing


# ============================================================
# DATA LOADING
# ============================================================
def load_data(dset_path, split_files):
    """
    Load text segments from canonical dset.json for the given split files.

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
                # Quality filter: non-empty labels and minimum length
                if topics and len(text) > MIN_TEXT_LENGTH:
                    texts.append(text)
                    labels.append(topics)

    return texts, labels


# ============================================================
# FEW-SHOT EXAMPLE SELECTION
# ============================================================
def select_few_shot_examples(train_texts, train_labels, n_examples=5):
    """
    Select diverse examples from training set for few-shot learning.

    Selects examples with different label combinations to provide
    the model with varied patterns to learn from.

    Args:
        train_texts (list): Training texts
        train_labels (list): Training labels
        n_examples (int): Number of examples to select

    Returns:
        list: Selected examples with texts and labels
    """
    import random

    examples = []
    seen_label_combos = set()

    # Shuffle for random selection
    indices = list(range(len(train_texts)))
    random.shuffle(indices)

    for idx in indices:
        label_combo = tuple(sorted(train_labels[idx]))

        # Add unique label combinations
        if label_combo not in seen_label_combos and len(train_labels[idx]) > 0:
            examples.append({
                "text": train_texts[idx],
                "labels": train_labels[idx]
            })
            seen_label_combos.add(label_combo)

            if len(examples) >= n_examples:
                break

    return examples


# ============================================================
# PROMPT ENGINEERING
# ============================================================
def create_few_shot_prompt(text, label_list, examples):
    """
    Create few-shot prompt with training examples.

    Constructs a structured prompt that includes:
    - Available labels
    - Example classifications
    - Classification instructions
    - Target text to classify

    Args:
        text (str): Text to classify
        label_list (list): List of valid labels
        examples (list): Few-shot training examples

    Returns:
        str: Formatted prompt for Gemini API
    """
    labels_str = ", ".join([f'"{label}"' for label in label_list])

    # Build examples section
    examples_str = ""
    for i, example in enumerate(examples, 1):
        example_labels = ", ".join(example["labels"])
        # Truncate long texts
        example_text = example["text"][:300]
        if len(example["text"]) > 300:
            example_text += "..."

        examples_str += f"""
**Example {i}**:
Text: {example_text}
Topics: {example_labels}
"""

    prompt = f"""You are an expert in classifying Portuguese municipal council meeting minutes.

**Available Topics (you MUST choose ONLY from this list)**:
{labels_str}

**Examples from Training Data**:
{examples_str}

**Now, classify this new text**:
{text}

**Instructions**:
1. Read the text carefully
2. Identify ALL topics that are discussed or mentioned (like in the examples above)
3. Return ONLY the topic names, separated by commas
4. Use the EXACT names from the available topics list
5. If multiple topics apply, list all of them
6. If no topics clearly apply, return "Nenhum"

**Your Response** (topic names only, comma-separated):
"""

    return prompt


# ============================================================
# GEMINI API CLASSIFICATION
# ============================================================
def classify_with_gemini(text, label_list, model, few_shot_examples, max_retries=MAX_RETRIES):
    """
    Classify text using Gemini API with few-shot learning.

    Implements robust error handling including:
    - Rate limit management with exponential backoff
    - Quota error detection and recovery
    - Response parsing and validation

    Args:
        text (str): Text to classify
        label_list (list): List of valid labels
        model: Gemini model instance
        few_shot_examples (list): Training examples for few-shot learning
        max_retries (int): Maximum retry attempts

    Returns:
        list: Predicted labels
    """
    prompt = create_few_shot_prompt(text, label_list, few_shot_examples)

    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            predicted_text = response.text.strip()

            # Handle empty/negative responses
            if predicted_text.lower() == "nenhum":
                return []

            # Parse comma-separated labels
            predicted_labels = [
                label.strip().strip('"').strip("'")
                for label in predicted_text.split(",")
            ]

            # Filter to only valid labels
            valid_predictions = [
                label for label in predicted_labels
                if label in label_list
            ]

            return valid_predictions

        except Exception as e:
            error_msg = str(e)

            # Handle rate limit errors
            if '429' in error_msg or 'quota' in error_msg.lower() or 'rate limit' in error_msg.lower():
                if attempt < max_retries - 1:
                    # Extract wait time from error message
                    wait_time = 60  # Default
                    match = re.search(r'retry in (\d+\.?\d*)', error_msg.lower())
                    if match:
                        wait_time = float(match.group(1)) + 5  # Add buffer

                    print(f"  Rate limit hit. Waiting {wait_time:.0f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  Error: Quota exceeded after {max_retries} attempts")
                    return []

            # Exponential backoff for other errors
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2  # 2s, 4s, 8s, 16s
                print(f"  Attempt {attempt + 1} failed. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  Error: Failed after {max_retries} attempts")
                return []

    return []


# ============================================================
# CHECKPOINT MANAGEMENT
# ============================================================
def save_checkpoint(split_name, predictions, index):
    """Save checkpoint of predictions for resumable execution."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{split_name}_checkpoint.json")

    checkpoint_data = {
        "split": split_name,
        "last_index": index,
        "predictions": predictions,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)


def load_checkpoint(split_name):
    """Load checkpoint if exists."""
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{split_name}_checkpoint.json")

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"  Loaded checkpoint: {data['last_index'] + 1} samples completed")
            return data["predictions"], data["last_index"] + 1

    return [], 0


# ============================================================
# BATCH PREDICTION
# ============================================================
def run_predictions_on_split(texts, labels, split_name, model, label_list,
                             few_shot_examples, checkpoint_interval=CHECKPOINT_INTERVAL):
    """
    Run Gemini predictions on a data split with checkpoints.

    Args:
        texts (list): Input texts
        labels (list): Ground truth labels (for validation)
        split_name (str): Name of data split
        model: Gemini model instance
        label_list (list): List of valid labels
        few_shot_examples (list): Training examples
        checkpoint_interval (int): Save checkpoint every N samples

    Returns:
        list: Predicted labels for all texts
    """
    print(f"\nRunning predictions on {split_name} ({len(texts)} samples)...")

    # Try to resume from checkpoint
    predicted_labels, start_index = load_checkpoint(split_name)

    if start_index > 0:
        print(f"  Resuming from sample {start_index + 1}")

    failed_count = 0

    for i in range(start_index, len(texts)):
        # Progress update
        if (i + 1) % 10 == 0:
            progress = (i + 1) / len(texts) * 100
            print(f"  Progress: {i + 1}/{len(texts)} ({progress:.1f}%)")

        # Classify with Gemini
        predictions = classify_with_gemini(
            texts[i], label_list, model, few_shot_examples
        )
        predicted_labels.append(predictions)

        # Track failures
        if not predictions and labels[i]:
            failed_count += 1

        # Save checkpoint
        if (i + 1) % checkpoint_interval == 0:
            save_checkpoint(split_name, predicted_labels, i)

        # Rate limiting
        time.sleep(REQUEST_DELAY)

    # Final checkpoint
    save_checkpoint(split_name, predicted_labels, len(texts) - 1)

    print(f"  Completed! Failed: {failed_count}/{len(texts)}")

    return predicted_labels


# ============================================================
# EVALUATION
# ============================================================
def evaluate_predictions(y_true, y_pred, split_name):
    """
    Evaluate predictions and print metrics.

    Args:
        y_true (numpy.ndarray): Ground truth binary labels
        y_pred (numpy.ndarray): Predicted binary labels
        split_name (str): Name of data split

    Returns:
        dict: Evaluation metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    hamming = hamming_loss(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Average precision
    try:
        avg_precision_macro = average_precision_score(y_true, y_pred, average="macro")
        avg_precision_micro = average_precision_score(y_true, y_pred, average="micro")
    except:
        avg_precision_macro = 0.0
        avg_precision_micro = 0.0

    print(f"\n{'='*70}")
    print(f"GEMINI BASELINE - {split_name} RESULTS")
    print(f"{'='*70}")
    print(f"Accuracy:              {accuracy:.4f}")
    print(f"Hamming Loss:          {hamming:.4f}")
    print(f"F1-Macro:              {f1_macro:.4f}")
    print(f"F1-Micro:              {f1_micro:.4f}")
    print(f"F1-Weighted:           {f1_weighted:.4f}")
    if avg_precision_macro > 0:
        print(f"Avg Precision (Macro): {avg_precision_macro:.4f}")
    print(f"{'='*70}")

    return {
        "accuracy": accuracy,
        "hamming_loss": hamming,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
        "average_precision_macro": avg_precision_macro,
        "average_precision_micro": avg_precision_micro
    }


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    """
    Main training and evaluation pipeline.
    """
    print("\n" + "="*70)
    print("GEMINI API BASELINE - FEW-SHOT MULTI-LABEL CLASSIFICATION")
    print("="*70 + "\n")

    # -------------------- API Configuration --------------------
    print("Configuring Gemini API...")

    # Get API key from environment variable
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("\n" + "="*70)
        print("ERROR: GEMINI API KEY REQUIRED")
        print("="*70)
        print("\nGemini API is FREE but requires an API key.")
        print("\nGet your FREE API key:")
        print("  1. Go to: https://aistudio.google.com/app/apikey")
        print("  2. Sign in with Google account")
        print("  3. Click 'Create API Key'")
        print("  4. Copy the key")
        print("\nSetup: Set environment variable")
        print("  export GOOGLE_API_KEY='your-api-key-here'")
        print("\n" + "="*70)
        exit(1)

    genai.configure(api_key=api_key)

    # Configure model with fallback
    print(f"Configuring {MODEL_NAME}...")
    model = None
    model_name = None

    for name in [MODEL_NAME] + FALLBACK_MODELS:
        try:
            model = genai.GenerativeModel(name)
            model_name = name
            print(f"  Using model: {model_name}")
            break
        except Exception as e:
            print(f"  {name} not available, trying fallback...")
            continue

    if not model:
        print("Error: No Gemini model available")
        exit(1)

    # -------------------- Data Loading --------------------
    print("\nLoading data...")
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
    label_list = mlb.classes_.tolist()
    print(f"  Number of unique labels: {n_labels}")

    # -------------------- Few-Shot Example Selection --------------------
    print(f"\nSelecting {N_FEW_SHOT_EXAMPLES} few-shot examples...")
    few_shot_examples = select_few_shot_examples(
        train_texts, train_labels, N_FEW_SHOT_EXAMPLES
    )

    for i, example in enumerate(few_shot_examples, 1):
        print(f"  Example {i}: {len(example['text'])} chars, "
              f"Labels: {', '.join(example['labels'])}")

    # -------------------- Run Predictions --------------------
    print("\n" + "="*70)
    print("RUNNING PREDICTIONS")
    print("="*70)
    print("\nNote: You can safely interrupt (Ctrl+C) and resume later.")
    print(f"Checkpoints saved every {CHECKPOINT_INTERVAL} samples.\n")

    try:
        test_predictions = run_predictions_on_split(
            test_texts, test_labels, "TEST", model, label_list,
            few_shot_examples, CHECKPOINT_INTERVAL
        )
    except KeyboardInterrupt:
        print("\n\nProcess interrupted! Progress saved in checkpoints.")
        print("Run again to resume.")
        exit(0)

    # -------------------- Evaluation --------------------
    y_test_pred = mlb.transform(test_predictions)
    test_metrics = evaluate_predictions(y_test, y_test_pred, "TEST")

    # Per-label report
    class_report = classification_report(
        y_test, y_test_pred,
        target_names=mlb.classes_,
        zero_division=0,
        output_dict=True
    )

    print("\nPer-Label Performance:")
    print("-" * 70)
    for label in mlb.classes_:
        if label in class_report:
            m = class_report[label]
            print(f"{label:30s} | P: {m['precision']:.3f} | "
                  f"R: {m['recall']:.3f} | F1: {m['f1-score']:.3f}")

    # -------------------- Save Results --------------------
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    # Save metrics
    results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            results = json.load(f)

    results["Gemini_FewShot"] = {
        "accuracy": float(test_metrics["accuracy"]),
        "f1_macro": float(test_metrics["f1_macro"]),
        "f1_micro": float(test_metrics["f1_micro"]),
        "f1_weighted": float(test_metrics["f1_weighted"]),
        "hamming_loss": float(test_metrics["hamming_loss"]),
        "average_precision_macro": float(test_metrics["average_precision_macro"]),
        "classification_report": class_report,
        "model": model_name,
        "few_shot_examples": N_FEW_SHOT_EXAMPLES,
        "method": "few-shot learning"
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=4)
    print(f"  ✓ Metrics saved to: {RESULTS_PATH}")

    # Save detailed predictions
    detailed_predictions = {
        "few_shot_examples": [
            {
                "text": ex["text"][:200] + "..." if len(ex["text"]) > 200 else ex["text"],
                "labels": ex["labels"]
            }
            for ex in few_shot_examples
        ],
        "test": [
            {
                "sample_id": i,
                "text": text[:200] + "..." if len(text) > 200 else text,
                "ground_truth": true_labels,
                "predicted": pred_labels,
                "correct": set(true_labels) == set(pred_labels)
            }
            for i, (text, true_labels, pred_labels) in
            enumerate(zip(test_texts, test_labels, test_predictions))
        ]
    }

    with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(detailed_predictions, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Predictions saved to: {PREDICTIONS_PATH}")

    # Clean up checkpoints
    import shutil
    if os.path.exists(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR)
        print(f"  ✓ Cleaned up checkpoints")

    # -------------------- Summary --------------------
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nModel: {model_name}")
    print(f"Method: Few-shot learning ({N_FEW_SHOT_EXAMPLES} examples)")
    print(f"\nTest Results:")
    print(f"  F1-Macro:  {test_metrics['f1_macro']:.4f}")
    print(f"  F1-Micro:  {test_metrics['f1_micro']:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
