# CitiLink: Multi-Label Topic Classification on Municipal Meeting Minutes

[![License: CC-BY-ND 4.0](https://img.shields.io/badge/License-CC--BY--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nd/4.0/)
[![Python 3.10](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

This repository presents a multi-label topic classificationi taks, presenting scripts and baselines for **multi-label topic classification** of subjects of discussion extracted from Portuguese Municipal Meeting Minutes. Each segment is classified into one or more of **22 administrative topic categories** using a fine-tuned BERTimbau model and seven baseline approaches.

> **🎯 Try the topic classification model now**: Test the models interactively at [https://huggingface.co/spaces/liaad/CitiLink-Theme-Generation-and-Segment-Level-Summarization-Demo](https://huggingface.co/spaces/liaad/CitiLink-Multi-Label-Topic-Classification-Demo)

---

## Table of Contents

1. [Description](#description)
2. [Project Status](#project-status)
3. [Technology Stack](#technology-stack)
4. [Dependencies](#dependencies)
5. [Installation](#installation)
6. [Repository Structure](#repository-structure)
7. [Usage](#usage)
   - [Main Model — Fine-Tuned BERTimbau](#main-model--fine-tuned-bertimbau)
   - [Baselines](#baselines)
   - [Evaluation](#evaluation)
8. [Dataset](#dataset)
9. [Architecture](#architecture)
   - [Classification Pipeline](#classification-pipeline)
   - [Dynamic Threshold Optimization](#dynamic-threshold-optimization)
10. [Evaluation Metrics](#evaluation-metrics)
11. [Experimental Results](#experimental-results)
12. [Known Issues](#known-issues)
13. [License](#license)
14. [Resources](#resources)
15. [Acknowledgments](#acknowledgments)
16. [Citation](#citation)

---

## Description

The system addresses **multi-label topic classification** for Portuguese municipal meeting minutes: given a text segment from a municipal meeting minute, assign one or more topic labels from a fixed taxonomy of 22 Portuguese administrative categories.

### Key Features

- **BERTimbau Fine-Tuning**: End-to-end fine-tuning of `neuralmind/bert-large-portuguese-cased` with a multi-label classification head for Portuguese municipal domain text.
- **Dynamic Per-Label Threshold Optimization**: For each of the 22 labels independently, the optimal decision threshold is found via grid search (0.1–0.9, step 0.05) on the validation set, maximising per-label F1.
- **Seven Baselines**: Decision Tree, SVM, Logistic Regression, Gradient Boosting, and Gemini API baselines with TF-IDF, BERT embeddings, and hybrid feature representations.
- **Domain-Specific Preprocessing**: A `smart_preprocess` function normalizes Portuguese municipal terminology (e.g., "câmara municipal" → "camara_municipal", legal reference normalization).
- **Reproducible Experiments**: All code is made available to ensure all presented results are reproducible.

---

## Project Status

✅ The classification pipeline is **fully implemented and validated** for research use. The codebase is actively maintained to ensure reproducibility of the published results.

- **Dataset**: Available in `dataset_sample/dset.json` (canonical format)
- **Train/Val/Test Split**: Available in `split_info.json` (temporal split strategy)
- **Full Dataset**: Available through the [CitiLink-Summ repository](https://github.com/INESCTEC/citilink-summ)

---

## Technology Stack

**Language**: Python

**Core Frameworks**:
- **PyTorch**: Deep learning backend for tensor computations, gradient calculation, and GPU acceleration during BERTimbau fine-tuning and embedding extraction.
- **Hugging Face Transformers**: Used for loading BERTimbau models, tokenization, and the `Trainer` API for fine-tuning.
- **scikit-learn**: Used for traditional ML classifiers (SVM, Decision Tree, Logistic Regression, Gradient Boosting), `MultiLabelBinarizer`, TF-IDF vectorization, and evaluation metrics.

**Key Libraries**:
- `numpy`: Array operations for embedding manipulation, threshold optimization, and label encoding.
- `scipy`: Sparse matrix operations for hybrid feature concatenation (TF-IDF + BERT).
- `joblib`: Model serialization for trained classifiers.
- `json`: Parsing the hierarchical JSON structure of the municipal minutes dataset.
- `os`: Directory and path management for model checkpoints and output files.

---

## Dependencies

### Core Dependencies

- **`transformers`** (>=4.30.0) — Load BERTimbau models, handle tokenization, and run the fine-tuning `Trainer`.
- **`torch`** (>=2.0.0) — PyTorch backend for tensor operations, gradient accumulation, and GPU-accelerated training.
- **`scikit-learn`** (>=1.3.0) — Traditional ML classifiers, `MultiLabelBinarizer`, TF-IDF, and evaluation metrics.
- **`numpy`** (>=1.24.0) — Array operations for embeddings and threshold optimization.
- **`pandas`** (>=2.0.0) — Tabular data manipulation for results analysis.
- **`scipy`** (>=1.10.0) — Sparse matrix operations for hybrid feature concatenation.
- **`joblib`** (>=1.3.0) — Model serialization.

### Optional Dependencies

- **`google-generativeai`** — Required only for the Gemini API baseline. Requires an API key ([get one here](https://aistudio.google.com/app/apikey)).

### Installing Dependencies

```bash
pip install transformers torch scikit-learn numpy pandas scipy joblib
```

For PyTorch with CUDA support (match your NVIDIA driver version):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Installation

### Prerequisites
- Python 3.10 or higher
- CUDA-capable GPU recommended (at least 8 GB VRAM for BERTimbau Large fine-tuning; CPU possible for baselines)
- At least 8 GB system RAM

### Setup Steps

1. **Navigate to this sub-project**
```bash
cd TextClassification_ResourcePaper
```

2. **Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import torch; from transformers import AutoTokenizer; print('CUDA Available:', torch.cuda.is_available()); AutoTokenizer.from_pretrained('neuralmind/bert-large-portuguese-cased')"
```

---

## Repository Structure

```
TextClassification_ResourcePaper/
├── README.md                                        # This file
├── requirements.txt                                 # Python dependencies
├── split_info.json                                  # Train/Val/Test file-level split (temporal, 72/23/24)
├── TRAIN_bertimbau_finetune.py                      # Main model: BERTimbau Large fine-tune
│
├── dataset_sample/
│   └── dset.json                                    # Full dataset in canonical format
│
├── baselines/                                       # ── BASELINE MODELS ──
│   ├── run_all_baselines.sh                         # Run all baselines sequentially
│   ├── baseline_DT_tfidf.py                        # Decision Tree + TF-IDF
│   ├── baseline_SVM_tfidf.py                       # SVM + TF-IDF
│   ├── baseline_SVM_bert_embeddings.py             # SVM + BERTimbau Large embeddings
│   ├── baseline_LogReg_bert.py                     # Logistic Regression + BERTimbau CLS
│   ├── baseline_SVM_hybrid.py                      # SVM + Hybrid (TF-IDF + BERTimbau)
│   ├── baseline_gradient_boosting.py               # Gradient Boosting Ensemble + Active Learning
│   └── baseline_gemini.py                           # Gemini API few-shot (exploratory)
│
├── evaluation/                                      # ── EVALUATION ──
│   ├── analyze_results.py                           # Results analysis and ranking
│   └── statistical_significance_analysis.py         # Statistical significance tests
│
└── results/
    └── pt_results.json                              # Evaluation metrics for all models
```

---

## Usage

### Main Model — Fine-Tuned BERTimbau

The main model fine-tunes `neuralmind/bert-large-portuguese-cased` end-to-end for multi-label classification.

**Train & evaluate:**
```bash
python TRAIN_bertimbau_finetune.py
```

This fine-tunes BERTimbau Large, optimizes per-label thresholds on the validation set, evaluates on the test set, and saves metrics to `results/pt_results.json`.

---

### Baselines

#### Run All Baselines

```bash
cd baselines
bash run_all_baselines.sh
```

This runs all 6 baselines sequentially (Gemini is commented out by default as it requires an API key). Each baseline appends its metrics to `results/pt_results.json`.

#### Run Individual Baselines

```bash
python baselines/baseline_DT_tfidf.py
python baselines/baseline_SVM_tfidf.py
python baselines/baseline_SVM_bert_embeddings.py
python baselines/baseline_LogReg_bert.py
python baselines/baseline_SVM_hybrid.py
python baselines/baseline_gradient_boosting.py
python baselines/baseline_gemini.py  # Requires GEMINI_API_KEY
```

---

### Evaluation

**Results analysis and ranking:**
```bash
python evaluation/analyze_results.py
```
Generates a ranked comparison of all models and saves a detailed analysis to `results/final_analysis.json`.

**Statistical significance tests:**
```bash
python evaluation/statistical_significance_analysis.py
```

---

### Key Hyperparameters

#### Main Model — Fine-Tuned BERTimbau Large

| Parameter | Value |
|-----------|-------|
| Base Model | `neuralmind/bert-large-portuguese-cased` |
| Max Sequence Length | 512 tokens |
| Batch Size | 16 |
| Epochs | 10 |
| Learning Rate | 5e-5 |
| Weight Decay | 0.01 |
| Warmup Ratio | 0.1 |
| Threshold Range | 0.1–0.9 (step 0.05) |

#### Baseline — Gradient Boosting + Active Learning

| Parameter | Value |
|-----------|-------|
| TF-IDF n-gram Range | (1, 3) |
| TF-IDF Max Features | 10,000 |
| BERT Model | `neuralmind/bert-base-portuguese-cased` |
| GB Ensemble Size | 3 models |
| Active Learning Samples | 100 |

#### Baseline — SVM + TF-IDF

| Parameter | Value |
|-----------|-------|
| TF-IDF n-gram Range | (1, 2) |
| TF-IDF Max Features | 10,000 |
| SVM Kernel | Linear |
| Class Weight | Balanced |

---

## Dataset

### Overview

This dataset contains annotated subjects of discussion from Portuguese municipal meeting minutes, each labeled with one or more of 22 administrative topic categories.

### Dataset Statistics

| Attribute | Value |
|-----------|-------|
| Total Documents (Minutes) | 119 |
| Total Segments (Agenda Items) | ~2,848 |
| Number of Municipalities | 7 |
| Number of Labels (Topics) | 22 |
| Avg. Labels per Segment | ~1.5 |
| Administrative Term | 2021–2024 |
| Train / Val / Test Split | 72 / 23 / 24 documents |
| Split Strategy | Temporal |

### Municipalities

Alandroal, Campomaior, Covilhã, Fundão, Guimarães, Porto

### Label Taxonomy

The 22 administrative topic categories:

| # | Label (Portuguese) |
|---|-------------------|
| 1 | Administração Geral, Finanças e Recursos Humanos |
| 2 | Obras Públicas |
| 3 | Obras Particulares |
| 4 | Cultura |
| 5 | Educação e Formação Profissional |
| 6 | Ambiente |
| 7 | Ação Social |
| 8 | Desporto |
| 9 | Trânsito e Transportes |
| 10 | Saúde |
| 11 | Ordenamento do Território |
| 12 | Habitação |
| 13 | Comunicação e Relações Públicas |
| 14 | Cooperação Externa e Relações Internacionais |
| 15 | Ciência |
| 16 | Atividades Económicas |
| 17 | Energia e Telecomunicações |
| 18 | Património |
| 19 | Proteção Civil |
| 20 | Polícia Municipal |
| 21 | Proteção Animal |
| 22 | Outros |

Labels are encoded with `MultiLabelBinarizer` fit on all splits combined to ensure a consistent label space.

### Dataset Structure

```json
{
  "municipalities": [
    {
      "municipality": "Alandroal",
      "minutes": [
        {
          "minute_id": "Alandroal_cm_002_2021-10-22",
          "agenda_items": [
            {
              "text": "Complete text of the subject of discussion",
              "topics": ["Administração Geral, Finanças e Recursos Humanos"],
              "theme": "Aprovação da ata anterior",
              "topic_ids": [1]
            }
          ]
        }
      ]
    }
  ]
}
```

### Data Files

- [dataset_sample/dset.json](dataset_sample/dset.json) — Full dataset in canonical format
- [split_info.json](split_info.json) — Train/validation/test document-level split (temporal strategy)

### Using the Dataset

For instructions about dataset usage, consult the [dataset GitHub repository](https://github.com/INESCTEC/citilink-summ). The full dataset can be accessed through it.

---

## Architecture

### Classification Pipeline

The classification pipeline supports two approaches: (1) end-to-end fine-tuning of BERTimbau Large and (2) traditional ML classifiers with various feature representations.

```text
┌────────────────────────────────────────────────────────┐
│              Raw Municipal Minutes (JSON)               │
└───────────────────────────┬────────────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   Data Loading & Split    │
              │  (dset.json + split_info) │
              └─────────────┬─────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                                   │
┌─────────▼──────────┐           ┌────────────▼───────────┐
│  Text Preprocessing│           │   BERTimbau Fine-Tune  │
│  (clean_text or    │           │  (End-to-end training  │
│   smart_preprocess)│           │   with classification  │
└─────────┬──────────┘           │   head)                │
          │                      └────────────┬───────────┘
          │                                   │
┌─────────▼──────────┐           ┌────────────▼───────────┐
│ Feature Extraction │           │  Sigmoid Probabilities │
│ ┌────────────────┐ │           └────────────┬───────────┘
│ │ TF-IDF         │ │                        │
│ │ BERT Embeddings│ │                        │
│ │ Hybrid         │ │                        │
│ └────────────────┘ │                        │
└─────────┬──────────┘                        │
          │                                   │
┌─────────▼──────────┐                        │
│ ML Classifier      │                        │
│ (SVM/LR/DT/GB OvR) │                        │
└─────────┬──────────┘                        │
          │                                   │
          └───────────────┬───────────────────┘
                          │
              ┌───────────▼───────────┐
              │  Dynamic Threshold    │
              │  Optimization         │
              │  (Per-label, val set) │
              └───────────┬───────────┘
                          │
┌─────────────────────────▼──────────────────────────────┐
│            Multi-Label Predictions (22 topics)          │
└────────────────────────────────────────────────────────┘
```

### Dynamic Threshold Optimization

All models (except Gemini) use **dynamic per-label threshold optimization** on the validation set:

1. **Train** the model on the training set.
2. **Obtain predicted probabilities** on the validation set.
3. **For each of the 22 labels independently**, sweep thresholds from 0.1 to 0.9 (step 0.05) and select the one maximizing that label's F1-score.
4. **Apply the optimized thresholds** to the test set predictions.

This approach handles class imbalance more effectively than a fixed 0.5 threshold, as rare labels often benefit from lower thresholds and frequent labels from higher ones.

### Text Preprocessing Strategies

Three preprocessing strategies are used across models:

- **`clean_text`** — Lowercasing, whitespace normalization, punctuation removal. Used by most baselines.
- **`smart_preprocess`** — Domain-specific normalization for Portuguese municipal texts: compound term preservation (e.g., "câmara municipal" → "camara_municipal"), legal reference normalization (e.g., "Decreto Lei" → "decreto_lei"), minimum token filtering. Used by the Gradient Boosting baseline.
- **Raw text** — No preprocessing, relying on the LLM's own text understanding. Used by the Gemini API baseline.

### BERT Feature Extraction Strategies

Two embedding strategies are used for feature-based classifiers:

- **CLS token** — The `[CLS]` token representation from the last hidden layer, providing a fixed-length sentence embedding. Used by LogReg + BERT.
- **Mean pooling** — Average of all token embeddings from the last hidden layer, capturing distributed information across the sequence. Used by SVM + BERTimbau and SVM Hybrid.

---

## Evaluation Metrics

### F1-macro

The arithmetic mean of per-label F1 scores. Treats all labels equally regardless of frequency.
- **Range**: 0.0 to 1.0. Higher is better.
- Penalizes models that perform poorly on rare labels.

### F1-micro

Computes F1 globally by counting total true positives, false positives, and false negatives across all labels.
- **Range**: 0.0 to 1.0. Higher is better.
- Gives more weight to frequent labels.

### Subset Accuracy (Exact Match)

The fraction of samples where the predicted label set exactly matches the true label set.
- **Range**: 0.0 to 1.0. Higher is better.
- Strictest metric — partial matches receive zero credit.

### Hamming Loss

The fraction of wrong labels out of the total number of labels, averaged across all samples.
- **Range**: 0.0 to 1.0. Lower is better.
- Measures the average number of individual label errors.

### Average Precision (macro)

Mean of per-label average precision scores. Summarizes the precision-recall curve.
- **Range**: 0.0 to 1.0. Higher is better.

---

## Experimental Results

Evaluation on the held-out **test set** (24 documents):

| Model | F1-macro | F1-micro | Subset Accuracy | Hamming Loss |
|-------|----------|----------|-----------------|--------------|
| Decision Tree + TF-IDF | 0.3546 | 0.5770 | 0.2609 | 0.0689 |
| SVM + TF-IDF | 0.5070 | 0.7167 | 0.4197 | 0.0441 |
| SVM + BERTimbau Embeddings | 0.4602 | 0.6866 | 0.3837 | 0.0504 |
| Fine-Tuned BERTimbau Large | 0.4819 | 0.7300 | 0.4234 | 0.0435 |
| Logistic Regression + BERT CLS | 0.5422 | 0.7290 | 0.4442 | 0.0438 |
| SVM + Hybrid (TF-IDF + BERT) | 0.5053 | 0.7276 | 0.4216 | 0.0428 |
| Gradient Boosting + AL | 0.5485 | 0.7363 | 0.4518 | 0.0412 |
| Gemini API (few-shot) | 0.496 | 0.525 | - | 0.070 |
| **BERTimbau Fine-Tune** | **0.642** | **0.822** | **0.62** | **0.029** |

**Best model**: BERTimbau Fine-Tune

### Key Observations

- Traditional ML baselines with BERT features (LogReg + BERT, GB + AL) outperform the fully fine-tuned BERTimbau model on F1-macro, likely due to the limited training data (72 documents).
- Dynamic threshold optimization is critical — it significantly improves F1-macro over fixed 0.5 thresholds for imbalanced label distributions.
- The domain-specific `smart_preprocess` function in the GB + AL baseline (Portuguese municipal terminology normalization) provides measurable gains.
- The hybrid feature approach (TF-IDF + BERT) consistently performs well, combining statistical and semantic representations.

### Models & Approaches Summary

| # | Model | Features | Classifier | Text Preprocessing |
|---|-------|----------|------------|-------------------|
| 1 | **DT + TF-IDF** | TF-IDF (1,2)-grams | Decision Tree (OvR) | `clean_text` |
| 2 | **SVM + TF-IDF** | TF-IDF (1,2)-grams, 10k features | Linear SVM (OvR, balanced) | `clean_text` |
| 3 | **SVM + BERTimbau Emb** | BERTimbau Large mean-pooled embeddings | Linear SVM (OvR, balanced) | `clean_text` |
| 4 | **LogReg + BERT CLS** | BERTimbau Base CLS token | Logistic Regression (OvR, balanced) | `clean_text` |
| 5 | **SVM + Hybrid** | TF-IDF (1,2)-grams + BERTimbau Base mean-pooled | Linear SVM (OvR, balanced) | `clean_text` |
| 6 | **Gradient Boosting + AL** | TF-IDF (1,3)-grams + BERTimbau Base | GB Ensemble (3 models) + Active Learning | `smart_preprocess` |
| 7 | **Gemini API** | Raw text, few-shot examples | Gemini 2.5 Pro (few-shot) | None (raw text) |
| 8 | **BERTimbau Fine-Tune** | | | |

---

## Known Issues

- **Limited training data**: With only 72 training documents, end-to-end fine-tuning of BERTimbau Large underperforms some simpler baselines. Feature-based approaches with traditional classifiers are more robust in this low-resource setting.
- **Label imbalance**: Some labels (e.g., "Comunicação e Relações Públicas", "Polícia Municipal") have very few examples in the test set, leading to high variance in per-label F1 scores.
- **Gemini API**: The Gemini baseline is exploratory and requires an active API key. Results depend on model availability and rate limits.

### Reporting Issues

Please report issues on GitHub. Include:
- Python and library versions
- GPU model and CUDA version (if applicable)
- Steps to reproduce the issue
- Error traceback or unexpected output

---

## License

This project is licensed under **Creative Commons Attribution-NoDerivatives 4.0 International** (CC BY-ND 4.0).

You are free to:
- **Share**: Copy and redistribute the material in any medium or format for any purpose, even commercially.

Under the following terms:
- **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **NoDerivatives**: If you remix, transform, or build upon the material, you may not distribute the modified material.

---

## Resources

### Dataset

- **CitiLink-Summ Dataset**: https://github.com/INESCTEC/citilink-summ

### Pre-trained Models

- **BERTimbau Large**: https://huggingface.co/liaad/Citilink-BERTimbau-large-Topic-Classification-pt
---

## Acknowledgments

We would like to extend our gratitude to the following institutions for their invaluable support, research contributions, and collaboration in making this project possible:

- **INESC TEC**
- **University of Beira Interior**
- **University of Porto**

### Funding

This work was funded within the scope of the project CitiLink, with reference 2024.07509.IACDC, which is co-funded by Component 5 - Capitalization and Business Innovation, integrated in the Resilience Dimension of the Recovery and Resilience Plan within the scope of the Recovery and Resilience Mechanism (MRR) of the European Union (EU), framed in the Next Generation EU, for the period 2021 - 2026, measure RE-C05-i08.M04 - "To support the launch of a programme of R&D projects geared towards the development and implementation of advanced cybersecurity, artificial intelligence and data science systems in public administration, as well as a scientific training programme," as part of the funding contract signed between the Recovering Portugal Mission Structure (EMRP) and the FCT - Fundação para a Ciência e a Tecnologia, I.P. (Portuguese Foundation for Science and Technology), as intermediary beneficiary. https://doi.org/10.54499/2024.07509.IACDC

### Tools and Libraries

We also acknowledge the open-source community, particularly the maintainers of **Hugging Face**, **PyTorch**, **scikit-learn**, and **BERTimbau**, whose tools and pre-trained models were fundamental to the development of this classification pipeline.

---

**Last Updated**: February 28, 2026
**Maintained by**: Miguel Marques
