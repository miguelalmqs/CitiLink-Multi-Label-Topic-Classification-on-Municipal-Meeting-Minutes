#!/bin/bash
# Run all baseline models for multi-label text classification.
# Each script loads data from ../dataset_sample/dset.json and ../split_info.json
# and appends results to ../results/pt_results.json.
#
# Usage:
#   cd TextClassification_ResourcePaper/baselines
#   bash run_all_baselines.sh

set -e

echo "=============================================="
echo " Running all classification baselines"
echo "=============================================="

echo ""
echo "[1/7] Decision Tree + TF-IDF"
python baseline_DT_tfidf.py

echo ""
echo "[2/7] SVM + TF-IDF"
python baseline_SVM_tfidf.py

echo ""
echo "[3/7] SVM + BERTimbau Embeddings"
python baseline_SVM_bert_embeddings.py

echo ""
echo "[4/7] Logistic Regression + BERTimbau CLS"
python baseline_LogReg_bert.py

echo ""
echo "[5/7] SVM + Hybrid (TF-IDF + BERTimbau)"
python baseline_SVM_hybrid.py

echo ""
echo "[6/7] Gradient Boosting + Active Learning"
python baseline_gradient_boosting.py

echo ""
echo "[7/7] Gemini API Few-Shot (requires GEMINI_API_KEY)"
echo "       Skipping by default — uncomment below to run."
# python baseline_gemini.py

echo ""
echo "=============================================="
echo " All baselines complete."
echo " Results saved to ../results/pt_results.json"
echo "=============================================="
