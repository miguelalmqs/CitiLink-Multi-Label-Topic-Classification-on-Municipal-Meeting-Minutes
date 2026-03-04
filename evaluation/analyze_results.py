import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_PATH = os.path.join(ROOT_DIR, "results", "pt_results.json")

# Carregar resultados
with open(RESULTS_PATH, "r") as f:
    results = json.load(f)

print("🏆 ANÁLISE COMPLETA DOS RESULTADOS - CLASSIFICAÇÃO MULTI-LABEL")
print("=" * 80)

# Extrair métricas principais
models_data = []
for model_name, data in results.items():
    models_data.append({
        "Model": model_name,
        "F1-macro": data["f1_macro"],
        "F1-micro": data["f1_micro"],
        "Accuracy": data["accuracy"],
        "Hamming Loss": data["hamming_loss"],
        "Avg Precision": data.get("average_precision_macro", 0)
    })

# Criar DataFrame para análise
df = pd.DataFrame(models_data)
df = df.sort_values("F1-macro", ascending=False)

print("\n📊 RANKING DOS MODELOS (por F1-macro):")
print("-" * 60)
for i, row in df.iterrows():
    print(f"{row.name+1:2d}. {row['Model']:35s} | F1-macro: {row['F1-macro']:.4f}")

print(f"\n🥇 MELHOR MODELO: {df.iloc[0]['Model']}")
print(f"   F1-macro: {df.iloc[0]['F1-macro']:.4f}")
print(f"   F1-micro: {df.iloc[0]['F1-micro']:.4f}")
print(f"   Accuracy: {df.iloc[0]['Accuracy']:.4f}")

print("\n📈 ANÁLISE COMPARATIVA:")
print("-" * 60)

# Comparação detalhada dos top 3
top_3 = df.head(3)
print("Top 3 modelos:")
for i, row in top_3.iterrows():
    print(f"\n{i+1}. {row['Model']}")
    print(f"   F1-macro: {row['F1-macro']:.4f} | F1-micro: {row['F1-micro']:.4f}")
    print(f"   Accuracy: {row['Accuracy']:.4f} | Hamming Loss: {row['Hamming Loss']:.4f}")

# Diferenças percentuais
best_f1 = df.iloc[0]['F1-macro']
print(f"\n🔍 GAPS DE PERFORMANCE:")
for i, row in df.iterrows():
    gap = (best_f1 - row['F1-macro']) / best_f1 * 100
    if gap > 0:
        print(f"   {row['Model']:35s}: -{gap:.1f}%")

print("\n📊 INSIGHTS TÉCNICOS:")
print("-" * 60)

# Análise de técnicas
print("✅ TÉCNICAS QUE FUNCIONARAM MELHOR:")
print("   1. Logistic Regression + BERT embeddings")
print("   2. Dynamic threshold optimization")
print("   3. TF-IDF com n-gramas (1,3)")
print("   4. Class balancing (balanced weights)")
print("   5. Português BERT models (BERTimbau)")

print("\n❌ TÉCNICAS QUE NÃO MELHORARAM:")
print("   1. Ensemble complexos (overfitting)")
print("   2. Filtros de qualidade muito restritivos")
print("   3. Feature engineering estatística")
print("   4. Modelos tree-based (Decision Tree)")
print("   5. Char n-grams (pouco valor para português)")

print("\n🎯 RECOMENDAÇÕES FINAIS:")
print("-" * 60)
print("Para classificação multi-label em português:")
print("1. Use Logistic Regression com BERT embeddings")
print("2. Otimize thresholds por label individualmente")
print("3. Mantenha simplicidade - ensembles podem prejudicar")
print("4. TF-IDF (1,3)-grams como baseline robusto")
print("5. BERTimbau > BERT multilingual para português")

# Salvar análise
analysis_results = {
    "analysis_date": datetime.now().isoformat(),
    "best_model": {
        "name": df.iloc[0]['Model'],
        "f1_macro": df.iloc[0]['F1-macro'],
        "f1_micro": df.iloc[0]['F1-micro'],
        "accuracy": df.iloc[0]['Accuracy']
    },
    "ranking": df.to_dict('records'),
    "insights": {
        "best_techniques": [
            "Logistic Regression + BERT embeddings",
            "Dynamic threshold optimization",
            "TF-IDF with (1,3)-grams",
            "Class balancing",
            "Portuguese BERT models"
        ],
        "failed_techniques": [
            "Complex ensembles",
            "Quality filtering",
            "Statistical features",
            "Tree-based models",
            "Character n-grams"
        ]
    },
    "recommendations": [
        "Use LogisticRegression + BERT for Portuguese multi-label",
        "Optimize thresholds individually per label",
        "Keep models simple - avoid complex ensembles",
        "TF-IDF (1,3)-grams as robust baseline",
        "BERTimbau > multilingual BERT for Portuguese"
    ]
}

with open(os.path.join(ROOT_DIR, "results", "final_analysis.json"), "w") as f:
    json.dump(analysis_results, f, indent=4, ensure_ascii=False)

print(f"\n✅ Análise completa salva em 'results/final_analysis.json'")
print(f"\n🏁 PROJETO CONCLUÍDO COM SUCESSO!")
print(f"   Melhor resultado: F1-macro = {best_f1:.4f}")
print(f"   Modelo recomendado: {df.iloc[0]['Model']}")
print("=" * 80)
