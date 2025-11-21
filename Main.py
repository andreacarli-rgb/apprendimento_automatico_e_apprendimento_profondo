# === IMPORT ===
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import shap
from matplotlib.backends.backend_pdf import PdfPages

# === CREA CARTELLA OUTPUT ===
os.makedirs("output", exist_ok=True)

# === IMPORT DATASET UCI ===
from ucimlrepo import fetch_ucirepo
bc = fetch_ucirepo(id=17)

X = bc.data.features
y = bc.data.targets.iloc[:, 0].map({"M": 1, "B": 0}).astype(int)

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === Standardizzazione ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === PDF REPORT ===
pdf = PdfPages("output/Report_Elaborato.pdf")

# === PCA ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(6,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y_train, cmap="coolwarm", alpha=0.7)
plt.title("PCA - Prime 2 Componenti")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("output/PCA_plot.png")
pdf.savefig()
plt.close()

print("Varianza spiegata:", pca.explained_variance_ratio_)

# === MODELLI ===
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=200)
}

results = {}

# === Addestramento modelli ===
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:,1]

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_prob)
    }

# === Salva tabella risultati ===
results_df = pd.DataFrame(results).T
results_df.to_csv("output/risultati_modelli.csv")
print(results_df)

# === CONFUSION MATRIX ===
for name, model in models.items():
    plt.figure(figsize=(4,4))
    ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test)
    plt.title(f"Matrice di Confusione - {name}")
    plt.savefig(f"output/conf_matrix_{name}.png")
    pdf.savefig()
    plt.close()

# === ROC CURVE ===
plt.figure(figsize=(7,6))
for name, model in models.items():
    RocCurveDisplay.from_estimator(model, X_test_scaled, y_test)
plt.title("ROC Curve - Confronto Modelli")
plt.savefig("output/roc_curve.png")
pdf.savefig()
plt.close()

# === SHAP ===
best_model = models["Random Forest"]
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_train_scaled)

# Converti le feature scalate in DataFrame con i nomi originali
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)

# SHAP bar plot
plt.title("SHAP Value - Bar")
shap.summary_plot(shap_values, X_train_scaled_df, plot_type="bar", show=False)
plt.savefig("output/shap_bar.png")
pdf.savefig()
plt.close()

# SHAP summary plot
shap.summary_plot(shap_values, X_train_scaled_df, show=False)
plt.savefig("output/shap_summary.png")
pdf.savefig()
plt.close()

# === Salvataggio shap values ===
np.save("output/shap_values.npy", shap_values[1])

# === CHIUSURA PDF ===
pdf.close()

print("\nReport generato nella directory 'output/'\n")
