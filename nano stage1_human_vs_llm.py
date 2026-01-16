import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

print("🔹 Stage 1: Human vs LLM")

# =============================
# Load data
# =============================
df = pd.read_parquet("task_B/task_b_trial.parquet")

# Human = label "0" (adjust if needed)
df["binary_label"] = (df["label"] != 0).astype(int)

X = df["code"].astype(str).str.replace("\r\n", "\n")
y = df["binary_label"].values

# =============================
# Split
# =============================
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

# =============================
# Char n-grams
# =============================
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 6),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
    max_features=300_000
)

X_tr_vec = vectorizer.fit_transform(X_tr)
X_va_vec = vectorizer.transform(X_va)

# =============================
# Binary classifier
# =============================
clf = SGDClassifier(
    loss="log_loss",
    class_weight="balanced",
    max_iter=3000,
    random_state=42,
    n_jobs=-1
)

clf.fit(X_tr_vec, y_tr)

# =============================
# Eval
# =============================
y_pred = clf.predict(X_va_vec)

print("\nBinary F1 (Human vs LLM):",
      f1_score(y_va, y_pred))

print("\nReport:")
print(classification_report(y_va, y_pred))
