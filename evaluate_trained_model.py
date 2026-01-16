import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import f1_score, classification_report
from scipy.sparse import hstack, csr_matrix

# =========================================================
# CONFIG
# =========================================================
DATA_PATH = "task_B/task_b_trial.parquet"  # file WITH labels
EMB_PATH = "codebert_embeddings.npy"

# =========================================================
# LOAD ARTIFACTS
# =========================================================
print("🔹 Loading trained model artifacts...")

clf = joblib.load("final_classifier.joblib")
char_vec = joblib.load("final_char_vectorizer.joblib")
le = joblib.load("final_label_encoder.joblib")
lang_cols = joblib.load("final_language_columns.joblib")
thresholds = np.load("final_thresholds.npy")

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_parquet(DATA_PATH)

X_code = df["code"].astype(str)
X_lang = df["language"].astype(str)
y_true = le.transform(df["label"].astype(str))

# =========================================================
# LOAD EMBEDDINGS
# =========================================================
embeddings = np.load(EMB_PATH)

# =========================================================
# FEATURE EXTRACTION
# =========================================================
print("🔹 Extracting features...")

X_char = char_vec.transform(X_code)

lang_oh = pd.get_dummies(X_lang)
lang_oh = lang_oh.reindex(columns=lang_cols, fill_value=0)
X_lang_mat = csr_matrix(lang_oh.values)

X_all = hstack([
    X_char,
    csr_matrix(embeddings),
    X_lang_mat
])

# =========================================================
# PREDICTION (SAME DECODING AS TRAINING)
# =========================================================
print("🔹 Running inference...")

probs = clf.predict_proba(X_all)

num_classes = len(le.classes_)
HUMAN = le.transform(["0"])[0]
LLM_CLASSES = [i for i in range(num_classes) if i != HUMAN]

y_pred = []

for p in probs:
    p = p.copy()

    # Hierarchical restriction
    if p[LLM_CLASSES].sum() > p[HUMAN]:
        p[HUMAN] = 0
        p[LLM_CLASSES] /= p[LLM_CLASSES].sum() + 1e-12

    # Threshold decoding
    candidates = np.where(p >= thresholds)[0]
    if len(candidates) > 0:
        pred = candidates[np.argmax(p[candidates])]
    else:
        pred = np.argmax(p)

    y_pred.append(pred)

y_pred = np.array(y_pred)

# =========================================================
# EVALUATION
# =========================================================
macro_f1 = f1_score(y_true, y_pred, average="macro")

print("\n⚠️ TRAINING-SET (SELF-CHECK) MACRO F1:", round(macro_f1, 4))
print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=le.classes_,
        zero_division=0
    )
)
