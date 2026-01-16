import pandas as pd
import numpy as np
import random
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

from scipy.sparse import hstack, csr_matrix

# =========================================================
# Synthetic augmentation (USED ON ALL DATA)
# =========================================================
def exaggerate_whitespace(code):
    code = re.sub(r"\n", "\n\n\n", code)
    code = re.sub(r" {2,}", "    ", code)
    return code

def exaggerate_names(code):
    return re.sub(
        r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b",
        lambda m: m.group(1) + "_" * random.randint(1, 3),
        code
    )

def exaggerate_comments(code):
    lines = code.split("\n")
    out = []
    for l in lines:
        out.append(l)
        if random.random() < 0.2:
            out.append("# synthetic comment")
    return "\n".join(out)

def synthetic_variant(code):
    fn = random.choice([
        exaggerate_whitespace,
        exaggerate_names,
        exaggerate_comments
    ])
    return fn(code)

# =========================================================
# LOAD ALL DATA
# =========================================================
print("🔹 FINAL TRAINING — USING 100% OF DATA")

df = pd.read_parquet("task_B/task_b_trial.parquet")

X_code = df["code"].astype(str)
X_lang = df["language"].astype(str)
y = df["label"].astype(str)

# =========================================================
# LABEL ENCODING
# =========================================================
le = LabelEncoder()
y_enc = le.fit_transform(y)

HUMAN_CLASS = le.transform(["0"])[0]

# =========================================================
# FULL-DATA SYNTHETIC AUGMENTATION
# =========================================================
X_all = list(X_code)
y_all = list(y_enc)
lang_all = list(X_lang)

for code, label, lang in zip(X_code, y_enc, X_lang):
    if label != HUMAN_CLASS:
        for _ in range(2):
            X_all.append(synthetic_variant(code))
            y_all.append(label)
            lang_all.append(lang)

X_all = pd.Series(X_all)
y_all = np.array(y_all)
lang_all = pd.Series(lang_all)

print(f"Total samples after augmentation: {len(X_all)}")

# =========================================================
# LOAD CodeBERT EMBEDDINGS (ORIGINAL ONLY)
# =========================================================
embeddings = np.load("codebert_embeddings.npy")

emb_all = embeddings
pad = np.zeros(
    (len(X_all) - len(embeddings), embeddings.shape[1]),
    dtype=np.float32
)
emb_all = np.vstack([emb_all, pad])

# =========================================================
# CHAR N-GRAM FEATURES
# =========================================================
char_vec = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 6),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
    max_features=200_000
)

X_char = char_vec.fit_transform(X_all)

# =========================================================
# LANGUAGE FEATURES
# =========================================================
lang_oh = pd.get_dummies(lang_all)
X_lang_mat = csr_matrix(lang_oh.values)

# =========================================================
# COMBINE ALL FEATURES
# =========================================================
X_all_feat = hstack([
    X_char,
    csr_matrix(emb_all),
    X_lang_mat
])

# =========================================================
# FINAL MODEL TRAINING
# =========================================================
clf = LogisticRegression(
    max_iter=4000,
    class_weight="balanced",
    n_jobs=-1,
    verbose=1
)

clf.fit(X_all_feat, y_all)

# =========================================================
# SAVE EVERYTHING NEEDED FOR INFERENCE
# =========================================================
joblib.dump(clf, "final_classifier.joblib")
joblib.dump(char_vec, "final_char_vectorizer.joblib")
joblib.dump(le, "final_label_encoder.joblib")
joblib.dump(lang_oh.columns.tolist(), "final_language_columns.joblib")

# thresholds (FIXED, NOT TUNED)
thresholds = np.full(len(le.classes_), 0.18)
thresholds[HUMAN_CLASS] = 0.60
np.save("final_thresholds.npy", thresholds)

print("\n✅ FINAL MODEL TRAINED AND SAVED")
print("Artifacts saved:")
print(" - final_classifier.joblib")
print(" - final_char_vectorizer.joblib")
print(" - final_label_encoder.joblib")
print(" - final_language_columns.joblib")
print(" - final_thresholds.npy")
