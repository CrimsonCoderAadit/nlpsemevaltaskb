
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack, csr_matrix

print("="*60)
print("GENERATING SUBMISSION")
print("="*60)

# Load validation data
print("Loading validation data...")
val_df = pd.read_parquet('task_B/validation.parquet')
print(f"Validation shape: {val_df.shape}")

# Load artifacts
print("Loading model artifacts...")
model = joblib.load('xgb_classifier.joblib')
char_vec = joblib.load('xgb_char_vectorizer.joblib')
word_vec = joblib.load('xgb_word_vectorizer.joblib')
le = joblib.load('xgb_label_encoder.joblib')
lang_cols = joblib.load('xgb_language_columns.joblib')

# Extract features
print("\nExtracting features...")
X_char = char_vec.transform(val_df['code'])
X_word = word_vec.transform(val_df['code'])

language_dummies = pd.get_dummies(val_df['language'], prefix='lang')
# Align columns with training
for col in lang_cols:
    if col not in language_dummies.columns:
        language_dummies[col] = 0
language_dummies = language_dummies[lang_cols]
X_lang = csr_matrix(language_dummies.values)

X_val = hstack([X_char, X_word, X_lang])
print(f"Feature shape: {X_val.shape}")

# Predict
print("\nGenerating predictions...")
y_pred = model.predict(X_val)

# Create submission
submission = pd.DataFrame({
    'ID': range(len(val_df)),
    'label': y_pred
})

submission.to_csv('submission.csv', index=False)
print("\n✓ Saved: submission.csv")
print(f"✓ Shape: {submission.shape}")
print("\nFirst 10 predictions:")
print(submission.head(10))
print(f"\nPrediction distribution:\n{submission['label'].value_counts().sort_index()}")
