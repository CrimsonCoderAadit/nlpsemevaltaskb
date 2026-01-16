
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix

print("="*60)
print("GENERATING FINAL SUBMISSION - FULL MODEL")
print("="*60)

# Language detector
def detect_language(code):
    code_lower = code.lower()
    if 'public class' in code or 'import java.' in code: return 'Java'
    elif 'using System' in code or 'namespace' in code and '{' in code: return 'C#'
    elif 'def ' in code or 'import ' in code: return 'Python'
    elif 'function' in code or 'const ' in code or 'let ' in code: return 'JavaScript'
    elif '#include <iostream>' in code: return 'C++'
    elif '#include <stdio.h>' in code: return 'C'
    elif 'package main' in code: return 'Go'
    elif '<?php' in code: return 'PHP'
    else: return 'Python'

# Load test
test_df = pd.read_parquet('task_B/test.parquet')
test_df['language'] = test_df['code'].apply(detect_language)
print(f"Test shape: {test_df.shape}")

# Load FULL model
model = joblib.load('xgb_classifier_full.joblib')
char_vec = joblib.load('xgb_char_vectorizer_full.joblib')
word_vec = joblib.load('xgb_word_vectorizer_full.joblib')
le = joblib.load('xgb_label_encoder_full.joblib')
lang_cols = joblib.load('xgb_language_columns_full.joblib')

# Features
print("Extracting features...")
X_char = char_vec.transform(test_df['code'])
X_word = word_vec.transform(test_df['code'])

lang_dum = pd.get_dummies(test_df['language'], prefix='lang')
for col in lang_cols:
    if col not in lang_dum.columns:
        lang_dum[col] = 0
lang_dum = lang_dum[lang_cols]
X_lang = csr_matrix(lang_dum.values)

X_test = hstack([X_char, X_word, X_lang])
print(f"Feature shape: {X_test.shape}")

# Predict
print("Predicting...")
y_pred = model.predict(X_test)

# Save
submission = pd.DataFrame({'ID': test_df['ID'], 'label': y_pred})
submission.to_csv('test_submission_FINAL.csv', index=False)

print("\n" + "="*60)
print("✓ SAVED: test_submission_FINAL.csv")
print("="*60)
print(f"\nPrediction distribution:")
print(submission['label'].value_counts().sort_index())
