
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack, csr_matrix
import re

print("="*60)
print("GENERATING SUBMISSION FOR TEST SET (500K samples)")
print("="*60)

# Simple language detector based on patterns
def detect_language(code):
    code_lower = code.lower()
    
    # Java
    if 'public class' in code or 'import java.' in code:
        return 'Java'
    # C#
    elif 'using System' in code or 'namespace' in code and '{' in code:
        return 'C#'
    # Python
    elif 'def ' in code or 'import ' in code or 'print(' in code:
        return 'Python'
    # JavaScript
    elif 'function' in code or 'const ' in code or 'let ' in code or 'var ' in code:
        return 'JavaScript'
    # C++
    elif '#include <iostream>' in code or 'std::' in code:
        return 'C++'
    # C
    elif '#include <stdio.h>' in code or 'printf(' in code:
        return 'C'
    # Go
    elif 'package main' in code or 'func ' in code and 'import (' in code:
        return 'Go'
    # PHP
    elif '<?php' in code or '$_' in code:
        return 'PHP'
    else:
        return 'Python'  # Default

# Load TEST data
print("Loading test data...")
test_df = pd.read_parquet('task_B/test.parquet')
print(f"Test shape: {test_df.shape}")

print("Detecting languages...")
test_df['language'] = test_df['code'].apply(detect_language)
print(f"Language distribution:\n{test_df['language'].value_counts()}")

# Load balanced model
print("\nLoading model artifacts...")
model = joblib.load('xgb_classifier_balanced.joblib')
char_vec = joblib.load('xgb_char_vectorizer_balanced.joblib')
word_vec = joblib.load('xgb_word_vectorizer_balanced.joblib')
le = joblib.load('xgb_label_encoder_balanced.joblib')
lang_cols = joblib.load('xgb_language_columns_balanced.joblib')

# Extract features
print("\nExtracting char n-grams...")
X_char = char_vec.transform(test_df['code'])

print("Extracting word n-grams...")
X_word = word_vec.transform(test_df['code'])

print("Creating language features...")
language_dummies = pd.get_dummies(test_df['language'], prefix='lang')
for col in lang_cols:
    if col not in language_dummies.columns:
        language_dummies[col] = 0
language_dummies = language_dummies[lang_cols]
X_lang = csr_matrix(language_dummies.values)

X_test = hstack([X_char, X_word, X_lang])
print(f"Final shape: {X_test.shape}")

# Predict
print("\nGenerating predictions...")
y_pred = model.predict(X_test)

# Create submission
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'label': y_pred
})

submission.to_csv('test_submission.csv', index=False)
print("\n" + "="*60)
print("✓ Saved: test_submission.csv")
print("="*60)
print(f"\nPrediction distribution:")
print(submission['label'].value_counts().sort_index())
