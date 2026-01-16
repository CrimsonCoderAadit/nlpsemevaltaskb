
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
import joblib
from scipy.sparse import hstack, csr_matrix

print("="*60)
print("TRAINING WITH CLASS BALANCE")
print("="*60)

# Load data
print("Loading training data...")
df = pd.read_parquet('task_B/train.parquet')
print(f"Train shape: {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(df['label'])

# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
sample_weights = np.array([class_weights[y] for y in y_train])
print(f"\nClass weights: {dict(zip(le.classes_, class_weights))}")

# Feature 1: Char n-gram TF-IDF
print("\nExtracting char n-gram features...")
char_vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 6),
    max_features=10000,
    min_df=2
)
X_char = char_vectorizer.fit_transform(df['code'])
print(f"Char features shape: {X_char.shape}")

# Feature 2: Word n-gram TF-IDF
print("Extracting word n-gram features...")
word_vectorizer = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),
    max_features=5000,
    min_df=2
)
X_word = word_vectorizer.fit_transform(df['code'])
print(f"Word features shape: {X_word.shape}")

# Feature 3: Language one-hot
print("Creating language features...")
language_dummies = pd.get_dummies(df['language'], prefix='lang')
X_lang = csr_matrix(language_dummies.values)
print(f"Language features shape: {X_lang.shape}")

# Stack all features
print("\nStacking features...")
X_train = hstack([X_char, X_word, X_lang])
print(f"Final feature shape: {X_train.shape}")

# Train XGBoost with class weights
print("\nTraining XGBoost classifier...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    num_class=11,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights,
    verbose=True
)

# Save artifacts
print("\nSaving model artifacts...")
joblib.dump(model, 'xgb_classifier.joblib')
joblib.dump(char_vectorizer, 'xgb_char_vectorizer.joblib')
joblib.dump(word_vectorizer, 'xgb_word_vectorizer.joblib')
joblib.dump(le, 'xgb_label_encoder.joblib')
joblib.dump(language_dummies.columns.tolist(), 'xgb_language_columns.joblib')

print("✓ Training complete!")
print("✓ Saved: xgb_classifier.joblib")
print("✓ Saved: xgb_char_vectorizer.joblib")
print("✓ Saved: xgb_word_vectorizer.joblib")
print("✓ Saved: xgb_label_encoder.joblib")
print("✓ Saved: xgb_language_columns.joblib")
