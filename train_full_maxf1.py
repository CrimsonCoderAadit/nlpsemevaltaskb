
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
import joblib
from scipy.sparse import hstack, csr_matrix
import time

print("="*60)
print("MAX F1 TRAINING - FULL 500K + SYNTHETIC AUGMENTATION")
print("="*60)

start = time.time()

# Load FULL training data
print("Loading FULL training data...")
df = pd.read_parquet('task_B/train.parquet')
print(f"Original shape: {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")

# SYNTHETIC AUGMENTATION for rare classes (< 10K samples)
print("\nGenerating synthetic samples for rare classes...")
rare_threshold = 10000
synthetic_samples = []

for label in df['label'].unique():
    label_df = df[df['label'] == label]
    count = len(label_df)
    
    if count < rare_threshold:
        # Oversample to reach threshold
        n_synthetic = rare_threshold - count
        synthetic = label_df.sample(n=n_synthetic, replace=True, random_state=42)
        
        # Add small perturbations to code (simulate variations)
        synthetic = synthetic.copy()
        synthetic['code'] = synthetic['code'].apply(
            lambda x: x + '\n# augmented' if len(x) < 5000 else x
        )
        synthetic_samples.append(synthetic)
        print(f"  Class {label}: {count} → {rare_threshold} (+{n_synthetic} synthetic)")

if synthetic_samples:
    df_synthetic = pd.concat(synthetic_samples, ignore_index=True)
    df = pd.concat([df, df_synthetic], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nFinal shape after augmentation: {df.shape}")
print(f"Final distribution:\n{df['label'].value_counts().sort_index()}")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df['label'])

# MODERATE class weights (not too aggressive)
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
# Apply moderate scaling (1.2x instead of 1.5x)
class_weights = class_weights ** 1.2
sample_weights = np.array([class_weights[i] for i in y])

print(f"\nModerate class weights: {dict(zip(le.classes_, np.round(class_weights, 2)))}")

# Feature extraction
print("\nExtracting char n-grams (this takes time on 500K+)...")
t = time.time()
char_vec = TfidfVectorizer(analyzer='char', ngram_range=(3,5), max_features=8000, min_df=3)
X_char = char_vec.fit_transform(df['code'])
print(f"  Char: {X_char.shape} ({time.time()-t:.1f}s)")

print("Extracting word n-grams...")
t = time.time()
word_vec = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=5000, min_df=3)
X_word = word_vec.fit_transform(df['code'])
print(f"  Word: {X_word.shape} ({time.time()-t:.1f}s)")

print("Creating language features...")
lang_dum = pd.get_dummies(df['language'], prefix='lang')
X_lang = csr_matrix(lang_dum.values)

X = hstack([X_char, X_word, X_lang])
print(f"\nFinal feature shape: {X.shape}")

# Train XGBoost with optimal settings
print("\nTraining XGBoost (200 trees on full data)...")
t = time.time()
model = XGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0.1,
    objective='multi:softmax',
    num_class=11,
    n_jobs=-1,
    tree_method='hist',
    random_state=42
)

model.fit(X, y, sample_weight=sample_weights)
print(f"  Training done ({time.time()-t:.1f}s)")

# Save
print("\nSaving artifacts...")
joblib.dump(model, 'xgb_classifier_full.joblib')
joblib.dump(char_vec, 'xgb_char_vectorizer_full.joblib')
joblib.dump(word_vec, 'xgb_word_vectorizer_full.joblib')
joblib.dump(le, 'xgb_label_encoder_full.joblib')
joblib.dump(lang_dum.columns.tolist(), 'xgb_language_columns_full.joblib')

print(f"\n{'='*60}")
print(f"SUCCESS! Total time: {(time.time()-start)/60:.1f} minutes")
print(f"{'='*60}")
