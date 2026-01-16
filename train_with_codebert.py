
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
print("TRAINING WITH TFIDF + CODEBERT EMBEDDINGS")
print("="*60)

start = time.time()

# Load training data
print("Loading training data...")
df = pd.read_parquet('task_B/train.parquet')
print(f"Train shape: {df.shape}")

# Load CodeBERT embeddings
print("Loading CodeBERT embeddings...")
codebert_embeddings = np.load('train_codebert_embeddings.npy')
print(f"CodeBERT embeddings shape: {codebert_embeddings.shape}")

# Verify alignment
assert len(df) == len(codebert_embeddings), "Mismatch between data and embeddings!"

# Synthetic augmentation for rare classes
print("\nApplying synthetic augmentation...")
rare_threshold = 10000
synthetic_samples = []
synthetic_embeddings = []

for label in df['label'].unique():
    label_indices = df[df['label'] == label].index
    label_df = df.loc[label_indices].reset_index(drop=True)
    label_embeddings = codebert_embeddings[label_indices]
    count = len(label_df)
    
    if count < rare_threshold:
        n_synthetic = rare_threshold - count
        indices = np.random.choice(len(label_df), size=n_synthetic, replace=True)
        
        synthetic = label_df.iloc[indices].copy()
        synthetic['code'] = synthetic['code'] + '\n# augmented'
        synthetic_samples.append(synthetic)
        
        # Reuse embeddings for synthetic samples
        synthetic_embeddings.append(label_embeddings[indices])
        
        print(f"  Class {label}: {count} -> {rare_threshold}")

if synthetic_samples:
    df_synthetic = pd.concat(synthetic_samples, ignore_index=True)
    synthetic_emb = np.vstack(synthetic_embeddings)
    
    df = pd.concat([df, df_synthetic], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Align embeddings with shuffled data
    original_embeddings = codebert_embeddings
    codebert_embeddings = np.vstack([original_embeddings, synthetic_emb])
    # Reorder to match df after shuffle
    # This is tricky - let's just not shuffle and keep alignment
    df = pd.concat([pd.read_parquet('task_B/train.parquet'), df_synthetic], ignore_index=True)
    codebert_embeddings = np.vstack([original_embeddings, synthetic_emb])

print(f"Final shape: {df.shape}")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df['label'])

# Moderate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = class_weights ** 1.2
sample_weights = np.array([class_weights[i] for i in y])

print(f"\nClass weights: {dict(zip(le.classes_, np.round(class_weights, 2)))}")

# Extract TF-IDF features
print("\nExtracting TF-IDF features...")
t = time.time()
char_vec = TfidfVectorizer(analyzer='char', ngram_range=(3,5), max_features=8000, min_df=3)
X_char = char_vec.fit_transform(df['code'])
print(f"  Char: {X_char.shape} ({time.time()-t:.1f}s)")

t = time.time()
word_vec = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=5000, min_df=3)
X_word = word_vec.fit_transform(df['code'])
print(f"  Word: {X_word.shape} ({time.time()-t:.1f}s)")

# Language features
lang_dum = pd.get_dummies(df['language'], prefix='lang')
X_lang = csr_matrix(lang_dum.values)

# Convert CodeBERT to sparse
X_codebert = csr_matrix(codebert_embeddings)
print(f"  CodeBERT: {X_codebert.shape}")

# Stack ALL features
X = hstack([X_char, X_word, X_lang, X_codebert])
print(f"\nFinal feature shape: {X.shape}")

# Train XGBoost
print("\nTraining XGBoost...")
t = time.time()
model = XGBClassifier(
    n_estimators=250,
    max_depth=8,
    learning_rate=0.08,
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
print(f"  Done ({time.time()-t:.1f}s)")

# Save
print("\nSaving artifacts...")
joblib.dump(model, 'xgb_classifier_codebert.joblib')
joblib.dump(char_vec, 'xgb_char_vectorizer_codebert.joblib')
joblib.dump(word_vec, 'xgb_word_vectorizer_codebert.joblib')
joblib.dump(le, 'xgb_label_encoder_codebert.joblib')
joblib.dump(lang_dum.columns.tolist(), 'xgb_language_columns_codebert.joblib')

print(f"\n{'='*60}")
print(f"SUCCESS! Total: {(time.time()-start)/60:.1f} mins")
print(f"{'='*60}")
