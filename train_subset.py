
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
print("FAST BASELINE - 100K STRATIFIED SAMPLE")
print("="*60)

start = time.time()

print("Loading full dataset...")
df = pd.read_parquet('task_B/train.parquet')
print(f"Original shape: {df.shape}")

print("Sampling 20% stratified by label...")
df_sampled = df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(frac=0.2, random_state=42)
).reset_index(drop=True)
print(f"Sampled shape: {df_sampled.shape}")

le = LabelEncoder()
y = le.fit_transform(df_sampled['label'])

weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
sample_weights = np.array([weights[i] for i in y])

print("\nExtracting char n-grams...")
char_vec = TfidfVectorizer(analyzer='char', ngram_range=(3,5), max_features=5000, min_df=5)
X_char = char_vec.fit_transform(df_sampled['code'])
print(f"Char features: {X_char.shape}")

print("Extracting word n-grams...")
word_vec = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=3000, min_df=5)
X_word = word_vec.fit_transform(df_sampled['code'])
print(f"Word features: {X_word.shape}")

print("Creating language features...")
lang_dum = pd.get_dummies(df_sampled['language'], prefix='lang')
X_lang = csr_matrix(lang_dum.values)
print(f"Language features: {X_lang.shape}")

print("\nStacking features...")
X = hstack([X_char, X_word, X_lang])
print(f"Final shape: {X.shape}")

print("\nTraining XGBoost (100 trees on 100K samples)...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.15,
    subsample=0.8,
    objective='multi:softmax',
    num_class=11,
    n_jobs=-1,
    tree_method='hist',
    random_state=42
)

model.fit(X, y, sample_weight=sample_weights)
print("Training complete!")

print("\nSaving model artifacts...")
joblib.dump(model, 'xgb_classifier.joblib')
joblib.dump(char_vec, 'xgb_char_vectorizer.joblib')
joblib.dump(word_vec, 'xgb_word_vectorizer.joblib')
joblib.dump(le, 'xgb_label_encoder.joblib')
joblib.dump(lang_dum.columns.tolist(), 'xgb_language_columns.joblib')

print(f"\n{'='*60}")
print(f"SUCCESS! Total time: {(time.time()-start)/60:.1f} minutes")
print(f"{'='*60}")
