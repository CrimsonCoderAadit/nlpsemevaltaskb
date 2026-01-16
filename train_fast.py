
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
print("FAST TRAINING - OPTIMIZED")
print("="*60)

start_time = time.time()

print("Loading data...")
df = pd.read_parquet('task_B/train.parquet')
print(f"Shape: {df.shape}")

le = LabelEncoder()
y_train = le.fit_transform(df['label'])

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
sample_weights = np.array([class_weights[y] for y in y_train])

print("\nExtracting char n-grams...")
t = time.time()
char_vec = TfidfVectorizer(analyzer='char', ngram_range=(3,5), max_features=5000, min_df=5)
X_char = char_vec.fit_transform(df['code'])
print(f"✓ {X_char.shape} ({time.time()-t:.1f}s)")

print("Extracting word n-grams...")
t = time.time()
word_vec = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=3000, min_df=5)
X_word = word_vec.fit_transform(df['code'])
print(f"✓ {X_word.shape} ({time.time()-t:.1f}s)")

print("Creating language features...")
lang_dummies = pd.get_dummies(df['language'], prefix='lang')
X_lang = csr_matrix(lang_dummies.values)
print(f"✓ {X_lang.shape}")

print("\nStacking features...")
X_train = hstack([X_char, X_word, X_lang])
print(f"Final: {X_train.shape}")

print("\nTraining XGBoost (150 trees)...")
t = time.time()
model = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.15,
    subsample=0.8,
    objective='multi:softmax',
    num_class=11,
    n_jobs=-1,
    tree_method='hist',
    random_state=42
)

model.fit(X_train, y_train, sample_weight=sample_weights)
print(f"✓ Training done ({time.time()-t:.1f}s)")

print("\nSaving...")
joblib.dump(model, 'xgb_classifier.joblib')
joblib.dump(char_vec, 'xgb_char_vectorizer.joblib')
joblib.dump(word_vec, 'xgb_word_vectorizer.joblib')
joblib.dump(le, 'xgb_label_encoder.joblib')
joblib.dump(lang_dummies.columns.tolist(), 'xgb_language_columns.joblib')

print(f"\n✓ COMPLETE! Total time: {time.time()-start_time:.1f}s")
