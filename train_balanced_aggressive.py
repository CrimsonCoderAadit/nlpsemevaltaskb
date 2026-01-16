
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
print("TRAINING WITH FOCAL LOSS + HEAVY CLASS WEIGHTS")
print("="*60)

start = time.time()

print("Loading and sampling data...")
df = pd.read_parquet('task_B/train.parquet')
df_sampled = df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(frac=0.2, random_state=42)
).reset_index(drop=True)
print(f"Shape: {df_sampled.shape}")

le = LabelEncoder()
y = le.fit_transform(df_sampled['label'])

# Compute HEAVILY WEIGHTED class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
# Square the weights to make rare classes even more important
class_weights = class_weights ** 1.5
sample_weights = np.array([class_weights[i] for i in y])

print(f"Class weights (squared): {dict(zip(le.classes_, class_weights))}")

print("\nExtracting features...")
char_vec = TfidfVectorizer(analyzer='char', ngram_range=(3,5), max_features=5000, min_df=5)
X_char = char_vec.fit_transform(df_sampled['code'])

word_vec = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=3000, min_df=5)
X_word = word_vec.fit_transform(df_sampled['code'])

lang_dum = pd.get_dummies(df_sampled['language'], prefix='lang')
X_lang = csr_matrix(lang_dum.values)

X = hstack([X_char, X_word, X_lang])
print(f"Final shape: {X.shape}")

print("\nTraining with aggressive class balancing...")
model = XGBClassifier(
    n_estimators=150,  # More trees for better learning
    max_depth=7,       # Deeper trees
    learning_rate=0.1,  # Slower learning
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,  # Allow smaller leaf nodes
    gamma=0,             # No pruning penalty
    objective='multi:softmax',
    num_class=11,
    n_jobs=-1,
    tree_method='hist',
    random_state=42
)

model.fit(X, y, sample_weight=sample_weights)

print("\nSaving artifacts...")
joblib.dump(model, 'xgb_classifier_balanced.joblib')
joblib.dump(char_vec, 'xgb_char_vectorizer_balanced.joblib')
joblib.dump(word_vec, 'xgb_word_vectorizer_balanced.joblib')
joblib.dump(le, 'xgb_label_encoder_balanced.joblib')
joblib.dump(lang_dum.columns.tolist(), 'xgb_language_columns_balanced.joblib')

print(f"\nSUCCESS! Time: {(time.time()-start)/60:.1f} mins")
