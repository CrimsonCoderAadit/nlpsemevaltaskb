
import pandas as pd
from sklearn.metrics import classification_report, f1_score

print("="*60)
print("LOCAL EVALUATION")
print("="*60)

# Load validation data with labels
val_df = pd.read_parquet('task_B/validation.parquet')
y_true = val_df['label'].values

# Load predictions
submission = pd.read_csv('submission.csv')
y_pred = submission['label'].values

# Compute metrics
macro_f1 = f1_score(y_true, y_pred, average='macro')
print(f"\n{'='*60}")
print(f"MACRO F1 SCORE: {macro_f1:.4f}")
print(f"{'='*60}\n")

print("Per-class metrics:")
print(classification_report(y_true, y_pred, digits=4))
