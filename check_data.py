import pandas as pd
import json

df = pd.read_parquet("task_B/task_b_trial.parquet")

print("Columns:", df.columns)
print("Number of rows:", len(df))
print("\nFirst 5 rows:\n")
print(df.head())

with open("task_B/id_to_label.json") as f:
    id_to_label = json.load(f)

label_id = df["label"].iloc[0]
print("\nLabel example:")
print(label_id, "->", id_to_label[str(label_id)])


