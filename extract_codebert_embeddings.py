import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

MODEL = "microsoft/codebert-base"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print("🔹 Loading CodeBERT...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(DEVICE)
model.eval()

df = pd.read_parquet("task_B/task_b_trial.parquet")
codes = df["code"].astype(str).tolist()

embeddings = []

with torch.no_grad():
    for code in tqdm(codes):
        inputs = tokenizer(
            code,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt"
        ).to(DEVICE)

        out = model(**inputs)
        cls_emb = out.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_emb[0])

embeddings = np.vstack(embeddings)

np.save("codebert_embeddings.npy", embeddings)
print("✅ Saved embeddings:", embeddings.shape)
