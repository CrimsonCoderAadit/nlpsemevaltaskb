
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

print("Extracting CodeBERT for VALIDATION set...")

val_df = pd.read_parquet('task_B/validation.parquet')
print(f"Validation shape: {val_df.shape}")

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
model.eval()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
print(f"Using: {device}")

def get_embedding_batch(codes):
    inputs = tokenizer(codes, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

batch_size = 16
embeddings = []

for i in tqdm(range(0, len(val_df), batch_size)):
    batch = val_df['code'].iloc[i:i+batch_size].tolist()
    embeddings.append(get_embedding_batch(batch))

embeddings = np.vstack(embeddings)
np.save('val_codebert_embeddings.npy', embeddings)
print(f"✓ Saved: val_codebert_embeddings.npy ({embeddings.shape})")
