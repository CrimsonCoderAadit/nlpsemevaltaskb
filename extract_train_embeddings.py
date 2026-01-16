
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

print("Loading train data...")
df = pd.read_parquet('task_B/train.parquet')
print(f"Train shape: {df.shape}")

print("Loading CodeBERT model...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

def get_embedding(code_text, max_length=512):
    try:
        inputs = tokenizer(
            code_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding.flatten()
    except Exception as e:
        print(f"Error processing code: {e}")
        return np.zeros(768)

print("Extracting embeddings (this will take a while)...")
batch_size = 32
embeddings = []

for i in tqdm(range(0, len(df), batch_size)):
    batch = df['code'].iloc[i:i+batch_size].tolist()
    batch_embeddings = [get_embedding(code) for code in batch]
    embeddings.extend(batch_embeddings)

embeddings = np.array(embeddings)
print(f"Embeddings shape: {embeddings.shape}")

print("Saving embeddings...")
np.save('train_codebert_embeddings.npy', embeddings)
print("Done! Saved to train_codebert_embeddings.npy")
