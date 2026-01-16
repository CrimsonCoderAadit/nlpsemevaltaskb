
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import time

print("="*60)
print("EXTRACTING CODEBERT - TRAIN SET (OVERNIGHT)")
print("="*60)

start = time.time()

# Load train data
train_df = pd.read_parquet('task_B/train.parquet')
print(f"Train shape: {train_df.shape}")

# Load model
print("\nLoading CodeBERT...")
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

# Process in small batches
batch_size = 8  # Small for M2 Mac
embeddings = []

print(f"\nProcessing {len(train_df)} samples (batch size: {batch_size})...")
print("This will take 8-12 hours. Let it run overnight!")

for i in tqdm(range(0, len(train_df), batch_size)):
    batch = train_df['code'].iloc[i:i+batch_size].tolist()
    embeddings.append(get_embedding_batch(batch))
    
    # Save checkpoint every 50K
    if (i + batch_size) % 50000 == 0 and i > 0:
        checkpoint = np.vstack(embeddings)
        np.save(f'train_codebert_checkpoint_{i}.npy', checkpoint)
        print(f"\nCheckpoint saved at {i} samples")

embeddings = np.vstack(embeddings)
np.save('train_codebert_embeddings.npy', embeddings)

print(f"\n{'='*60}")
print(f"SUCCESS! Shape: {embeddings.shape}")
print(f"Total time: {(time.time()-start)/3600:.1f} hours")
print(f"{'='*60}")
