import pandas as pd
import numpy as np
import os

DATA_PATH = "data/asthma_disease_data.csv"
OUT_DIR = "data"
SEED = 42

# Ensure the output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Recommended: Drop non-numeric/admin columns before splitting 
# This ensures your FL clients don't crash during training later
df = df.drop(columns=['PatientID', 'DoctorInCharge'], errors='ignore')

df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Split into 3 clients
clients = np.array_split(df, 3)

for i, cdf in enumerate(clients, 1):
    out = os.path.join(OUT_DIR, f"client_{i}.csv")
    
    # Explicitly convert to DataFrame to avoid AttributeError
    client_df = pd.DataFrame(cdf) 
    
    client_df.to_csv(out, index=False)
    print(f"Saved: {out} | Shape: {client_df.shape}")

print("\n✅ All clients ready for Federated Learning.")