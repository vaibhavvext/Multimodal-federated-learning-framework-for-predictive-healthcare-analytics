import pandas as pd
import os

LABEL_COL = 'Diagnosis'
CLIENT_FILES = ['data/client_1.csv', 'data/client_2.csv', 'data/client_3.csv']

for file in CLIENT_FILES:
    if not os.path.exists(file):
        print(f"❌ File missing: {file}")
        continue
        
    df = pd.read_csv(file)
    
    # Strip whitespace from column names just in case
    df.columns = df.columns.str.strip()
    
    if LABEL_COL in df.columns:
        counts = df[LABEL_COL].value_counts()
        print(f"--- {file} ---")
        print(f"Total rows: {len(df)}")
        print(f"Class Distribution:\n{counts}")
        print("-" * 20)
    else:
        print(f"❌ Error: '{LABEL_COL}' not found in {file}")
        print(f"Available columns: {df.columns.tolist()}")