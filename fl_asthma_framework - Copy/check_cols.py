import pandas as pd

try:
    # Loading the dataset from your data folder
    df = pd.read_csv("data/asthma_disease_data.csv")
    
    print("-" * 30)
    print("COLUMN LIST:")
    print(df.columns.tolist())
    print("-" * 30)
    print(f"Total Rows: {len(df)}")
    print(f"Total Cols: {len(df.columns)}")
    print(f"SUGGESTED LABEL (Last Col): {df.columns[-1]}")
    print("-" * 30)
    
    # Show a preview of the last few columns' values
    print("PREVIEW OF TARGET DATA:")
    print(df[df.columns[-1]].head())
    
except FileNotFoundError:
    print("Error: 'asthma_disease_data.csv' not found in the 'data' folder.")