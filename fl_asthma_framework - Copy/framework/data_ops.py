import pandas as pd

def append_row(csv_path, row_dict):
    df = pd.read_csv(csv_path)

    missing = [c for c in df.columns if c not in row_dict]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Keep only known columns, maintain order
    new_row = {c: row_dict[c] for c in df.columns}
    df.loc[len(df)] = new_row
    df.to_csv(csv_path, index=False)

def append_row_from_csv(csv_path, one_row_df):
    df = pd.read_csv(csv_path)
    if len(one_row_df) != 1:
        raise ValueError("Uploaded CSV must contain exactly 1 row.")

    for c in df.columns:
        if c not in one_row_df.columns:
            raise ValueError(f"Missing column '{c}' in uploaded row CSV.")

    df.loc[len(df)] = [one_row_df.iloc[0][c] for c in df.columns]
    df.to_csv(csv_path, index=False)
