import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

def init_model(random_state=42):
    return SGDClassifier(
        loss="log_loss",
        max_iter=1,
        learning_rate="constant",
        eta0=0.01,
        random_state=random_state,
    )

def get_weights(model):
    return model.coef_.copy().astype(np.float32), model.intercept_.copy().astype(np.float32)

def set_weights(model, weights):
    coef, intercept = weights
    model.coef_ = coef.astype(np.float32)
    model.intercept_ = intercept.astype(np.float32)

def load_client_csv(path, label_col):
    df = pd.read_csv(path)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {path}")

    # Drop common non-feature identifier columns
    drop_cols = []
    for c in df.columns:
        cl = c.strip().lower()
        if cl in {"patientid", "patient_id", "id"}:
            drop_cols.append(c)

    # Separate X/y
    y = df[label_col].values
    X_df = df.drop(columns=[label_col] + drop_cols)

    # Convert non-numeric columns (except label) using factorization
    for c in X_df.columns:
        if X_df[c].dtype == object:
            X_df[c] = pd.factorize(X_df[c].astype(str))[0]

    X = X_df.values.astype(np.float32)

    # Convert labels to numeric binary if needed
    if y.dtype == object:
        y_str = pd.Series(y).astype(str).str.lower()
        if set(y_str.unique()).issubset({"0","1"}):
            y = y_str.astype(int).values
        elif set(y_str.unique()).issubset({"yes","no"}):
            y = (y_str == "yes").astype(int).values
        elif set(y_str.unique()).issubset({"true","false"}):
            y = (y_str == "true").astype(int).values
        else:
            y = pd.factorize(y_str)[0].astype(int)

    y = y.astype(int)

    return X, y

def fit_scaler_on_all(client_data):
    allX = np.vstack([X for (X, _) in client_data]).astype(np.float32)
    scaler = StandardScaler()
    allX_scaled = scaler.fit_transform(allX).astype(np.float32)
    return scaler, allX_scaled

def transform_clients_with_scaler(client_data, scaler):
    out = []
    for X, y in client_data:
        Xs = scaler.transform(X).astype(np.float32)
        out.append((Xs, y))
    return out
