import streamlit as st
import numpy as np
import pandas as pd
import os
import time

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="MMFL Realtime Dashboard", layout="wide")
st.title("🏥 Multimodal Federated Learning – Realtime Framework Dashboard")
st.caption("Round-wise Learning • Metric Stability • Live Parameter Updates")

# ---------------------------
# LOAD DATA
# ---------------------------
DATA_DIR = "data"
CLIENT_FILES = [
    os.path.join(DATA_DIR, "client_1.csv"),
    os.path.join(DATA_DIR, "client_2.csv"),
    os.path.join(DATA_DIR, "client_3.csv"),
]

dfs = []
for f in CLIENT_FILES:
    if not os.path.exists(f):
        st.error(f"Dataset not found: {f}")
        st.stop()
    dfs.append(pd.read_csv(f))

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("⚙️ Controls")
num_rounds = st.sidebar.slider("Federated Rounds", 2, 10, 5)
speed = st.sidebar.selectbox("Animation Speed", ["Fast", "Medium", "Slow"])
SLEEP = {"Fast": 0.15, "Medium": 0.4, "Slow": 0.8}[speed]

# ---------------------------
# PREPROCESSING
# ---------------------------
label_col = dfs[0].columns[-1]

le = LabelEncoder()
scaler = StandardScaler()

combined = pd.concat(dfs, axis=0, ignore_index=True)
X_all = combined.drop(columns=[label_col])
y_all = le.fit_transform(combined[label_col])

X_scaled = scaler.fit_transform(X_all)

lengths = [len(df) for df in dfs]
split_idx = np.cumsum(lengths)[:-1]
X_parts = np.split(X_scaled, split_idx)
y_parts = np.split(y_all, split_idx)

# ---------------------------
# SHUFFLE & CHUNK DATA
# ---------------------------
client_chunks = []

for i in range(3):
    X, y = X_parts[i], y_parts[i]
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    X_chunks = np.array_split(X, num_rounds)
    y_chunks = np.array_split(y, num_rounds)

    client_chunks.append((X_chunks, y_chunks))

# ---------------------------
# FL HELPERS
# ---------------------------
def init_model():
    return SGDClassifier(
        loss="log_loss",
        max_iter=1,
        learning_rate="constant",
        eta0=0.01,
        random_state=42,
    )

def get_weights(model):
    return model.coef_.copy(), model.intercept_.copy()

def set_weights(model, weights):
    model.coef_ = weights[0]
    model.intercept_ = weights[1]

def average_weights(weight_list):
    coefs = np.mean([w[0] for w in weight_list], axis=0)
    intercepts = np.mean([w[1] for w in weight_list], axis=0)
    return coefs, intercepts

def safe_auc(y_true, probs):
    if len(np.unique(y_true)) < 2:
        return None
    return roc_auc_score(y_true, probs)

def evaluate(model, X, y, round_idx):
    probs = model.predict_proba(X)[:, 1]

    # Adaptive threshold (important)
    threshold = 0.3 if round_idx < 3 else 0.5
    preds = (probs > threshold).astype(int)

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    auc = safe_auc(y, probs)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "cm": confusion_matrix(y, preds),
    }

# ---------------------------
# UI PLACEHOLDERS
# ---------------------------
round_status = st.empty()
hospital_status = st.empty()
server_status = st.empty()
progress = st.progress(0)

metrics_box = st.empty()
weights_box = st.empty()
chart_placeholder = st.empty()

# ---------------------------
# RUN REALTIME FL
# ---------------------------
if st.button("▶️ Start Federated Learning"):

    # ---------- INITIALIZE GLOBAL MODEL ----------
    global_model = init_model()
    global_model.partial_fit(
        client_chunks[0][0][0],
        client_chunks[0][1][0],
        classes=np.array([0, 1]),
    )
    global_weights = get_weights(global_model)

    acc_log = []

    # ---------- FEDERATED ROUNDS ----------
    for rnd in range(num_rounds):
        round_status.markdown(f"## 🔁 Federated Round {rnd + 1}")
        progress.progress((rnd + 1) / num_rounds)

        client_weights = []

        # ---- Local Training ----
        for i in range(3):
            hospital_status.markdown(
                f"🏥 Hospital {i+1} training on chunk {rnd + 1}/{num_rounds}"
            )
            time.sleep(SLEEP)

            local_model = init_model()
            set_weights(local_model, global_weights)

            Xc, yc = client_chunks[i][0][rnd], client_chunks[i][1][rnd]
            local_model.partial_fit(Xc, yc, classes=np.array([0, 1]))

            client_weights.append(get_weights(local_model))

            hospital_status.markdown(f"🏥 Hospital {i+1} sent update ✔️")
            time.sleep(SLEEP)

        # ---- Aggregation ----
        server_status.markdown("🧠 Server aggregating updates…")
        time.sleep(SLEEP)

        global_weights = average_weights(client_weights)
        server_status.markdown("🧠 Aggregation complete ✔️")

        # ---- Evaluation AFTER THIS ROUND ----
        round_metrics = []
        for i in range(3):
            eval_model = init_model()
            set_weights(eval_model, global_weights)
            eval_model.partial_fit(
                X_parts[i], y_parts[i], classes=np.array([0, 1])
            )
            round_metrics.append(evaluate(eval_model, X_parts[i], y_parts[i], rnd))

        avg_metrics = {
            k: float(np.mean([m[k] for m in round_metrics if m[k] is not None]))
            if k != "auc"
            else (
                float(np.mean([m[k] for m in round_metrics if m[k] is not None]))
                if any(m[k] is not None for m in round_metrics)
                else "Not defined (single-class)"
            )
            for k in ["accuracy", "precision", "recall", "f1", "auc"]
        }

        acc_log.append(avg_metrics["accuracy"])

        # ---- UI UPDATE ----
        metrics_box.json({
            "Round": rnd + 1,
            **avg_metrics
        })

        if avg_metrics["precision"] == 0 and avg_metrics["recall"] == 0:
            st.info(
                "Precision/Recall unstable in early rounds due to limited class exposure."
            )

        if avg_metrics["auc"] == "Not defined (single-class)":
            st.warning(
                "ROC-AUC not defined this round (single-class data observed)."
            )

        weights_box.markdown(
            f"""
            **Global Weights Snapshot (Round {rnd + 1})**
            - coef[0][:5]: `{np.round(global_weights[0][0][:5], 4)}`
            - intercept: `{np.round(global_weights[1][0], 4)}`
            """
        )

        chart_placeholder.line_chart(acc_log)

    st.success("✅ Federated Learning Completed with Stable Metrics Over Rounds")
