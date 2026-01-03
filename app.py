import streamlit as st
import numpy as np
import pandas as pd
import os
import time

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="MMFL Realtime Dashboard", layout="wide")
st.title("🏥 Multimodal Federated Learning – Realtime Evaluation Dashboard")
st.caption("Rich Before vs After Evaluation + Live Federated Learning")

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
num_rounds = st.sidebar.slider("Federated Rounds", 1, 10, 5)
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

def evaluate(model, X, y):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "auc": roc_auc_score(y, probs),
        "cm": confusion_matrix(y, preds),
        "probs": probs,
    }

# ---------------------------
# UI PLACEHOLDERS
# ---------------------------
before_col, after_col = st.columns(2)
round_status = st.empty()
hospital_status = st.empty()
server_status = st.empty()
progress = st.progress(0)
chart_placeholder = st.empty()

# ---------------------------
# RUN REALTIME FL
# ---------------------------
if st.button("▶️ Start Federated Learning"):

    # ========= BEFORE FL =========
    before_metrics = []

    for i in range(3):
        model = init_model()
        model.partial_fit(X_parts[i], y_parts[i], classes=np.array([0, 1]))
        before_metrics.append(evaluate(model, X_parts[i], y_parts[i]))

    before_avg = {
        k: float(np.mean([m[k] for m in before_metrics if k != "cm" and k != "probs"]))
        for k in ["accuracy", "precision", "recall", "f1", "auc"]
    }

    with before_col:
        st.subheader("📉 Before Federated Learning")
        st.json(before_avg)
        st.write("Confusion Matrix (Hospital 1)")
        st.write(before_metrics[0]["cm"])

    # ========= INITIALIZE GLOBAL =========
    global_model = init_model()
    global_model.partial_fit(X_parts[0], y_parts[0], classes=np.array([0, 1]))
    global_weights = get_weights(global_model)

    acc_log = []

    # ========= FEDERATED ROUNDS (LIVE) =========
    for rnd in range(num_rounds):
        round_status.markdown(f"## 🔁 Federated Round {rnd + 1}")
        progress.progress((rnd + 1) / num_rounds)

        client_weights = []
        accs = []

        for i in range(3):
            hospital_status.markdown(f"🏥 Hospital {i+1} training locally…")
            time.sleep(SLEEP)

            local_model = init_model()
            set_weights(local_model, global_weights)

            X, y = X_parts[i], y_parts[i]
            local_model.partial_fit(X, y, classes=np.array([0, 1]))

            accs.append(local_model.score(X, y))
            client_weights.append(get_weights(local_model))

            hospital_status.markdown(f"🏥 Hospital {i+1} sent update ✔️")
            time.sleep(SLEEP)

        server_status.markdown("🧠 Server aggregating updates…")
        time.sleep(SLEEP)

        global_weights = average_weights(client_weights)
        server_status.markdown("🧠 Aggregation complete ✔️")

        acc_log.append(float(np.mean(accs)))
        chart_placeholder.line_chart(acc_log)

    # ========= AFTER FL =========
    after_metrics = []

    for i in range(3):
        model = init_model()
        set_weights(model, global_weights)
        model.partial_fit(X_parts[i], y_parts[i], classes=np.array([0, 1]))
        after_metrics.append(evaluate(model, X_parts[i], y_parts[i]))

    after_avg = {
        k: float(np.mean([m[k] for m in after_metrics if k != "cm" and k != "probs"]))
        for k in ["accuracy", "precision", "recall", "f1", "auc"]
    }

    with after_col:
        st.subheader("📈 After Federated Learning")
        st.json(after_avg)
        st.write("Confusion Matrix (Hospital 1)")
        st.write(after_metrics[0]["cm"])

    st.success("✅ Federated Learning Completed with Rich Evaluation")
