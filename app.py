import streamlit as st
import numpy as np
import pandas as pd
import os
import time

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import SGDClassifier

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="MMFL Realtime Dashboard", layout="wide")
st.title("🏥 Multimodal Federated Learning – Realtime Dashboard")
st.caption("Live simulation of federated learning across hospitals")

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

SLEEP = {"Fast": 0.2, "Medium": 0.5, "Slow": 1.0}[speed]

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

# ---------------------------
# UI PLACEHOLDERS
# ---------------------------
round_status = st.empty()
hospital_status = st.empty()
server_status = st.empty()
progress = st.progress(0)

chart_placeholder = st.empty()
acc_log = []

# ---------------------------
# RUN REALTIME FL
# ---------------------------
if st.button("▶️ Start Federated Learning"):

    # Initialize global model
    global_model = init_model()
    global_model.partial_fit(
        X_parts[0], y_parts[0], classes=np.array([0, 1])
    )
    global_weights = get_weights(global_model)

    for rnd in range(num_rounds):
        round_status.markdown(f"## 🔁 Federated Round {rnd + 1}")

        progress.progress((rnd + 1) / num_rounds)

        client_weights = []
        accs = []

        # ---- Local Training Phase ----
        for i in range(3):
            hospital_status.markdown(
                f"🏥 **Hospital {i+1}** training local model..."
            )
            time.sleep(SLEEP)

            local_model = init_model()
            set_weights(local_model, global_weights)

            X, y = X_parts[i], y_parts[i]
            local_model.partial_fit(X, y, classes=np.array([0, 1]))

            acc = local_model.score(X, y)
            accs.append(acc)
            client_weights.append(get_weights(local_model))

            hospital_status.markdown(
                f"🏥 Hospital {i+1} sent update to server ✔️"
            )
            time.sleep(SLEEP)

        # ---- Aggregation Phase ----
        server_status.markdown("🧠 **Server aggregating updates...**")
        time.sleep(SLEEP)

        global_weights = average_weights(client_weights)
        server_status.markdown("🧠 Aggregation complete ✔️")
        time.sleep(SLEEP)

        # ---- Metrics Update ----
        avg_acc = float(np.mean(accs))
        acc_log.append(avg_acc)

        chart_placeholder.line_chart(acc_log)

    st.success("✅ Federated Learning Completed")
