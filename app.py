import streamlit as st
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import SGDClassifier

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="MMFL Framework Dashboard", layout="wide")
st.title("🏥 Multimodal Federated Learning Framework")
st.caption("Framework Simulation + Monitoring Dashboard (Streamlit-safe)")

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
st.sidebar.header("⚙️ Framework Controls")
num_rounds = st.sidebar.slider("Federated Rounds", 1, 10, 5)

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
# RUN FEDERATED LEARNING (ONCE)
# ---------------------------
if "trained" not in st.session_state:
    st.session_state.trained = False

if st.button("🚀 Run Federated Learning"):

    acc_log = []

    # Initialize global model
    global_model = init_model()
    global_model.partial_fit(
        X_parts[0],
        y_parts[0],
        classes=np.array([0, 1])
    )

    global_weights = get_weights(global_model)
    initial_weights = global_weights

    progress = st.progress(0)
    status = st.empty()

    for rnd in range(num_rounds):
        status.markdown(f"### 🔁 Federated Round {rnd + 1}/{num_rounds}")
        progress.progress((rnd + 1) / num_rounds)

        client_weights = []
        accs = []

        for i in range(3):
            local_model = init_model()
            set_weights(local_model, global_weights)

            X, y = X_parts[i], y_parts[i]
            local_model.partial_fit(X, y, classes=np.array([0, 1]))

            client_weights.append(get_weights(local_model))
            accs.append(local_model.score(X, y))

        global_weights = average_weights(client_weights)
        acc_log.append(float(np.mean(accs)))

    # Store results
    st.session_state.trained = True
    st.session_state.acc_log = acc_log
    st.session_state.initial_weights = initial_weights
    st.session_state.final_weights = global_weights

    st.success("Federated training completed successfully.")

# ---------------------------
# DASHBOARD VIEW (READ-ONLY)
# ---------------------------
if st.session_state.trained:

    st.divider()
    st.subheader("📊 Federated Learning Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Accuracy Across Rounds")
        st.line_chart(st.session_state.acc_log)

    with col2:
        st.markdown("### Model Drift (Sample Weights)")
        st.write("Initial weights:")
        st.write(np.round(st.session_state.initial_weights[0][0][:5], 4))
        st.write("Final weights:")
        st.write(np.round(st.session_state.final_weights[0][0][:5], 4))

    st.info(
        "Dashboard is read-only. Raw hospital data and local models remain private."
    )
