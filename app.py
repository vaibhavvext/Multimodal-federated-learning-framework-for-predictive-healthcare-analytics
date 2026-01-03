import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import os

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Federated Learning Simulation", layout="wide")
st.title("🏥 Lightweight Federated Learning Simulation")

st.markdown("""
This app simulates **Federated Learning across 3 hospitals**
using a lightweight Logistic Regression model.
""")

# ---------------------------
# LOAD DATA
# ---------------------------
DATA_DIR = "data"
CLIENT_FILES = [
    os.path.join(DATA_DIR, "client_1.csv"),
    os.path.join(DATA_DIR, "client_2.csv"),
    os.path.join(DATA_DIR, "client_3.csv")
]

dfs = []
for f in CLIENT_FILES:
    if not os.path.exists(f):
        st.error(f"❌ Dataset not found: {f}")
        st.stop()
    dfs.append(pd.read_csv(f))

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("⚙️ Settings")
num_rounds = st.sidebar.slider("Federated Rounds", 1, 10, 5)

# ---------------------------
# PREPROCESSING
# ---------------------------
label_col = dfs[0].columns[-1]

le = LabelEncoder()
scaler = StandardScaler()

combined = pd.concat(dfs, axis=0)
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
    return LogisticRegression(
        max_iter=200,
        solver="lbfgs"
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
# INITIALIZE GLOBAL WEIGHTS SAFELY
# ---------------------------
global_weights = None

for i in range(3):
    if len(np.unique(y_parts[i])) > 1:
        model = init_model()
        model.fit(X_parts[i], y_parts[i])
        global_weights = get_weights(model)
        break

if global_weights is None:
    st.error("All clients have single-class data. Cannot train Logistic Regression.")
    st.stop()

initial_weights = global_weights
acc_log = []

# ---------------------------
# FEDERATED TRAINING
# ---------------------------
progress = st.progress(0)
status = st.empty()

for rnd in range(num_rounds):
    status.markdown(f"### 🔁 Federated Round {rnd + 1}/{num_rounds}")
    progress.progress((rnd + 1) / num_rounds)

    client_weights = []
    accs = []

    for i in range(3):
        # Skip invalid clients
        if len(np.unique(y_parts[i])) < 2:
            continue

        local_model = init_model()
        set_weights(local_model, global_weights)

        X, y = X_parts[i], y_parts[i]
        local_model.fit(X, y)

        client_weights.append(get_weights(local_model))
        accs.append(local_model.score(X, y))

    global_weights = average_weights(client_weights)
    acc_log.append(np.mean(accs))

# ---------------------------
# RESULTS
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Accuracy Trend")
    st.line_chart(acc_log)

with col2:
    st.subheader("⚖️ Weight Comparison (Sample)")
    st.write("**Initial weights (first 5):**")
    st.write(np.round(initial_weights[0][0][:5], 4))
    st.write("**Final weights (first 5):**")
    st.write(np.round(global_weights[0][0][:5], 4))

st.success("✅ Federated Learning Simulation Complete!")
