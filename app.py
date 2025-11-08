import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Federated Learning Simulation", layout="wide")
st.title("🏥 Multimodal Federated Learning Simulation")
st.markdown("""
Simulate **Federated Learning across 3 Hospitals** where each hospital trains a local model, 
sends model weights to a central server, which aggregates them and sends updated weights back.
""")

# ---------------------------
# LOAD LOCAL CSV FILES
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

st.sidebar.header("⚙️ Simulation Settings")
num_rounds = st.sidebar.slider("Communication Rounds", 1, 10, 5)
local_epochs = st.sidebar.slider("Local Epochs", 1, 5, 2)

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def create_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def get_weights(model):
    return model.get_weights()

def set_weights(model, weights):
    model.set_weights(weights)

def average_weights(weight_list):
    avg = []
    for weights in zip(*weight_list):
        avg.append(np.mean(np.array(weights), axis=0))
    return avg

# ---------------------------
# PREPROCESSING
# ---------------------------
label_col = dfs[0].columns[-1]  # assume last column is label

le = LabelEncoder()
scaler = StandardScaler()

combined = pd.concat(dfs, axis=0)
X_all = combined.drop(columns=[label_col])
y_all = le.fit_transform(combined[label_col])
X_scaled = scaler.fit_transform(X_all)

# Split back per client
lengths = [len(df) for df in dfs]
split_idx = np.cumsum(lengths)[:-1]
X_parts = np.split(X_scaled, split_idx)
y_parts = np.split(y_all, split_idx)
input_dim = X_parts[0].shape[1]

# ---------------------------
# FEDERATED TRAINING
# ---------------------------
global_model = create_model(input_dim)
initial_weights = get_weights(global_model)

progress = st.progress(0)
status = st.empty()
col1, col2 = st.columns([2, 1])

# Visualization setup
with col1:
    st.subheader("🧠 Federated Learning Animation")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    hospital_pos = [(2, 8), (2, 5), (2, 2)]
    server_pos = (8, 5)
    ax.text(server_pos[0], server_pos[1]+0.5, "Server", ha='center', fontsize=12, color='red')
    for i, (x, y) in enumerate(hospital_pos, 1):
        ax.text(x, y+0.5, f"Hospital {i}", ha='center', fontsize=10, color='blue')
    client_dots = [ax.plot(x, y, 'bo', markersize=10)[0] for x, y in hospital_pos]
    server_dot = ax.plot(server_pos[0], server_pos[1], 'ro', markersize=12)[0]
    lines = [ax.plot([], [], 'g--')[0] for _ in range(3)]
    anim_placeholder = st.pyplot(fig)

# Logs
acc_log = []

# Federated rounds
for rnd in range(num_rounds):
    progress.progress((rnd + 1) / num_rounds)
    status.markdown(f"### 🔁 Round {rnd+1}/{num_rounds}")
    client_weights = []
    accs = []

    for i in range(3):
        local_model = create_model(input_dim)
        set_weights(local_model, get_weights(global_model))
        X, y = X_parts[i], y_parts[i]
        local_model.fit(X, y, epochs=local_epochs, verbose=0)
        client_weights.append(get_weights(local_model))
        _, acc = local_model.evaluate(X, y, verbose=0)
        accs.append(acc)

        # animate: line to server
        lines[i].set_data([hospital_pos[i][0], server_pos[0]], [hospital_pos[i][1], server_pos[1]])
        anim_placeholder.pyplot(fig)
        time.sleep(0.3)

    # server aggregates
    new_weights = average_weights(client_weights)
    set_weights(global_model, new_weights)

    # animate: send back
    for i in range(3):
        lines[i].set_data([], [])
        anim_placeholder.pyplot(fig)
        time.sleep(0.2)

    avg_acc = np.mean(accs)
    acc_log.append(avg_acc)

# ---------------------------
# RESULTS
# ---------------------------
final_weights = get_weights(global_model)

with col2:
    st.subheader("📊 Weights Comparison")
    st.write("**Initial (sample):**")
    st.text(np.round(initial_weights[0][:2], 4))
    st.write("**Final (sample):**")
    st.text(np.round(final_weights[0][:2], 4))

st.subheader("📈 Accuracy Trend")
st.line_chart(acc_log)
st.success("✅ Training Complete!")
