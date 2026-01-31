import streamlit as st
import pandas as pd
import json
import os

from framework.registry_ops import load_registry, save_registry, add_hospital, remove_hospital
from framework.data_ops import append_row, append_row_from_csv
from framework.trainer import run_training

st.set_page_config(page_title="FL Framework Dashboard", layout="wide")
st.title("🧠 Federated Learning Framework Dashboard (Local)")

REG_PATH = "data/hospitals.json"
HOSP_DIR = "data/hospitals"
STATE_DIR = "state"
OUT_DIR = "outputs"

os.makedirs(HOSP_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

reg = load_registry(REG_PATH)

st.sidebar.header("⚙️ Training Controls")
algo = st.sidebar.selectbox("Aggregation Algorithm", ["fedavg", "weighted_fedavg"])
rounds = st.sidebar.slider("Rounds", 2, 20, 5)
seed = st.sidebar.number_input("Seed", 0, 999999, 42)

tabs = st.tabs(["Hospitals", "Data Manager", "Training", "Results"])

# -------------------------
# TAB 1: Hospitals
# -------------------------
with tabs[0]:
    st.subheader("🏥 Hospitals Registry")
    st.write(f"Label Column: `{reg['label_col']}`")
    st.dataframe(pd.DataFrame(reg["hospitals"]))

    st.markdown("### ➕ Add Hospital (Upload CSV)")
    new_id = st.text_input("Hospital ID (unique, e.g., H4)")
    new_name = st.text_input("Hospital Name")
    up = st.file_uploader("Upload Hospital Dataset CSV", type=["csv"])

    if st.button("Add Hospital"):
        if not new_id or not new_name or up is None:
            st.error("Provide ID, name, and CSV.")
        else:
            out_path = os.path.join(HOSP_DIR, f"{new_id}.csv")
            with open(out_path, "wb") as f:
                f.write(up.getbuffer())

            reg2 = add_hospital(reg, new_id, new_name, out_path.replace("\\", "/"))
            save_registry(reg2, REG_PATH)
            st.success("Hospital added. Restart dashboard to refresh.")
            st.stop()

    st.markdown("### ➖ Remove Hospital")
    rm_id = st.selectbox("Select Hospital ID", [h["id"] for h in reg["hospitals"]])
    if st.button("Remove Selected Hospital"):
        reg2 = remove_hospital(reg, rm_id)
        save_registry(reg2, REG_PATH)
        st.success("Hospital removed. Restart dashboard to refresh.")
        st.stop()

# -------------------------
# TAB 2: Data Manager
# -------------------------
with tabs[1]:
    st.subheader("📂 Data Manager")
    hid = st.selectbox("Select Hospital", [h["id"] for h in reg["hospitals"]])
    h = next(x for x in reg["hospitals"] if x["id"] == hid)

    df = pd.read_csv(h["path"])
    st.write(f"Dataset: `{h['path']}`")
    st.dataframe(df.head(50))

    st.markdown("### ➕ Add Row (Upload 1-row CSV)")
    row_up = st.file_uploader("Upload a 1-row CSV (must have same columns)", type=["csv"], key="rowcsv")
    if st.button("Append Uploaded Row"):
        if row_up is None:
            st.error("Upload a CSV with exactly 1 row.")
        else:
            one = pd.read_csv(row_up)
            append_row_from_csv(h["path"], one)
            st.success("Row appended. Reload tab to see changes.")
            st.stop()

    st.markdown("### ➕ Add Row (Manual k=v)")
    st.caption("Example: Age=25,Gender=1,...,ExerciseInduced=0")
    row_txt = st.text_area("Enter row as comma-separated k=v pairs")
    if st.button("Append Manual Row"):
        if not row_txt.strip():
            st.error("Enter row values.")
        else:
            parts = [p.strip() for p in row_txt.split(",") if p.strip()]
            row = {}
            for p in parts:
                k, v = p.split("=")
                row[k.strip()] = v.strip()
            append_row(h["path"], row)
            st.success("Row appended. Reload tab to see changes.")
            st.stop()

# -------------------------
# TAB 3: Training
# -------------------------
with tabs[2]:
    st.subheader("🏃 Run Federated Learning")
    st.write("This will update global weights + client weights and write logs to `state/run_log.jsonl`.")

    log_box = st.empty()
    metric_box = st.empty()

    def callback(ev):
        # show live events + round metrics
        if ev["event"] == "round_metrics":
            metric_box.json(ev)
        log_box.write(ev)

    if st.button("Start Training Now"):
        metrics = run_training(
            hospitals=reg["hospitals"],
            label_col=reg["label_col"],
            algo=algo,
            rounds=rounds,
            seed=seed,
            state_dir=STATE_DIR,
            outputs_dir=OUT_DIR,
            callback=callback,
        )
        st.success("Training complete.")
        st.dataframe(pd.DataFrame(metrics))

# -------------------------
# TAB 4: Results
# -------------------------
with tabs[3]:
    st.subheader("📊 Results")

    metrics_path = os.path.join(OUT_DIR, "metrics.csv")
    report_path = os.path.join(OUT_DIR, "report.json")
    log_path = os.path.join(STATE_DIR, "run_log.jsonl")

    if os.path.exists(metrics_path):
        dfm = pd.read_csv(metrics_path)
        st.dataframe(dfm)
        st.line_chart(dfm.set_index("round")[["accuracy", "f1"]])

        if "auc" in dfm.columns:
            st.line_chart(dfm.set_index("round")[["auc"]])
    else:
        st.warning("No metrics.csv found yet. Run training first.")

    st.markdown("### Global Weights Snapshot")
    gw_path = os.path.join(STATE_DIR, "global_weights.npz")
    if os.path.exists(gw_path):
        import numpy as np
        d = np.load(gw_path)
        st.write("coef[0][:5]:", d["coef"][0][:5])
        st.write("intercept:", d["intercept"])
    else:
        st.warning("No global weights saved yet.")

    st.markdown("### Background Log (tail)")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-30:]
        st.code("".join(lines))
    else:
        st.warning("No run_log.jsonl found yet.")
