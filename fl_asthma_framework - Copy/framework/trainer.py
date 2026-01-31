import os, json, time
import numpy as np
import pandas as pd

from framework.algorithms import fedavg, weighted_fedavg
from framework.clients import load_client_csv, init_model, set_weights, get_weights, fit_scaler_on_all, transform_clients_with_scaler
from framework.metrics import evaluate_binary
from framework.state import save_weights, load_weights, ensure_dir

def log_event(log_path, event):
    ensure_dir(os.path.dirname(log_path))
    event["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

def run_training(
    hospitals,
    label_col,
    algo="weighted_fedavg",
    rounds=5,
    seed=42,
    state_dir="state",
    outputs_dir="outputs",
    callback=None,
):
    ensure_dir(state_dir)
    ensure_dir(outputs_dir)

    log_path = os.path.join(state_dir, "run_log.jsonl")

    # Load datasets
    client_raw = []
    sizes = []
    for h in hospitals:
        X, y = load_client_csv(h["path"], label_col)
        client_raw.append((X, y))
        sizes.append(len(X))

    # Fit one scaler on all clients (aligned feature scaling)
    scaler, _ = fit_scaler_on_all(client_raw)
    client_data = transform_clients_with_scaler(client_raw, scaler)

    # Chunk per round (incremental learning)
    rng = np.random.default_rng(seed)
    chunks = []
    for X, y in client_data:
        perm = rng.permutation(len(X))
        X, y = X[perm], y[perm]
        X_chunks = np.array_split(X, rounds)
        y_chunks = np.array_split(y, rounds)
        chunks.append((X_chunks, y_chunks))

    global_path = os.path.join(state_dir, "global_weights.npz")
    gw = load_weights(global_path)

    if gw is None:
        # init global weights
        m0 = init_model(random_state=seed)
        m0.partial_fit(chunks[0][0][0], chunks[0][1][0], classes=np.array([0,1]))
        gw = get_weights(m0)
        save_weights(global_path, gw[0], gw[1])

    metrics_log = []

    for r in range(rounds):
        log_event(log_path, {"event":"round_start", "round": r+1, "algo": algo})
        if callback: callback({"event":"round_start", "round": r+1})

        client_updates = []
        for idx, h in enumerate(hospitals):
            Xc = chunks[idx][0][r]
            yc = chunks[idx][1][r]

            m = init_model(random_state=seed + idx + r)
            # load global weights
            m.coef_, m.intercept_ = gw[0].copy(), gw[1].copy()
            m.partial_fit(Xc, yc, classes=np.array([0,1]))

            upd = get_weights(m)
            client_updates.append(upd)

            save_weights(os.path.join(state_dir, f"{h['id']}_weights.npz"), upd[0], upd[1])

            log_event(log_path, {"event":"client_update", "round": r+1, "hospital": h["id"], "samples": int(len(Xc))})
            if callback: callback({"event":"client_update", "round": r+1, "hospital": h["id"]})

        # aggregate
        if algo == "fedavg":
            gw = fedavg(client_updates)
        else:
            gw = weighted_fedavg(client_updates, sizes)

        save_weights(global_path, gw[0], gw[1])

        log_event(log_path, {"event":"aggregate_done", "round": r+1})
        if callback: callback({"event":"aggregate_done", "round": r+1})

        # evaluate global weights on full client data
        m_eval = init_model(random_state=seed)
        m_eval.coef_, m_eval.intercept_ = gw[0].copy(), gw[1].copy()

        round_metrics = []
        for Xfull, yfull in client_data:
            probs = m_eval.predict_proba(Xfull)[:, 1]
            thr = 0.3 if r < 2 else 0.5
            round_metrics.append(evaluate_binary(yfull, probs, threshold=thr))

        avg = {
            "round": r+1,
            "accuracy": float(np.mean([m["accuracy"] for m in round_metrics])),
            "precision": float(np.mean([m["precision"] for m in round_metrics])),
            "recall": float(np.mean([m["recall"] for m in round_metrics])),
            "f1": float(np.mean([m["f1"] for m in round_metrics])),
            "auc": (
                float(np.mean([m["auc"] for m in round_metrics if m["auc"] is not None]))
                if any(m["auc"] is not None for m in round_metrics) else None
            ),
            "global_coef_head": gw[0][0][:5].tolist(),
            "global_intercept": float(gw[1][0]),
        }

        metrics_log.append(avg)

        log_event(log_path, {"event":"round_metrics", **avg})
        if callback: callback({"event":"round_metrics", **avg})

    # Save outputs
    pd.DataFrame(metrics_log).to_csv(os.path.join(outputs_dir, "metrics.csv"), index=False)
    with open(os.path.join(outputs_dir, "report.json"), "w") as f:
        json.dump(metrics_log, f, indent=2)

    return metrics_log
