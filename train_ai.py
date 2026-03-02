import os
import pickle

import joblib
import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = ["task_size", "current_edge_queue_time", "current_leo_latency"]
TARGET = "total_experienced_latency"


def derive_best_node_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Supervised label for LVQ: which node should be chosen for *lower latency*.

    The training CSV does not contain counterfactual "latency if sent to the other node",
    so we derive a deterministic proxy label using the known simulation parameters:

      Edge proxy latency = tx_edge + current_edge_queue_time + (task_size / edge_rate)
      LEO  proxy latency = tx_leo  + (task_size / leo_rate)
    """
    task_size = df["task_size"].to_numpy(dtype=float)
    edge_q_time = df["current_edge_queue_time"].to_numpy(dtype=float)
    leo_lat_ms = df["current_leo_latency"].to_numpy(dtype=float)

    tx_edge_s = 10.0 / 1000.0  # Edge base latency (10 ms) in our congested config
    edge_rate = 10.0  # Edge processing_rate_mb_per_time

    tx_leo_s = leo_lat_ms / 1000.0
    leo_rate = 50.0  # LEO processing_rate_mb_per_time

    edge_proxy = tx_edge_s + edge_q_time + (task_size / edge_rate)
    leo_proxy = tx_leo_s + (task_size / leo_rate)

    return np.where(edge_proxy <= leo_proxy, "edge", "leo")


def som_majority_vote_accuracy(som: MiniSom, X_scaled: np.ndarray, y_labels: np.ndarray) -> float:
    """
    Evaluate SOM as a classifier using majority-vote labels per neuron.
    """
    votes: dict[tuple[int, int], dict[str, int]] = {}
    for x_vec, y in zip(X_scaled, y_labels):
        bmu = som.winner(x_vec)
        d = votes.setdefault(bmu, {})
        y_str = str(y)
        d[y_str] = d.get(y_str, 0) + 1

    neuron_majority: dict[tuple[int, int], str] = {}
    for neuron, v in votes.items():
        neuron_majority[neuron] = max(v.items(), key=lambda kv: kv[1])[0]

    correct = 0
    for x_vec, y in zip(X_scaled, y_labels):
        bmu = som.winner(x_vec)
        pred = neuron_majority.get(bmu, "edge")
        correct += int(pred == str(y))

    return correct / max(1, len(y_labels))


def lvq1_fine_tune_som(
    som: MiniSom,
    X_scaled: np.ndarray,
    y_labels: np.ndarray,
    epochs: int = 10,
    alpha0: float = 0.3,
    seed: int = 42,
) -> dict[tuple[int, int], str]:
    """
    LVQ1 fine-tuning:
      - Find BMU
      - Reward (pull) BMU weights if BMU class matches label
      - Punish (push) otherwise

    Returns neuron->class labels after tuning (majority vote).
    """
    rng = np.random.default_rng(seed)

    # Initial neuron class labels (majority vote)
    init_votes: dict[tuple[int, int], dict[str, int]] = {}
    for x_vec, y in zip(X_scaled, y_labels):
        bmu = som.winner(x_vec)
        d = init_votes.setdefault(bmu, {})
        y_str = str(y)
        d[y_str] = d.get(y_str, 0) + 1

    neuron_label: dict[tuple[int, int], str] = {}
    for neuron, v in init_votes.items():
        neuron_label[neuron] = max(v.items(), key=lambda kv: kv[1])[0]

    n = len(X_scaled)
    indices = np.arange(n)

    for epoch in range(epochs):
        rng.shuffle(indices)
        alpha = alpha0 * (1.0 - (epoch / max(1, epochs)))

        for idx in indices:
            x = X_scaled[idx]
            y = str(y_labels[idx])

            bmu = som.winner(x)
            bmu_label = neuron_label.get(bmu, "edge")

            w = som._weights[bmu[0], bmu[1], :]
            if bmu_label == y:
                som._weights[bmu[0], bmu[1], :] = w + alpha * (x - w)
            else:
                som._weights[bmu[0], bmu[1], :] = w - alpha * (x - w)

    # Recompute neuron class labels after tuning (majority vote)
    final_votes: dict[tuple[int, int], dict[str, int]] = {}
    for x_vec, y in zip(X_scaled, y_labels):
        bmu = som.winner(x_vec)
        d = final_votes.setdefault(bmu, {})
        y_str = str(y)
        d[y_str] = d.get(y_str, 0) + 1

    final_neuron_label: dict[tuple[int, int], str] = {}
    for neuron, v in final_votes.items():
        final_neuron_label[neuron] = max(v.items(), key=lambda kv: kv[1])[0]

    return final_neuron_label

def train_one_model(df: pd.DataFrame, random_state: int = 42) -> Pipeline:
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    learning_rate_init=0.001,
                    max_iter=500,
                    random_state=random_state,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    print(f"Model trained. Test R^2 = {r2:.4f} on {len(X_test)} samples.")
    return model


def main() -> None:
    csv_path = "training_data.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find {csv_path!r}. Run 'python phase1_simulation.py' first to generate it."
        )

    df = pd.read_csv(csv_path)

    # Basic sanity checks
    missing = [c for c in (FEATURES + [TARGET, "chosen_node"]) if c not in df.columns]
    if missing:
        raise ValueError(f"training_data.csv is missing columns: {missing}")

    df_edge = df[df["chosen_node"] == "edge"].copy()
    df_leo = df[df["chosen_node"] == "leo"].copy()

    if len(df_edge) < 50 or len(df_leo) < 50:
        raise ValueError(
            f"Not enough samples per class. edge={len(df_edge)}, leo={len(df_leo)}. "
            "Increase num_tasks or adjust congestion so both routes occur."
        )

    print(f"Loaded {len(df)} rows: edge={len(df_edge)}, leo={len(df_leo)}")

    print("\nTraining Edge model...")
    edge_model = train_one_model(df_edge)
    joblib.dump(edge_model, "mlp_edge.joblib")
    print("Saved: mlp_edge.joblib")

    print("\nTraining LEO model...")
    leo_model = train_one_model(df_leo)
    joblib.dump(leo_model, "mlp_leo.joblib")
    print("Saved: mlp_leo.joblib")

    # --- Train SOM on overall network state (Hybrid Architecture) ---
    print("\nTraining Self-Organizing Map (SOM) on network state...")

    som_features = ["current_edge_queue_time", "current_leo_latency", "task_size"]
    X_som = df[som_features].values

    som_scaler = StandardScaler()
    X_som_scaled = som_scaler.fit_transform(X_som)

    som_dim = 3  # 3x3 SOM
    som = MiniSom(
        x=som_dim,
        y=som_dim,
        input_len=X_som_scaled.shape[1],
        sigma=1.0,
        learning_rate=0.5,
        neighborhood_function="gaussian",
        random_seed=42,
    )

    som.random_weights_init(X_som_scaled)
    som.train_random(X_som_scaled, num_iteration=10 * len(X_som_scaled))

    # Cluster neurons into "low-stress" vs "high-stress" states based on
    # average experienced latency of samples mapped to each neuron.
    neuron_latencies: dict[tuple[int, int], list[float]] = {}
    for x_vec, latency in zip(X_som_scaled, df[TARGET].values):
        winner = som.winner(x_vec)
        neuron_latencies.setdefault(winner, []).append(float(latency))

    neuron_avg_latency: dict[tuple[int, int], float] = {}
    for neuron, vals in neuron_latencies.items():
        neuron_avg_latency[neuron] = sum(vals) / len(vals)

    if neuron_avg_latency:
        global_median = sorted(neuron_avg_latency.values())[len(neuron_avg_latency) // 2]
    else:
        global_median = df[TARGET].median()

    cluster_labels: dict[tuple[int, int], str] = {}
    for neuron, avg_lat in neuron_avg_latency.items():
        cluster_labels[neuron] = "low" if avg_lat <= global_median else "high"

    print("SOM neuron average latencies and assigned stress levels:")
    for (i, j), avg_lat in sorted(neuron_avg_latency.items()):
        label = cluster_labels[(i, j)]
        print(f"  Neuron ({i},{j}): avg_latency={avg_lat:.3f}s -> {label}-stress")

    som_bundle = {
        "som": som,
        "scaler": som_scaler,
        "cluster_labels": cluster_labels,
        "features": som_features,
    }

    with open("som_network_state.pkl", "wb") as f:
        pickle.dump(som_bundle, f)

    print("Saved SOM model and metadata to 'som_network_state.pkl'.")

    # --- LVQ layer: supervised fine-tuning of SOM weights ---
    print("\nLVQ Upgrade: deriving supervised best-node labels...")
    y_best = derive_best_node_labels(df)

    raw_acc = som_majority_vote_accuracy(som, X_som_scaled, y_best)
    print(f"Raw SOM accuracy (majority-vote classifier): {raw_acc:.4f}")

    # Clone SOM weights into a new SOM instance for LVQ tuning
    som_lvq = MiniSom(
        x=som_dim,
        y=som_dim,
        input_len=X_som_scaled.shape[1],
        sigma=1.0,
        learning_rate=0.5,
        neighborhood_function="gaussian",
        random_seed=42,
    )
    som_lvq._weights = som._weights.copy()

    print("Running LVQ1 fine-tuning loop...")
    neuron_class_labels = lvq1_fine_tune_som(
        som_lvq,
        X_som_scaled,
        y_best,
        epochs=10,
        alpha0=0.3,
        seed=42,
    )

    lvq_acc = som_majority_vote_accuracy(som_lvq, X_som_scaled, y_best)
    print(f"LVQ-tuned SOM accuracy (majority-vote classifier): {lvq_acc:.4f}")

    lvq_bundle = {
        "som": som_lvq,
        "scaler": som_scaler,
        "features": som_features,
        # Maps BMU neuron -> predicted best node ("edge" / "leo")
        "neuron_class_labels": neuron_class_labels,
    }

    with open("lvq_som_network_state.pkl", "wb") as f:
        pickle.dump(lvq_bundle, f)

    print("Saved LVQ-tuned SOM bundle to 'lvq_som_network_state.pkl'.")


if __name__ == "__main__":
    main()

