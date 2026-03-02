import joblib
import pickle
import simpy

from phase1_simulation import (
    EdgeServer,
    LEOSatellite,
    IoTDevice,
    estimate_edge_queue_time,
    ms_to_time_units,
)


def make_hybrid_ai_allocator(edge_model, leo_model, som_bundle, counters: dict):
    """
    Hybrid AI allocator using a SOM + MLPs.

    - First, classify the current network state with the SOM.
    - If the neuron is labeled as "low-stress", bypass the MLPs and go
      directly to Edge (cheap decision).
    - If "high-stress", call the MLPs to predict Edge vs LEO latency and
      pick the better route.
    """

    som = som_bundle["som"]
    som_scaler = som_bundle["scaler"]
    cluster_labels = som_bundle["cluster_labels"]

    def allocator(env, task: IoTDevice.Task, edge_server: EdgeServer, leo_satellite: LEOSatellite):
        # Extract current network state (same features used for SOM training)
        task_size = float(task.compute_size_mb)
        current_leo_latency = float(leo_satellite.latency_ms(env.now))
        current_edge_queue_time = float(estimate_edge_queue_time(edge_server, 27.5))

        som_features = [current_edge_queue_time, current_leo_latency, task_size]
        som_vec_scaled = som_scaler.transform([som_features])[0]
        winner = som.winner(som_vec_scaled)
        stress_label = cluster_labels.get(winner, "high")

        used_mlp = False

        if stress_label == "low":
            # Low-stress network state: cheap heuristic, always Edge
            chosen_node = "edge"
            pred_edge = pred_leo = None
            resource = edge_server.resource
            processing_rate = edge_server.processing_rate_mb_per_time
            tx_latency_ms = edge_server.base_latency_ms
            counters["som_bypass"] += 1
        else:
            # High-stress: use full MLP prediction pipeline
            current_edge_queue_time_for_mlp = current_edge_queue_time
            mlp_features = [
                [
                    task_size,
                    current_edge_queue_time_for_mlp,
                    current_leo_latency,
                ]
            ]

            pred_edge = float(edge_model.predict(mlp_features)[0])
            pred_leo = float(leo_model.predict(mlp_features)[0])
            used_mlp = True
            counters["mlp_calls"] += 1

            if pred_edge <= pred_leo:
                chosen_node = "edge"
                resource = edge_server.resource
                processing_rate = edge_server.processing_rate_mb_per_time
                tx_latency_ms = edge_server.base_latency_ms
            else:
                chosen_node = "leo"
                resource = leo_satellite.resource
                processing_rate = leo_satellite.processing_rate_mb_per_time
                tx_latency_ms = current_leo_latency

        if chosen_node == "edge":
            counters["edge"] += 1
        else:
            counters["leo"] += 1

        # Simulate actual execution path
        t_gen = env.now

        # Transmission delay
        tx_delay = ms_to_time_units(tx_latency_ms)
        yield env.timeout(tx_delay)

        # Queue + processing
        queue_enter_time = env.now
        with resource.request() as req:
            yield req
            queue_delay = env.now - queue_enter_time

            processing_time = task.compute_size_mb / processing_rate
            yield env.timeout(processing_time)

        total_latency = env.now - task.created_at

        # Log outcome
        if used_mlp:
            print(
                f"[t={t_gen:6.1f}] Task {task.id} | "
                f"SOM={stress_label} | "
                f"AI Pred Edge={pred_edge:6.3f}s, AI Pred LEO={pred_leo:6.3f}s -> "
                f"Routed: {chosen_node.upper()} | Actual={total_latency:6.3f}s"
            )
        else:
            print(
                f"[t={t_gen:6.1f}] Task {task.id} | "
                f"SOM={stress_label} (bypass MLPs) -> "
                f"Routed: {chosen_node.upper()} | Actual={total_latency:6.3f}s"
            )

    return allocator


def run_ai_simulation(num_tasks: int = 500, task_interval: float = 3.0) -> None:
    """
    Phase 4: AI-driven simulation using pre-trained MLP models.

    Inherits the same environment, nodes, and bottleneck parameters from
    the congested Phase 2 simulation.
    """
    # Load trained models
    edge_model = joblib.load("mlp_edge.joblib")
    leo_model = joblib.load("mlp_leo.joblib")

    # Use unsupervised SOM gatekeeper for bypass/energy savings
    with open("som_network_state.pkl", "rb") as f:
        som_bundle = pickle.load(f)

    env = simpy.Environment()

    # Congested Edge / LEO configuration (same as Phase 2)
    edge = EdgeServer(
        env=env,
        name="Edge-1",
        compute_capacity=1,  # single processing slot to force queuing
        processing_rate_mb_per_time=10.0,
        base_latency_ms=10.0,
        max_queue_length=3,
    )

    leo = LEOSatellite(
        env=env,
        name="LEO-1",
        compute_capacity=2,
        orbital_period=200.0,
        min_latency_ms=20.0,
        max_latency_ms=200.0,
        processing_rate_mb_per_time=50.0,
        max_queue_length=20,
    )

    counters = {"edge": 0, "leo": 0, "som_bypass": 0, "mlp_calls": 0}
    allocator = make_hybrid_ai_allocator(edge_model, leo_model, som_bundle, counters)

    IoTDevice(
        env=env,
        name="IoT-AI",
        edge_server=edge,
        leo_satellite=leo,
        task_interval=task_interval,
        allocator=allocator,
        max_tasks=num_tasks,
        verbose=False,  # suppress generator logs; AIAllocator prints instead
    )

    print(f"Starting AI-driven simulation for {num_tasks} tasks...")
    env.run()

    print("\n=== Hybrid AI Routing Summary ===")
    print(f"Total tasks routed to EDGE: {counters['edge']}")
    print(f"Total tasks routed to LEO : {counters['leo']}")
    total_tasks = counters["edge"] + counters["leo"]
    print(f"Total tasks: {total_tasks}")
    print(f"SOM bypassed MLPs for {counters['som_bypass']} tasks.")
    print(f"MLPs were evaluated for {counters['mlp_calls']} tasks.")
    if total_tasks > 0:
        bypass_pct = 100.0 * counters["som_bypass"] / total_tasks
        print(f"Percentage of tasks bypassing MLPs: {bypass_pct:.2f}%")


if __name__ == "__main__":
    run_ai_simulation()

