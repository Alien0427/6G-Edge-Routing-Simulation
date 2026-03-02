import random
import pickle
from typing import Deque, List, Tuple
from collections import deque

import joblib
import matplotlib.pyplot as plt
import simpy

from phase1_simulation import (
    EdgeServer,
    LEOSatellite,
    IoTDevice,
    estimate_edge_queue_time,
    ms_to_time_units,
)

# --- Energy constants (Joules) for Green 6G metric ---

ENERGY_TX_EDGE = 0.5  # Energy to transmit to Edge
ENERGY_TX_LEO = 2.5  # Energy to transmit to Satellite
ENERGY_AI_MLP = 1.2  # Energy consumed to run the MLP prediction
ENERGY_AI_SOM = 0.1  # Energy consumed to run the lightweight SOM


# --- Shared network configuration (same bottlenecks as Phase 2 / AI sim) ---


class DFA_Controller:
    """
    Deterministic Finite Automaton (DFA) to stabilize oscillatory routing.

    States:
      - q0: Stable
      - q1: Warning
      - q2: Cool-down (force next N tasks to LEO)

    Logic:
      - Track last 5 routing decisions made by the MLPs.
      - Count route switches within that window.
      - If switches >= 3:
          q0 -> q1
          q1 -> q2 (start cooldown for next 10 tasks)
      - In q2: override and force next 10 tasks to LEO, then q2 -> q0.
    """

    def __init__(self, history_len: int = 5, cooldown_tasks: int = 10) -> None:
        self.state = "q0"
        self.history: Deque[str] = deque(maxlen=history_len)
        self.cooldown_tasks = cooldown_tasks
        self.cooldown_remaining = 0
        self.overrides = 0  # number of times we entered q2

    @staticmethod
    def _switch_count(decisions: List[str]) -> int:
        switches = 0
        for i in range(1, len(decisions)):
            switches += int(decisions[i] != decisions[i - 1])
        return switches

    def observe_mlp_decision(self, decision: str) -> None:
        self.history.append(decision)

        if len(self.history) < self.history.maxlen:
            return

        switches = self._switch_count(list(self.history))
        oscillating = switches >= 3

        if self.state == "q0":
            if oscillating:
                self.state = "q1"
        elif self.state == "q1":
            if oscillating:
                self.state = "q2"
                self.cooldown_remaining = self.cooldown_tasks
                self.overrides += 1
                self.history.clear()
            else:
                # recovered
                self.state = "q0"

    def override_route_if_needed(self) -> str | None:
        """
        If in cooldown, force LEO for this task and decrement counter.
        Returns 'leo' when overriding, otherwise None.
        """
        if self.state != "q2":
            return None

        if self.cooldown_remaining <= 0:
            self.state = "q0"
            return None

        self.cooldown_remaining -= 1
        if self.cooldown_remaining == 0:
            self.state = "q0"
        return "leo"


def build_congested_network(env: simpy.Environment) -> Tuple[EdgeServer, LEOSatellite]:
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

    return edge, leo


# --- Static allocator (baseline) with latency recording ---


def make_static_allocator_with_logging(latencies: List[float], energy: dict):
    def allocator(env: simpy.Environment, task: IoTDevice.Task, edge_server: EdgeServer, leo_satellite: LEOSatellite):
        # Static policy: try Edge first, overflow to LEO when Edge queue is "full"
        use_edge = len(edge_server.resource.queue) < edge_server.max_queue_length

        if use_edge:
            energy["total"] += ENERGY_TX_EDGE
            resource = edge_server.resource
            processing_rate = edge_server.processing_rate_mb_per_time
            tx_latency_ms = edge_server.base_latency_ms
        else:
            energy["total"] += ENERGY_TX_LEO
            resource = leo_satellite.resource
            processing_rate = leo_satellite.processing_rate_mb_per_time
            tx_latency_ms = leo_satellite.latency_ms(env.now)

        # Transmission
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
        latencies.append(total_latency)

    return allocator


def run_static_simulation(num_tasks: int = 500, task_interval: float = 3.0, seed: int = 123) -> Tuple[List[float], float]:
    random.seed(seed)
    env = simpy.Environment()

    edge, leo = build_congested_network(env)
    latencies: List[float] = []
    energy = {"total": 0.0}
    allocator = make_static_allocator_with_logging(latencies, energy)

    IoTDevice(
        env=env,
        name="IoT-Static",
        edge_server=edge,
        leo_satellite=leo,
        task_interval=task_interval,
        allocator=allocator,
        max_tasks=num_tasks,
        verbose=False,
    )

    env.run()
    return latencies, float(energy["total"])


# --- AI allocator with latency recording ---


def make_ai_allocator_with_logging(edge_model, leo_model, latencies: List[float], energy: dict):
    """
    AI-driven allocator identical in behavior to Phase 4, but records
    total latency instead of printing every detail (printing is handled
    here if needed).
    """

    def allocator(env: simpy.Environment, task: IoTDevice.Task, edge_server: EdgeServer, leo_satellite: LEOSatellite):
        # Extract current network state
        task_size = float(task.compute_size_mb)
        current_leo_latency = float(leo_satellite.latency_ms(env.now))
        # Same queue-time proxy as earlier (avg task size ≈ 27.5 MB)
        current_edge_queue_time = float(estimate_edge_queue_time(edge_server, 27.5))

        features = [[task_size, current_edge_queue_time, current_leo_latency]]

        # Pure AI: always spend energy running the MLP decision
        energy["total"] += ENERGY_AI_MLP

        pred_edge = float(edge_model.predict(features)[0])
        pred_leo = float(leo_model.predict(features)[0])

        if pred_edge <= pred_leo:
            energy["total"] += ENERGY_TX_EDGE
            resource = edge_server.resource
            processing_rate = edge_server.processing_rate_mb_per_time
            tx_latency_ms = edge_server.base_latency_ms
        else:
            energy["total"] += ENERGY_TX_LEO
            resource = leo_satellite.resource
            processing_rate = leo_satellite.processing_rate_mb_per_time
            tx_latency_ms = current_leo_latency

        # Transmission
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
        latencies.append(total_latency)

    return allocator


def run_ai_simulation(num_tasks: int = 500, task_interval: float = 3.0, seed: int = 123) -> Tuple[List[float], float]:
    random.seed(seed)
    env = simpy.Environment()

    edge, leo = build_congested_network(env)
    edge_model = joblib.load("mlp_edge.joblib")
    leo_model = joblib.load("mlp_leo.joblib")

    latencies: List[float] = []
    energy = {"total": 0.0}
    allocator = make_ai_allocator_with_logging(edge_model, leo_model, latencies, energy)

    IoTDevice(
        env=env,
        name="IoT-AI",
        edge_server=edge,
        leo_satellite=leo,
        task_interval=task_interval,
        allocator=allocator,
        max_tasks=num_tasks,
        verbose=False,
    )

    env.run()
    return latencies, float(energy["total"])


# --- Hybrid allocator (SOM + MLP) with latency recording ---


def make_hybrid_allocator_with_logging(
    edge_model,
    leo_model,
    som_bundle,
    latencies: List[float],
    bypass_counter: dict,
    energy: dict,
):
    som = som_bundle["som"]
    som_scaler = som_bundle["scaler"]
    cluster_labels = som_bundle["cluster_labels"]

    def allocator(env: simpy.Environment, task: IoTDevice.Task, edge_server: EdgeServer, leo_satellite: LEOSatellite):
        # DFA override has highest priority: in cooldown, force routing to LEO
        forced = bypass_counter["dfa"].override_route_if_needed()
        if forced == "leo":
            # Cool-down: override AI entirely (no SOM/MLP compute), just transmit to LEO
            energy["total"] += ENERGY_TX_LEO
            resource = leo_satellite.resource
            processing_rate = leo_satellite.processing_rate_mb_per_time
            tx_latency_ms = float(leo_satellite.latency_ms(env.now))

            tx_delay = ms_to_time_units(tx_latency_ms)
            yield env.timeout(tx_delay)

            with resource.request() as req:
                yield req
                processing_time = task.compute_size_mb / processing_rate
                yield env.timeout(processing_time)

            total_latency = env.now - task.created_at
            latencies.append(total_latency)
            return

        task_size = float(task.compute_size_mb)
        current_leo_latency = float(leo_satellite.latency_ms(env.now))
        current_edge_queue_time = float(estimate_edge_queue_time(edge_server, 27.5))

        # SOM features order used in training: [edge_queue_time, leo_latency, task_size]
        energy["total"] += ENERGY_AI_SOM
        som_vec_scaled = som_scaler.transform([[current_edge_queue_time, current_leo_latency, task_size]])[0]
        winner = som.winner(som_vec_scaled)
        stress_label = cluster_labels.get(winner, "high")

        if stress_label == "low":
            # Bypass MLPs and route directly to Edge
            bypass_counter["som_bypass"] += 1
            energy["total"] += ENERGY_TX_EDGE
            resource = edge_server.resource
            processing_rate = edge_server.processing_rate_mb_per_time
            tx_latency_ms = edge_server.base_latency_ms
        else:
            # SOM triggers the heavy MLP decision
            energy["total"] += ENERGY_AI_MLP
            features = [[task_size, current_edge_queue_time, current_leo_latency]]
            pred_edge = float(edge_model.predict(features)[0])
            pred_leo = float(leo_model.predict(features)[0])

            if pred_edge <= pred_leo:
                energy["total"] += ENERGY_TX_EDGE
                resource = edge_server.resource
                processing_rate = edge_server.processing_rate_mb_per_time
                tx_latency_ms = edge_server.base_latency_ms
                mlp_decision = "edge"
            else:
                energy["total"] += ENERGY_TX_LEO
                resource = leo_satellite.resource
                processing_rate = leo_satellite.processing_rate_mb_per_time
                tx_latency_ms = current_leo_latency
                mlp_decision = "leo"

            # DFA observes MLP decisions to detect oscillation (affects future tasks)
            bypass_counter["dfa"].observe_mlp_decision(mlp_decision)

        # Transmission
        tx_delay = ms_to_time_units(tx_latency_ms)
        yield env.timeout(tx_delay)

        # Queue + processing
        with resource.request() as req:
            yield req
            processing_time = task.compute_size_mb / processing_rate
            yield env.timeout(processing_time)

        total_latency = env.now - task.created_at
        latencies.append(total_latency)

    return allocator


def run_hybrid_simulation(num_tasks: int = 500, task_interval: float = 3.0, seed: int = 123) -> Tuple[List[float], float]:
    random.seed(seed)
    env = simpy.Environment()

    edge, leo = build_congested_network(env)
    edge_model = joblib.load("mlp_edge.joblib")
    leo_model = joblib.load("mlp_leo.joblib")

    # IMPORTANT: use the unsupervised SOM gatekeeper for bypass/energy savings
    with open("som_network_state.pkl", "rb") as f:
        som_bundle = pickle.load(f)

    latencies: List[float] = []
    bypass_counter = {"som_bypass": 0, "dfa": DFA_Controller()}
    energy = {"total": 0.0}
    allocator = make_hybrid_allocator_with_logging(edge_model, leo_model, som_bundle, latencies, bypass_counter, energy)

    IoTDevice(
        env=env,
        name="IoT-Hybrid",
        edge_server=edge,
        leo_satellite=leo,
        task_interval=task_interval,
        allocator=allocator,
        max_tasks=num_tasks,
        verbose=False,
    )

    env.run()
    print(f"Hybrid SOM bypass count: {bypass_counter['som_bypass']} / {num_tasks}")
    print(f"DFA Overrides: {bypass_counter['dfa'].overrides} - System stabilized {bypass_counter['dfa'].overrides} times.")
    return latencies, float(energy["total"])


# --- Phase 5: Visualization ---


def main() -> None:
    num_tasks = 500
    seed = 123
    task_interval = 3.0

    print(f"Running StaticAllocator simulation for {num_tasks} tasks...")
    static_latencies, static_energy = run_static_simulation(num_tasks=num_tasks, task_interval=task_interval, seed=seed)

    print(f"Running pure AIAllocator (MLP-only) simulation for {num_tasks} tasks with same seed...")
    ai_latencies, ai_energy = run_ai_simulation(num_tasks=num_tasks, task_interval=task_interval, seed=seed)

    print(f"Running HybridAllocator (SOM + MLP) simulation for {num_tasks} tasks with same seed...")
    hybrid_latencies, hybrid_energy = run_hybrid_simulation(num_tasks=num_tasks, task_interval=task_interval, seed=seed)

    # Ensure all runs produced the expected number of tasks
    if len(static_latencies) != num_tasks or len(ai_latencies) != num_tasks or len(hybrid_latencies) != num_tasks:
        print(
            f"Warning: expected {num_tasks} tasks but got "
            f"{len(static_latencies)} (static), {len(ai_latencies)} (ai), and {len(hybrid_latencies)} (hybrid)."
        )

    tasks = list(range(1, min(len(static_latencies), len(ai_latencies), len(hybrid_latencies)) + 1))

    # Graph A: per-task latency comparison (line chart)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.plot(tasks, static_latencies[: len(tasks)], color="red", label="Static Allocator")
    ax1.plot(tasks, ai_latencies[: len(tasks)], color="blue", label="AI Allocator (MLP-only)")
    ax1.plot(tasks, hybrid_latencies[: len(tasks)], color="green", label="Hybrid Allocator (SOM+MLP)")
    ax1.set_xlabel("Task Number")
    ax1.set_ylabel("Total Latency (s)")
    ax1.set_title("Per-Task Latency: Static vs AI vs Hybrid")
    ax1.legend()

    # Graph B: average latency comparison (bar chart)
    avg_static = sum(static_latencies) / len(static_latencies)
    avg_ai = sum(ai_latencies) / len(ai_latencies)
    avg_hybrid = sum(hybrid_latencies) / len(hybrid_latencies)

    ax2.bar(["Static", "AI", "Hybrid"], [avg_static, avg_ai, avg_hybrid], color=["red", "blue", "green"])
    ax2.set_ylabel("Average Total Latency (s)")
    ax2.set_title("Average Latency Comparison")

    # Graph C: total energy consumption (bar chart)
    ax3.bar(
        ["Static", "AI", "Hybrid"],
        [static_energy, ai_energy, hybrid_energy],
        color=["red", "blue", "green"],
    )
    ax3.set_ylabel("Total Energy Consumed (Joules)")
    ax3.set_title("Energy Consumption Comparison")

    plt.tight_layout()
    plt.savefig("final_results.png", dpi=300)
    plt.close(fig)

    print("Saved visualization to 'final_results.png'.")


if __name__ == "__main__":
    main()

