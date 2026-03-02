import pickle
import random
from dataclasses import dataclass
from typing import Callable, Dict, Generator, List, Tuple

import joblib
import matplotlib.pyplot as plt
import simpy
import streamlit as st

from phase1_simulation import (
    EdgeServer,
    IoTDevice,
    LEOSatellite,
    estimate_edge_queue_time,
    ms_to_time_units,
)
from phase5_visualization import DFA_Controller


# --- Energy constants (Joules) ---

ENERGY_TX_EDGE = 0.5
ENERGY_TX_LEO = 2.5
ENERGY_AI_MLP = 1.2
ENERGY_AI_SOM = 0.1


@dataclass
class RunResult:
    latencies: List[float]
    energy_j: float
    som_bypass_tasks: int = 0
    dfa_overrides: int = 0  # number of times DFA entered cooldown (stabilized)


@st.cache_resource
def load_models():
    edge_model = joblib.load("mlp_edge.joblib")
    leo_model = joblib.load("mlp_leo.joblib")
    with open("som_network_state.pkl", "rb") as f:
        som_bundle = pickle.load(f)
    return edge_model, leo_model, som_bundle


def build_congested_network(env: simpy.Environment, edge_speed_multiplier: float) -> Tuple[EdgeServer, LEOSatellite]:
    """
    Same Phase 2 bottlenecks, except Edge processing speed is scaled by the multiplier.
    """
    edge = EdgeServer(
        env=env,
        name="Edge-1",
        compute_capacity=1,
        processing_rate_mb_per_time=10.0 * float(edge_speed_multiplier),
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


def make_static_allocator(latencies: List[float], energy: Dict[str, float]) -> Callable:
    def allocator(
        env: simpy.Environment,
        task: IoTDevice.Task,
        edge_server: EdgeServer,
        leo_satellite: LEOSatellite,
    ) -> Generator:
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

        yield env.timeout(ms_to_time_units(tx_latency_ms))

        with resource.request() as req:
            yield req
            yield env.timeout(task.compute_size_mb / processing_rate)

        latencies.append(env.now - task.created_at)

    return allocator


def make_pure_ai_allocator(
    edge_model,
    leo_model,
    latencies: List[float],
    energy: Dict[str, float],
    avg_task_size_mb: float,
) -> Callable:
    def allocator(
        env: simpy.Environment,
        task: IoTDevice.Task,
        edge_server: EdgeServer,
        leo_satellite: LEOSatellite,
    ) -> Generator:
        task_size = float(task.compute_size_mb)
        current_leo_latency = float(leo_satellite.latency_ms(env.now))
        current_edge_q = float(estimate_edge_queue_time(edge_server, avg_task_size_mb))

        # Pure AI always runs MLPs
        energy["total"] += ENERGY_AI_MLP
        features = [[task_size, current_edge_q, current_leo_latency]]
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

        yield env.timeout(ms_to_time_units(tx_latency_ms))

        with resource.request() as req:
            yield req
            yield env.timeout(task.compute_size_mb / processing_rate)

        latencies.append(env.now - task.created_at)

    return allocator


def make_hybrid_allocator(
    edge_model,
    leo_model,
    som_bundle: dict,
    latencies: List[float],
    energy: Dict[str, float],
    avg_task_size_mb: float,
    dfa: DFA_Controller,
    som_bypass_counter: Dict[str, int],
    dfa_override_counter: Dict[str, int],
) -> Callable:
    som = som_bundle["som"]
    som_scaler = som_bundle["scaler"]
    cluster_labels = som_bundle["cluster_labels"]

    def allocator(
        env: simpy.Environment,
        task: IoTDevice.Task,
        edge_server: EdgeServer,
        leo_satellite: LEOSatellite,
    ) -> Generator:
        # DFA override (cool-down) has top priority
        forced = dfa.override_route_if_needed()
        if forced == "leo":
            energy["total"] += ENERGY_TX_LEO
            tx_latency_ms = float(leo_satellite.latency_ms(env.now))
            yield env.timeout(ms_to_time_units(tx_latency_ms))

            with leo_satellite.resource.request() as req:
                yield req
                yield env.timeout(task.compute_size_mb / leo_satellite.processing_rate_mb_per_time)

            latencies.append(env.now - task.created_at)
            return

        task_size = float(task.compute_size_mb)
        current_leo_latency = float(leo_satellite.latency_ms(env.now))
        current_edge_q = float(estimate_edge_queue_time(edge_server, avg_task_size_mb))

        # SOM gatekeeper (cheap)
        energy["total"] += ENERGY_AI_SOM
        som_vec_scaled = som_scaler.transform([[current_edge_q, current_leo_latency, task_size]])[0]
        winner = som.winner(som_vec_scaled)
        stress = cluster_labels.get(winner, "high")

        if stress == "low":
            som_bypass_counter["count"] += 1
            energy["total"] += ENERGY_TX_EDGE
            tx_latency_ms = edge_server.base_latency_ms
            resource = edge_server.resource
            processing_rate = edge_server.processing_rate_mb_per_time
        else:
            # High stress: run MLPs
            energy["total"] += ENERGY_AI_MLP
            features = [[task_size, current_edge_q, current_leo_latency]]
            pred_edge = float(edge_model.predict(features)[0])
            pred_leo = float(leo_model.predict(features)[0])

            if pred_edge <= pred_leo:
                dfa.observe_mlp_decision("edge")
                dfa_override_counter["count"] = dfa.overrides
                energy["total"] += ENERGY_TX_EDGE
                tx_latency_ms = edge_server.base_latency_ms
                resource = edge_server.resource
                processing_rate = edge_server.processing_rate_mb_per_time
            else:
                dfa.observe_mlp_decision("leo")
                dfa_override_counter["count"] = dfa.overrides
                energy["total"] += ENERGY_TX_LEO
                tx_latency_ms = current_leo_latency
                resource = leo_satellite.resource
                processing_rate = leo_satellite.processing_rate_mb_per_time

        yield env.timeout(ms_to_time_units(tx_latency_ms))

        with resource.request() as req:
            yield req
            yield env.timeout(task.compute_size_mb / processing_rate)

        latencies.append(env.now - task.created_at)

    return allocator


def run_one(
    method: str,
    *,
    num_tasks: int,
    max_task_size_mb: float,
    edge_speed_multiplier: float,
    seed: int,
) -> RunResult:
    random.seed(seed)
    env = simpy.Environment()
    edge, leo = build_congested_network(env, edge_speed_multiplier=edge_speed_multiplier)

    latencies: List[float] = []
    energy = {"total": 0.0}

    min_task_mb = 5.0
    max_task_mb = float(max_task_size_mb)
    avg_task_mb = 0.5 * (min_task_mb + max_task_mb)

    if method == "static":
        allocator = make_static_allocator(latencies, energy)
        som_bypass = 0
        dfa_overrides = 0
    else:
        edge_model, leo_model, som_bundle = load_models()

        if method == "ai":
            allocator = make_pure_ai_allocator(edge_model, leo_model, latencies, energy, avg_task_mb)
            som_bypass = 0
            dfa_overrides = 0
        elif method == "hybrid":
            dfa = DFA_Controller()
            som_bypass_counter = {"count": 0}
            dfa_override_counter = {"count": 0}
            allocator = make_hybrid_allocator(
                edge_model,
                leo_model,
                som_bundle,
                latencies,
                energy,
                avg_task_mb,
                dfa,
                som_bypass_counter,
                dfa_override_counter,
            )
            # capture after env.run()
            som_bypass = None
            dfa_overrides = None
        else:
            raise ValueError(f"Unknown method: {method}")

    IoTDevice(
        env=env,
        name=f"IoT-{method}",
        edge_server=edge,
        leo_satellite=leo,
        task_interval=3.0,
        allocator=allocator,
        max_tasks=num_tasks,
        verbose=False,
        min_compute_mb=min_task_mb,
        max_compute_mb=max_task_mb,
    )

    env.run()

    if method == "hybrid":
        som_bypass = som_bypass_counter["count"]
        dfa_overrides = dfa_override_counter["count"]

    return RunResult(
        latencies=latencies,
        energy_j=float(energy["total"]),
        som_bypass_tasks=som_bypass,
        dfa_overrides=dfa_overrides,
    )


def run_all(
    *,
    num_tasks: int,
    max_task_size_mb: float,
    edge_speed_multiplier: float,
    seed: int = 123,
) -> Tuple[RunResult, RunResult, RunResult]:
    static = run_one(
        "static",
        num_tasks=num_tasks,
        max_task_size_mb=max_task_size_mb,
        edge_speed_multiplier=edge_speed_multiplier,
        seed=seed,
    )
    ai = run_one(
        "ai",
        num_tasks=num_tasks,
        max_task_size_mb=max_task_size_mb,
        edge_speed_multiplier=edge_speed_multiplier,
        seed=seed,
    )
    hybrid = run_one(
        "hybrid",
        num_tasks=num_tasks,
        max_task_size_mb=max_task_size_mb,
        edge_speed_multiplier=edge_speed_multiplier,
        seed=seed,
    )
    return static, ai, hybrid


def build_figure(static: RunResult, ai: RunResult, hybrid: RunResult) -> plt.Figure:
    n = min(len(static.latencies), len(ai.latencies), len(hybrid.latencies))
    tasks = list(range(1, n + 1))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.plot(tasks, static.latencies[:n], color="red", label="Static")
    ax1.plot(tasks, ai.latencies[:n], color="blue", label="Pure AI (MLP)")
    ax1.plot(tasks, hybrid.latencies[:n], color="green", label="Hybrid (SOM+MLP+DFA)")
    ax1.set_xlabel("Task Number")
    ax1.set_ylabel("Total Latency (s)")
    ax1.set_title("Per-Task Latency")
    ax1.legend()

    avg_static = sum(static.latencies) / max(1, len(static.latencies))
    avg_ai = sum(ai.latencies) / max(1, len(ai.latencies))
    avg_hybrid = sum(hybrid.latencies) / max(1, len(hybrid.latencies))

    ax2.bar(["Static", "AI", "Hybrid"], [avg_static, avg_ai, avg_hybrid], color=["red", "blue", "green"])
    ax2.set_ylabel("Average Latency (s)")
    ax2.set_title("Average Latency")

    ax3.bar(
        ["Static", "AI", "Hybrid"],
        [static.energy_j, ai.energy_j, hybrid.energy_j],
        color=["red", "blue", "green"],
    )
    ax3.set_ylabel("Total Energy (J)")
    ax3.set_title("Total IoT Energy")

    fig.tight_layout()
    return fig


def main() -> None:
    st.title("Green 6G Edge-LEO Routing Simulator")

    st.markdown(
        """
This app simulates **Green 6G task offloading** across a congested terrestrial **Edge** server and a dynamic-latency **LEO satellite**.

**Hybrid SOM + MLP + DFA architecture**
- **SOM (Kohonen)**: quickly classifies the current network state into **low-stress / high-stress** clusters.  
  In *low-stress*, it **bypasses** heavy inference and routes directly to Edge (energy-efficient).
- **MLPs**: when the SOM detects *high-stress*, two trained regressors predict end-to-end latency for **Edge** and **LEO**, and choose the lower.
- **DFA controller**: monitors **route oscillations** (ping-pong) in MLP decisions and can enforce a **cool-down** period routing to LEO to stabilize the system.
"""
    )

    st.sidebar.header("Simulation Controls")
    task_count = st.sidebar.slider("Task Count", min_value=100, max_value=1000, value=500, step=50)
    max_task_size = st.sidebar.slider("Max Task Size (MB)", min_value=10, max_value=100, value=50, step=5)
    edge_speed = st.sidebar.slider("Edge Processing Speed (multiplier)", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

    if st.button("Run Network Simulation"):
        # Ensure models exist before running
        try:
            load_models()
        except FileNotFoundError as e:
            st.error(f"Missing model file: {e}")
            st.stop()

        with st.spinner("Running simulations (Static vs Pure AI vs Hybrid)..."):
            static, ai, hybrid = run_all(
                num_tasks=task_count,
                max_task_size_mb=max_task_size,
                edge_speed_multiplier=edge_speed,
                seed=123,
            )

        fig = build_figure(static, ai, hybrid)
        st.pyplot(fig)

        # Metrics row
        c1, c2, c3 = st.columns(3)
        c1.metric("Tasks Bypassed by SOM", f"{hybrid.som_bypass_tasks}")
        c2.metric("Total Energy Saved (J)", f"{(ai.energy_j - hybrid.energy_j):.2f}")
        c3.metric("DFA Overrides", f"{hybrid.dfa_overrides}")


if __name__ == "__main__":
    main()

