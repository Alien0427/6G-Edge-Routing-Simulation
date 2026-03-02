import simpy
import math
import random
from dataclasses import dataclass
from typing import Callable, Generator, Optional, Tuple

import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


@dataclass
class EdgeServer:
    """
    Edge server with fixed compute capacity.

    Capacity is an abstract measure (e.g., tasks per time unit).
    """

    env: simpy.Environment
    name: str
    compute_capacity: float
    processing_rate_mb_per_time: float = 25.0  # MB per simulation time unit
    base_latency_ms: float = 10.0  # one-way / effective edge latency
    max_queue_length: int = 10

    def __post_init__(self) -> None:
        # Use a SimPy Resource to represent compute slots
        self.resource = simpy.Resource(self.env, capacity=int(self.compute_capacity))


@dataclass
class LEOSatellite:
    """
    LEO satellite with higher compute capacity and time-varying latency.

    The latency between an IoT device and the satellite is modeled as a
    smooth periodic function over an orbital period:

    - Lowest latency when the satellite is overhead.
    - Highest latency when it is near the horizon.
    """

    env: simpy.Environment
    name: str
    compute_capacity: float
    orbital_period: float  # in simulation time units
    min_latency_ms: float  # minimum RTT when overhead
    max_latency_ms: float  # maximum RTT when at horizon
    processing_rate_mb_per_time: float = 50.0  # MB per simulation time unit
    max_queue_length: int = 20

    def __post_init__(self) -> None:
        self.resource = simpy.Resource(self.env, capacity=int(self.compute_capacity))

    def latency_ms(self, t: float | None = None) -> float:
        """
        Return the current latency (in ms) as a function of time.

        We model the satellite's motion using a cosine wave over the orbital period:

        latency(t) = min + 0.5 * (max - min) * (1 + cos(2π * phase))

        where phase = (t mod orbital_period) / orbital_period.
        This gives:
        - latency = min_latency_ms at t = orbital_period / 2 (overhead)
        - latency = max_latency_ms at t = 0, orbital_period (horizon)
        """
        if t is None:
            t = self.env.now

        if self.orbital_period <= 0:
            return self.max_latency_ms

        phase = (t % self.orbital_period) / self.orbital_period
        cos_term = math.cos(2 * math.pi * phase)
        latency = self.min_latency_ms + 0.5 * (self.max_latency_ms - self.min_latency_ms) * (1 + cos_term)
        return latency


def ms_to_time_units(latency_ms: float) -> float:
    """
    Convert milliseconds to simulation time units.

    We interpret one simulation time unit as one second, so:
    time_units = ms / 1000.
    """
    return latency_ms / 1000.0


def static_allocator(
    env: simpy.Environment,
    task: "IoTDevice.Task",
    edge_server: EdgeServer,
    leo_satellite: LEOSatellite,
) -> Generator:
    """
    Baseline "StaticAllocator" policy:

    - Try to offload to the EdgeServer first.
    - If the EdgeServer queue is full (>= max_queue_length), send to LEO.
    - For each task, compute:
        total_latency = transmission_latency + queuing_delay + processing_time
    """
    # Decide target: Edge first, fall back to LEO if Edge queue is "full"
    use_edge = len(edge_server.resource.queue) < edge_server.max_queue_length

    if use_edge:
        target_name = edge_server.name
        resource = edge_server.resource
        processing_rate = edge_server.processing_rate_mb_per_time
        tx_latency_ms = edge_server.base_latency_ms
    else:
        target_name = leo_satellite.name
        resource = leo_satellite.resource
        processing_rate = leo_satellite.processing_rate_mb_per_time
        tx_latency_ms = leo_satellite.latency_ms(env.now)

    # Transmission latency
    tx_delay = ms_to_time_units(tx_latency_ms)
    yield env.timeout(tx_delay)

    # Queuing and processing at the chosen compute node
    queue_enter_time = env.now
    with resource.request() as req:
        yield req
        queue_delay = env.now - queue_enter_time

        processing_time = task.compute_size_mb / processing_rate
        yield env.timeout(processing_time)

    completion_time = env.now
    total_latency = completion_time - task.created_at

    print(
        f"    -> Task {task.id} processed at {target_name} | "
        f"tx={tx_delay:5.3f}s, queue={queue_delay:5.3f}s, proc={processing_time:5.3f}s | "
        f"total={total_latency:5.3f}s"
    )


def _random_allocator_process(
    env: simpy.Environment,
    task: "IoTDevice.Task",
    edge_server: EdgeServer,
    leo_satellite: LEOSatellite,
    records: list[dict],
) -> Generator:
    """
    Generator implementing a random allocator used for data generation.

    - Randomly assigns each task to Edge or LEO, regardless of queue state.
    - Records the state at assignment time and the total experienced latency.
    """
    current_edge_queue = len(edge_server.resource.queue)
    satellite_latency_ms = leo_satellite.latency_ms(env.now)
    task_size_mb = task.compute_size_mb

    chosen_node = random.choice(["edge", "leo"])

    if chosen_node == "edge":
        target_name = edge_server.name
        resource = edge_server.resource
        processing_rate = edge_server.processing_rate_mb_per_time
        tx_latency_ms = edge_server.base_latency_ms
    else:
        target_name = leo_satellite.name
        resource = leo_satellite.resource
        processing_rate = leo_satellite.processing_rate_mb_per_time
        tx_latency_ms = satellite_latency_ms

    tx_delay = ms_to_time_units(tx_latency_ms)
    yield env.timeout(tx_delay)

    queue_enter_time = env.now
    with resource.request() as req:
        yield req
        queue_delay = env.now - queue_enter_time

        processing_time = task.compute_size_mb / processing_rate
        yield env.timeout(processing_time)

    completion_time = env.now
    total_latency = completion_time - task.created_at

    records.append(
        {
            "current_edge_queue": current_edge_queue,
            "satellite_latency_ms": satellite_latency_ms,
            "task_size_mb": task_size_mb,
            "chosen_node": chosen_node,
            "total_experienced_latency_s": total_latency,
        }
    )

    # Optional: comment out in production to reduce logs
    print(
        f"    [data] Task {task.id} -> {target_name} | "
        f"state=(queue={current_edge_queue}, sat_lat={satellite_latency_ms:6.1f}ms, size={task_size_mb:5.1f}MB) | "
        f"total={total_latency:5.3f}s"
    )


def make_random_data_allocator(records: list[dict]) -> Callable:
    """
    Factory that returns an allocator compatible with IoTDevice,
    which internally uses _random_allocator_process to generate data.
    """
    def allocator(
        env: simpy.Environment,
        task: "IoTDevice.Task",
        edge_server: EdgeServer,
        leo_satellite: LEOSatellite,
    ) -> Generator:
        return _random_allocator_process(env, task, edge_server, leo_satellite, records)

    return allocator


def estimate_edge_queue_time(
    edge_server: EdgeServer,
    avg_task_size_mb: float,
) -> float:
    """
    Estimate how long a *new* task would wait if sent to the Edge right now.

    We only have access to queue length + number currently in service, not
    remaining service times, so we use a simple proxy:

        estimated_wait ~= backlog_ahead * avg_service_time / capacity

    where backlog_ahead is the number of tasks that would be served before the new one.
    """
    capacity = max(1, int(edge_server.resource.capacity))
    backlog_in_system = edge_server.resource.count + len(edge_server.resource.queue)

    # Approximate how many tasks are "ahead" of a new arrival
    backlog_ahead = max(0, backlog_in_system - capacity + 1)

    avg_service_time = avg_task_size_mb / edge_server.processing_rate_mb_per_time
    return (backlog_ahead * avg_service_time) / capacity


def make_static_data_allocator(
    records: list[dict],
    avg_task_size_mb_for_queue_estimate: float,
    verbose: bool = False,
) -> Callable:
    """
    Static allocator + data collection.

    Records features at the moment of task generation/assignment:
      [task_size, current_edge_queue_time, current_leo_latency, chosen_node]
    and target:
      [total_experienced_latency]

    Output column names match the user's specification.
    """

    def allocator(
        env: simpy.Environment,
        task: "IoTDevice.Task",
        edge_server: EdgeServer,
        leo_satellite: LEOSatellite,
    ) -> Generator:
        # Capture state at assignment time (same as generation time for our model)
        task_size = task.compute_size_mb
        current_leo_latency = float(leo_satellite.latency_ms(env.now))
        current_edge_queue_time = float(
            estimate_edge_queue_time(edge_server, avg_task_size_mb_for_queue_estimate)
        )

        chosen_node = "edge" if len(edge_server.resource.queue) < edge_server.max_queue_length else "leo"

        if chosen_node == "edge":
            resource = edge_server.resource
            processing_rate = edge_server.processing_rate_mb_per_time
            tx_latency_ms = edge_server.base_latency_ms
        else:
            resource = leo_satellite.resource
            processing_rate = leo_satellite.processing_rate_mb_per_time
            tx_latency_ms = current_leo_latency

        # Transmission latency
        tx_delay = ms_to_time_units(tx_latency_ms)
        yield env.timeout(tx_delay)

        # Queuing and processing
        queue_enter_time = env.now
        with resource.request() as req:
            yield req
            queue_delay = env.now - queue_enter_time

            processing_time = task.compute_size_mb / processing_rate
            yield env.timeout(processing_time)

        total_experienced_latency = env.now - task.created_at

        records.append(
            {
                "task_size": task_size,
                "current_edge_queue_time": current_edge_queue_time,
                "current_leo_latency": current_leo_latency,
                "chosen_node": chosen_node,
                "total_experienced_latency": total_experienced_latency,
            }
        )

        if verbose:
            print(
                f"    [csv] Task {task.id} -> {chosen_node} | "
                f"feat=(size={task_size:5.1f}MB, edge_q={current_edge_queue_time:5.3f}s, leo_lat={current_leo_latency:6.1f}ms) | "
                f"total={total_experienced_latency:5.3f}s (queue={queue_delay:5.3f}s, tx={tx_delay:5.3f}s)"
            )

    return allocator


@dataclass
class IoTDevice:
    """
    IoT device that generates tasks over time.

    For Phase 1, this class primarily illustrates:
    - Time progression in the SimPy environment.
    - Interaction with Edge and LEO objects.
    """

    env: simpy.Environment
    name: str
    edge_server: EdgeServer
    leo_satellite: LEOSatellite
    task_interval: float  # baseline time between task generations
    min_compute_mb: float = 5.0
    max_compute_mb: float = 50.0
    allocator: Optional[
        Callable[
            [simpy.Environment, "IoTDevice.Task", EdgeServer, LEOSatellite],
            Generator,
        ]
    ] = None
    max_tasks: Optional[int] = None  # if set, stop after generating this many tasks
    verbose: bool = True

    def __post_init__(self) -> None:
        # Default to the static baseline allocator if none is provided
        if self.allocator is None:
            self.allocator = static_allocator
        self.process = self.env.process(self.run())

    @dataclass
    class Task:
        """
        Simple representation of a computational task generated by an IoT device.

        - compute_size_mb: amount of data / computation size in MB.
        - created_at: simulation timestamp when the task was created.
        - source: reference/name of the originating IoT device.
        """

        id: int
        compute_size_mb: float
        created_at: float
        source: str

    def _next_interarrival(self) -> float:
        """
        Generate an erratic inter-arrival time to simulate bursty traffic.

        Strategy:
        - With a small probability, create a short interval (burst).
        - Otherwise, vary around the baseline task_interval with randomness.
        """
        # Probability of entering a "burst" (very frequent tasks)
        if random.random() < 0.2:
            # Burst: very short interval (e.g., 10–30% of baseline)
            return max(0.1, self.task_interval * random.uniform(0.1, 0.3))

        # Normal mode: jittered around the baseline interval
        low = 0.5 * self.task_interval
        high = 1.5 * self.task_interval
        return random.uniform(low, high)

    def _generate_task(self, task_id: int) -> "IoTDevice.Task":
        """
        Create a new Task object with a random compute_size within the
        configured bounds.
        """
        compute_size = random.uniform(self.min_compute_mb, self.max_compute_mb)
        return IoTDevice.Task(
            id=task_id,
            compute_size_mb=compute_size,
            created_at=self.env.now,
            source=self.name,
        )

    def run(self):
        """
        Traffic generator for the IoT device.

        This process:
        - Waits for an erratic inter-arrival time.
        - Generates a Task with a random compute_size (MB).
        - Logs the task and current satellite latency.
        """
        task_id = 0
        while self.max_tasks is None or task_id < self.max_tasks:
            # Wait for the next (possibly bursty) arrival
            interarrival = self._next_interarrival()
            yield self.env.timeout(interarrival)
            task_id += 1

            current_time = self.env.now
            sat_latency = self.leo_satellite.latency_ms(current_time)
            task = self._generate_task(task_id)

            if self.verbose:
                print(
                    f"[t={current_time:6.1f}] {self.name} generated task {task.id} | "
                    f"size={task.compute_size_mb:5.1f} MB | "
                    f"LEO latency ~ {sat_latency:6.1f} ms"
                )

            # Send task to the selected allocation policy (as a SimPy process)
            self.env.process(
                self.allocator(
                    env=self.env,
                    task=task,
                    edge_server=self.edge_server,
                    leo_satellite=self.leo_satellite,
                )
            )


def run_phase1_demo(
    sim_time: float = 300.0,
    task_interval: float = 3.0,
) -> None:
    """
    Simple demonstration of the Phase 1 simulation environment.

    - Creates one Edge server with fixed capacity.
    - Creates one LEO satellite with higher capacity and time-varying latency.
    - Creates one IoT device that periodically generates tasks and logs
      the satellite latency as it "orbits".
    """
    env = simpy.Environment()

    edge = EdgeServer(
        env=env,
        name="Edge-1",
        compute_capacity=1,  # single processing slot to force queuing
        processing_rate_mb_per_time=10.0,  # slower processing to create congestion
        base_latency_ms=10.0,
        max_queue_length=3,  # small queue so tasks overflow to LEO
    )

    leo = LEOSatellite(
        env=env,
        name="LEO-1",
        compute_capacity=2,  # a bit more parallelism than edge
        orbital_period=200.0,  # time units for a full "orbit"
        min_latency_ms=20.0,
        max_latency_ms=200.0,
        processing_rate_mb_per_time=50.0,  # faster processing
        max_queue_length=20,
    )

    IoTDevice(
        env=env,
        name="IoT-1",
        edge_server=edge,
        leo_satellite=leo,
        task_interval=task_interval,
        # Use baseline static allocator in the demo
        allocator=static_allocator,
    )

    print("Starting Phase 1 simulation...")
    env.run(until=sim_time)
    print("Phase 1 simulation completed.")


def generate_training_data(
    num_tasks: int = 10_000,
    task_interval: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run the simulation to generate training data for the AI model.

    For each task, we record:
        - current_edge_queue
        - satellite_latency_ms
        - task_size_mb
        - chosen_node ("edge" or "leo")
    and the target:
        - total_experienced_latency_s

    Returns a Pandas DataFrame with one row per task.
    """
    random.seed(seed)

    env = simpy.Environment()

    edge = EdgeServer(
        env=env,
        name="Edge-train",
        compute_capacity=1,
        processing_rate_mb_per_time=10.0,
        base_latency_ms=10.0,
        max_queue_length=10,
    )

    leo = LEOSatellite(
        env=env,
        name="LEO-train",
        compute_capacity=2,
        orbital_period=200.0,
        min_latency_ms=20.0,
        max_latency_ms=200.0,
        processing_rate_mb_per_time=50.0,
        max_queue_length=50,
    )

    records: list[dict] = []
    allocator = make_random_data_allocator(records)

    IoTDevice(
        env=env,
        name="IoT-train",
        edge_server=edge,
        leo_satellite=leo,
        task_interval=task_interval,
        allocator=allocator,
        max_tasks=num_tasks,
    )

    print(f"Starting training data simulation for up to {num_tasks} tasks...")
    env.run()  # runs until there are no more events (all tasks processed)
    print(f"Training data simulation completed. Collected {len(records)} samples.")

    df = pd.DataFrame.from_records(records)
    return df


def collect_and_export_training_data_csv(
    output_csv: str = "training_data.csv",
    num_tasks: int = 5_000,
    task_interval: float = 3.0,
    seed: int = 42,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Collect per-task features + target and export to CSV.

    IMPORTANT: Congestion parameters are kept identical to run_phase1_demo().
    """
    random.seed(seed)

    env = simpy.Environment()

    # Keep congestion parameters exactly the same as the current demo
    edge = EdgeServer(
        env=env,
        name="Edge-1",
        compute_capacity=1,
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

    records: list[dict] = []

    # Use the device's own size range to estimate queue time proxy
    min_mb = 5.0
    max_mb = 50.0
    avg_mb = 0.5 * (min_mb + max_mb)

    allocator = make_static_data_allocator(
        records=records,
        avg_task_size_mb_for_queue_estimate=avg_mb,
        verbose=verbose,
    )

    IoTDevice(
        env=env,
        name="IoT-1",
        edge_server=edge,
        leo_satellite=leo,
        task_interval=task_interval,
        allocator=allocator,
        max_tasks=num_tasks,
        verbose=verbose,
        min_compute_mb=min_mb,
        max_compute_mb=max_mb,
    )

    print(f"Collecting {num_tasks} tasks and exporting to {output_csv!r} ...")
    env.run()

    df = pd.DataFrame.from_records(records)
    df.to_csv(output_csv, index=False)
    print(f"Done. Wrote {len(df)} rows to {output_csv!r}.")
    return df


def train_latency_mlp(
    df: Optional[pd.DataFrame] = None,
    hidden_layer_sizes: Tuple[int, ...] = (64, 32),
    random_state: int = 42,
) -> Pipeline:
    """
    Train an MLPRegressor model to predict total_experienced_latency_s
    from the observed network state.

    Features:
        - current_edge_queue
        - satellite_latency_ms
        - task_size_mb
        - chosen_node (one-hot encoded: is_edge)

    Target:
        - total_experienced_latency_s

    Returns a scikit-learn Pipeline that includes feature scaling and
    the trained MLPRegressor.
    """
    if df is None:
        df = generate_training_data()

    # Prepare features
    X = df[["current_edge_queue", "satellite_latency_ms", "task_size_mb"]].copy()
    # Encode chosen_node as binary: 1 if edge, 0 if leo
    X["is_edge"] = (df["chosen_node"] == "edge").astype(float)

    y = df["total_experienced_latency_s"].values

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    learning_rate_init=0.001,
                    max_iter=300,
                    random_state=random_state,
                ),
            ),
        ]
    )

    print("Training MLPRegressor on latency data...")
    model.fit(X, y)
    print("Training complete.")

    return model


if __name__ == "__main__":
    # Default behavior: generate the CSV dataset needed for Phase 3+.
    collect_and_export_training_data_csv()

