"""
Traffic Generator
=================
Generates realistic data center traffic patterns for the Fat-Tree simulator.

Supported patterns:
  1. Uniform Random  — baseline low-utilization traffic
  2. Hotspot/Incast  — many-to-one bursts (e.g., Amazon sale scenario)
  3. Elephant + Mice  — mix of large persistent and small bursty flows
  4. Stride           — pod-to-pod cross-section traffic
"""

import random
import numpy as np
import networkx as nx
from config import TRAFFIC, SIMULATION, TOPOLOGY


class Flow:
    """Represents a single network flow."""

    __slots__ = ["flow_id", "src", "dst", "rate", "start_tick", "duration",
                 "remaining_duration", "current_path", "flow_type", "active"]

    def __init__(self, flow_id: int, src: str, dst: str, rate: float,
                 start_tick: int, duration: int, flow_type: str = "mice"):
        self.flow_id = flow_id
        self.src = src
        self.dst = dst
        self.rate = rate                  # Packets per tick
        self.start_tick = start_tick
        self.duration = duration          # In ticks
        self.remaining_duration = duration
        self.current_path = []            # Set by routing
        self.flow_type = flow_type        # "mice", "elephant", "incast"
        self.active = False

    def __repr__(self):
        return (f"Flow({self.flow_id}: {self.src}→{self.dst}, "
                f"rate={self.rate:.1f}, type={self.flow_type})")


class TrafficGenerator:
    """
    Generates traffic scenarios for the data center simulation.
    """

    def __init__(self, graph: nx.Graph, seed: int = 42):
        self.G = graph
        self.hosts = graph.graph["hosts"]
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.flow_counter = 0
        self.capacity = TOPOLOGY["link_capacity_packets_per_tick"]

    def generate_scenario(self, total_ticks: int = None,
                          pattern: str = "mixed") -> list:
        """
        Generate a complete traffic scenario.

        Parameters
        ----------
        total_ticks : int
            Duration of the scenario in ticks.
        pattern : str
            One of "uniform", "hotspot", "elephant_mice", "stride", "mixed".

        Returns
        -------
        list[Flow]
            List of all flows in the scenario.
        """
        if total_ticks is None:
            total_ticks = SIMULATION["total_ticks"]

        if pattern == "uniform":
            return self._gen_uniform(total_ticks)
        elif pattern == "hotspot":
            return self._gen_hotspot(total_ticks)
        elif pattern == "elephant_mice":
            return self._gen_elephant_mice(total_ticks)
        elif pattern == "stride":
            return self._gen_stride(total_ticks)
        elif pattern == "llm_training":
            return self._gen_llm_training(total_ticks)
        elif pattern == "mixed":
            return self._gen_mixed(total_ticks)
        else:
            raise ValueError(f"Unknown traffic pattern: {pattern}")

    def _new_flow(self, src, dst, rate, start_tick, duration, flow_type="mice"):
        """Create a new flow with unique ID."""
        self.flow_counter += 1
        return Flow(self.flow_counter, src, dst, rate, start_tick, duration, flow_type)

    def _random_host_pair(self):
        """Pick two distinct random hosts."""
        src = self.rng.choice(self.hosts)
        dst = self.rng.choice(self.hosts)
        while dst == src:
            dst = self.rng.choice(self.hosts)
        return src, dst

    # ── Pattern: Uniform Random ──────────────────────────────────
    def _gen_uniform(self, total_ticks: int) -> list:
        """Low-rate background traffic between random host pairs."""
        flows = []
        num_flows = len(self.hosts) * 2  # 2 flows per host on average

        for _ in range(num_flows):
            src, dst = self._random_host_pair()
            rate = self.capacity * TRAFFIC["baseline_utilization"] * self.np_rng.uniform(0.3, 1.0)
            start = self.rng.randint(0, max(1, total_ticks // 4))
            duration = self.rng.randint(total_ticks // 2, total_ticks)
            flows.append(self._new_flow(src, dst, rate, start, duration, "mice"))

        return flows

    # ── Pattern: Hotspot / Incast ────────────────────────────────
    def _gen_hotspot(self, total_ticks: int) -> list:
        """
        Multiple sources bombard a single destination.
        Simulates the "Amazon Great Indian Festival" scenario.
        """
        flows = []

        # Background traffic
        flows.extend(self._gen_uniform(total_ticks))

        # Create multiple hotspot events at different times
        num_hotspots = max(1, total_ticks // 50)
        for _ in range(num_hotspots):
            target = self.rng.choice(self.hosts)
            num_sources = TRAFFIC["incast_degree"]
            sources = self.rng.sample(
                [h for h in self.hosts if h != target],
                min(num_sources, len(self.hosts) - 1)
            )
            start = self.rng.randint(total_ticks // 4, 3 * total_ticks // 4)
            duration = self.rng.randint(20, 60)

            for src in sources:
                rate = self.capacity * TRAFFIC["hotspot_intensity"] * self.np_rng.uniform(0.7, 1.0)
                flows.append(self._new_flow(src, target, rate, start, duration, "incast"))

        return flows

    # ── Pattern: Elephant + Mice ─────────────────────────────────
    def _gen_elephant_mice(self, total_ticks: int) -> list:
        """Mix of large persistent flows (elephant) and short bursty ones (mice)."""
        flows = []
        total_flows = len(self.hosts) * 3

        for _ in range(total_flows):
            src, dst = self._random_host_pair()

            if self.np_rng.random() < TRAFFIC["elephant_fraction"]:
                # Elephant flow: high rate, long duration
                rate = self.capacity * TRAFFIC["elephant_flow_rate"] * self.np_rng.uniform(0.5, 1.0)
                start = self.rng.randint(0, total_ticks // 4)
                duration = self.rng.randint(total_ticks // 2, total_ticks)
                flows.append(self._new_flow(src, dst, rate, start, duration, "elephant"))
            else:
                # Mice flow: low rate, short duration
                rate = self.capacity * TRAFFIC["mice_flow_rate"] * self.np_rng.uniform(0.5, 2.0)
                start = self.rng.randint(0, total_ticks - 10)
                duration = self.rng.randint(5, 30)
                flows.append(self._new_flow(src, dst, rate, start, duration, "mice"))

        return flows

    # ── Pattern: Stride ──────────────────────────────────────────
    def _gen_stride(self, total_ticks: int) -> list:
        """
        Pod-to-pod traffic pattern — stresses the core switches.
        Host i in pod p sends to host i in pod (p + stride) % num_pods.
        """
        flows = []
        k = self.G.graph["k"]
        stride = 1

        hosts_by_pod = {}
        for h in self.hosts:
            parts = h.split("_")
            pod = int(parts[1])
            hosts_by_pod.setdefault(pod, []).append(h)

        for pod in range(k):
            target_pod = (pod + stride) % k
            src_hosts = hosts_by_pod.get(pod, [])
            dst_hosts = hosts_by_pod.get(target_pod, [])

            for i, src in enumerate(src_hosts):
                if i < len(dst_hosts):
                    dst = dst_hosts[i]
                    rate = self.capacity * 0.5 * self.np_rng.uniform(0.5, 1.0)
                    start = self.rng.randint(0, total_ticks // 4)
                    duration = self.rng.randint(total_ticks // 2, total_ticks)
                    flows.append(self._new_flow(src, dst, rate, start, duration, "elephant"))

        return flows

    # ── Pattern: LLM Distributed Training (RoCEv2 Microbursts) ───
    def _gen_llm_training(self, total_ticks: int) -> list:
        """
        Generates traffic mimicking Distributed LLM Training clusters (e.g., GPU clusters).
        Specifically models Ring-All-Reduce and All-to-All communication phases which cause
        bursty traffic; this simulator does not implement PFC or RoCEv2.
        Synthetic collective-inspired traffic; no claim of novelty or trace fidelity.
        """
        flows = []
        # Group hosts into GPU pods (e.g., pods of 4)
        gpu_pods = [self.hosts[i:i+4] for i in range(0, len(self.hosts), 4)]
        
        for tick in range(0, total_ticks, 50): # Synchronous burst every 50 ticks (compute phase ends)
            if self.np_rng.random() > 0.8: continue # 80% chance of synchronization phase
            
            # 1. Intra-pod Ring-All-Reduce (High bandwidth, short burst)
            for pod in gpu_pods:
                for i in range(len(pod)):
                    src = pod[i]
                    dst = pod[(i + 1) % len(pod)]
                    rate = self.capacity * 0.95 * self.np_rng.uniform(0.9, 1.1)
                    flows.append(self._new_flow(src, dst, rate, tick, 5, "llm_ring"))
            
            # 2. Inter-pod All-to-All (The massive Microburst creator)
            if self.np_rng.random() > 0.5: # 50% chance of inter-pod parameter sync
                sync_start = tick + 5
                for i, pod_a in enumerate(gpu_pods):
                    for j, pod_b in enumerate(gpu_pods):
                        if i == j: continue
                        src = pod_a[0]
                        dst = pod_b[0]
                        rate = self.capacity * TRAFFIC["hotspot_intensity"] * 1.5
                        flows.append(self._new_flow(src, dst, rate, sync_start, 3, "llm_all_to_all"))
                        
        return flows

    # ── Pattern: Mixed ───────────────────────────────────────────
    def _gen_mixed(self, total_ticks: int) -> list:
        """
        Combines all patterns for realistic, diverse traffic.
        Best for training the GNN — captures varied congestion scenarios.
        """
        flows = []

        # Phase 1: Warm-up with uniform traffic
        flows.extend(self._gen_uniform(total_ticks))

        # Phase 2: Elephant flows appear
        flows.extend(self._gen_elephant_mice(total_ticks))

        # Phase 3: Hotspot/Incast bursts
        flows.extend(self._gen_hotspot(total_ticks))
        
        # Phase 4: Synthetic collective-inspired bursts
        flows.extend(self._gen_llm_training(total_ticks))

        return flows



def generate_training_scenarios(graph: nx.Graph, num_scenarios: int = None,
                                seed: int = 42) -> list:
    """
    Generate multiple diverse traffic scenarios for GNN training.

    Returns
    -------
    list[list[Flow]]
        List of scenarios, each containing a list of flows.
    """
    if num_scenarios is None:
        num_scenarios = SIMULATION["num_scenarios"]

    patterns = ["llm_training", "mixed", "hotspot", "elephant_mice", "stride", "uniform",
                "llm_training", "mixed", "hotspot", "elephant_mice"]

    scenarios = []
    for i in range(num_scenarios):
        pattern = patterns[i % len(patterns)]
        gen = TrafficGenerator(graph, seed=seed + i)
        scenario = gen.generate_scenario(pattern=pattern)
        scenarios.append(scenario)
        print(f"  Scenario {i+1}/{num_scenarios}: pattern={pattern}, flows={len(scenario)}")

    return scenarios
