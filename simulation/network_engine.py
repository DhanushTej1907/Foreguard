"""Causal, packet-batch, discrete-time teaching simulator.

One FIFO shared by both directions per physical link (not full-duplex RoCE).
Packets traverse at most one link per tick. No PFC, ECN, retransmission or NIC.
"""
from collections import deque, Counter
import hashlib
import networkx as nx
import numpy as np
from config import TOPOLOGY, SIMULATION


class LinkState:
    def __init__(self, capacity, buffer_size, prop_delay):
        if capacity <= 0 or buffer_size <= 0:
            raise ValueError("capacity and buffer size must be positive")
        self.capacity = int(capacity)
        self.buffer_size = int(buffer_size)
        self.prop_delay = prop_delay
        self.queue = self.queue_depth = 0
        self.utilization = 0.0
        self.latency_us = prop_delay
        self.batches = deque()
        self.prepare_tick()

    def prepare_tick(self):
        self.packets_in = self.packets_out = self.packets_dropped = 0

    def enqueue(self, num_packets, payload=None):
        if num_packets < 0:
            raise ValueError("negative packet count")
        accepted = min(num_packets, self.buffer_size - self.queue)
        dropped = num_packets - accepted
        self.queue += accepted
        self.packets_in += num_packets
        self.packets_dropped += dropped
        if accepted:
            self.batches.append([accepted, payload])
        return dropped

    def process_tick(self):
        budget = self.capacity
        departures = []
        while budget and self.batches:
            count, payload = self.batches[0]
            served = min(count, budget)
            departures.append((served, payload))
            budget -= served
            self.queue -= served
            self.packets_out += served
            if served == count:
                self.batches.popleft()
            else:
                self.batches[0][0] -= served
        # Offered load can exceed 1; served utilization is separately exported.
        self.utilization = self.packets_in / self.capacity
        self.queue_depth = self.queue
        self.latency_us = self.prop_delay + self.queue / self.capacity * SIMULATION['tick_duration_ms'] * 1000
        return departures


class NetworkEngine:
    def __init__(self, graph):
        self.G = graph
        self.all_switches = set(graph.graph['all_switches'])
        self.hosts = graph.graph['hosts']
        self.link_states = {self._edge_key(u, v): LinkState(d['capacity'], d['buffer_size'], d['propagation_delay_us'])
                            for u, v, d in graph.edges(data=True)}
        self.active_flows = []
        self._path_cache = {}
        self.reset()

    @staticmethod
    def _edge_key(u, v):
        return tuple(sorted((u, v)))

    def get_shortest_path(self, src, dst):
        try:
            return nx.shortest_path(self.G, src, dst)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_ecmp_path(self, flow):
        key = (flow.src, flow.dst)
        if key not in self._path_cache:
            try:
                self._path_cache[key] = sorted(nx.all_shortest_paths(self.G, *key))
            except nx.NetworkXNoPath:
                self._path_cache[key] = []
        paths = self._path_cache[key]
        digest = hashlib.sha256(f'{flow.src}|{flow.dst}|{flow.flow_id}'.encode()).digest()
        return list(paths[int.from_bytes(digest[:8], 'big') % len(paths)]) if paths else []

    def get_weighted_shortest_path(self, src, dst, weight_func=None):
        if weight_func is None:
            return self.get_shortest_path(src, dst)
        try:
            return nx.shortest_path(self.G, src, dst,
                weight=lambda u, v, d: weight_func(self.link_states[self._edge_key(u, v)]))
        except nx.NetworkXNoPath:
            return []

    def clear_path_cache(self):
        self._path_cache.clear()

    def set_flows(self, flows):
        if len({f.flow_id for f in flows}) != len(flows):
            raise ValueError('flow IDs must be unique')
        self.active_flows = flows
        for f in flows:
            if not f.current_path:
                f.current_path = self.get_ecmp_path(f)

    def step(self):
        # Tick zero is a real traffic tick. Snapshot tick counts completed ticks.
        now = self.tick
        for state in self.link_states.values():
            state.prepare_tick()
        arrivals, self.pending = self.pending, []
        for f in self.active_flows:
            f.active = f.start_tick <= now < f.start_tick + f.duration
            if not f.active:
                continue
            self._fractional[f.flow_id] += f.rate
            count = int(self._fractional[f.flow_id])
            self._fractional[f.flow_id] -= count
            self.total_packets += count
            path = tuple(f.current_path)
            if len(path) < 2:
                self.total_drops += count
            elif count:
                arrivals.append((count, (path, 0, now, f.flow_id)))
        for count, payload in arrivals:
            path, hop, born, flow_id = payload
            state = self.link_states[self._edge_key(path[hop], path[hop + 1])]
            self.total_drops += state.enqueue(count, payload)
        for state in self.link_states.values():
            for count, payload in state.process_tick():
                path, hop, born, flow_id = payload
                if hop + 1 == len(path) - 1:
                    self.total_delivered += count
                    self.delivery_latency_ticks[now - born + 1] += count
                else:
                    self.pending.append((count, (path, hop + 1, born, flow_id)))
        self.tick += 1
        self.link_latency_history.extend(s.latency_us for s in self.link_states.values())
        self.link_util_history.extend(s.packets_out / s.capacity for s in self.link_states.values())
        return self._collect_snapshot()

    @property
    def outstanding_packets(self):
        return sum(s.queue for s in self.link_states.values()) + sum(c for c, _ in self.pending)

    def _collect_snapshot(self):
        links = {key: {name: getattr(s, name) for name in
                 ['utilization', 'queue_depth', 'packets_dropped', 'latency_us', 'packets_in', 'packets_out']}
                 for key, s in self.link_states.items()}
        switches = {}
        for sw in self.all_switches:
            states = [self.get_link_state(sw, n) for n in self.G.neighbors(sw)]
            total_queue = sum(s.queue for s in states)
            switches[sw] = dict(avg_queue=total_queue / len(states),
                total_throughput=sum(s.packets_out for s in states),
                total_drops=sum(s.packets_dropped for s in states),
                max_utilization=max(s.utilization for s in states),
                buffer_usage=total_queue / sum(s.buffer_size for s in states))
        return dict(tick=self.tick, links=links, switches=switches,
                    active_flows=sum(f.active for f in self.active_flows),
                    total_packets=self.total_packets, total_drops=self.total_drops,
                    total_delivered=self.total_delivered, outstanding_packets=self.outstanding_packets)

    def get_link_state(self, u, v):
        return self.link_states.get(self._edge_key(u, v))

    def get_all_link_utilizations(self):
        return {k: s.utilization for k, s in self.link_states.items()
                if all(n in self.all_switches for n in k)}

    def reset(self):
        self.tick = self.total_packets = self.total_drops = self.total_delivered = 0
        self.pending = []
        self._fractional = Counter()
        self.delivery_latency_ticks = Counter()
        self.link_latency_history = []
        self.link_util_history = []
        self._path_cache.clear()
        for s in self.link_states.values():
            s.queue = s.queue_depth = 0
            s.batches.clear()
            s.utilization = 0.0
            s.latency_us = s.prop_delay
            s.prepare_tick()
        for f in self.active_flows:
            f.active = False

    def metrics(self):
        count = self.total_delivered
        avg_ticks = sum(t * n for t, n in self.delivery_latency_ticks.items()) / max(1, count)
        target, cumulative, p99 = int(np.ceil(count * .99)), 0, None
        for latency, n in sorted(self.delivery_latency_ticks.items()):
            cumulative += n
            if cumulative >= target:
                p99 = latency * SIMULATION['tick_duration_ms'] * 1000
                break
        return dict(avg_latency=float(np.mean(self.link_latency_history)) if self.link_latency_history else 0.,
            max_latency=max(self.link_latency_history, default=0.),
            avg_utilization=float(np.mean(self.link_util_history)) if self.link_util_history else 0.,
            total_drops=self.total_drops, total_offered=self.total_packets,
            total_delivered=count, outstanding_packets=self.outstanding_packets,
            avg_throughput=count / max(1, self.tick),
            delivered_mean_latency_us=avg_ticks * SIMULATION['tick_duration_ms'] * 1000 if count else None,
            delivered_p99_latency_us=p99,
            packet_conservation=self.total_packets == self.total_drops + count + self.outstanding_packets)
