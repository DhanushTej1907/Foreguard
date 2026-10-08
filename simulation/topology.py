"""
Fat-Tree Topology Builder
=========================
Constructs a k-ary Fat-Tree data center topology using NetworkX.
A k=4 Fat-Tree yields:
  - 4 Core switches
  - 8 Aggregation switches (2 pods × k/2 each)
  - 8 Edge (ToR) switches
  - 16 Hosts (servers)
  - 48 links total

Each node and edge carries metadata used by the simulator and GNN.
"""

import networkx as nx
from config import TOPOLOGY


def build_fat_tree(k: int = None) -> nx.Graph:
    """
    Build a k-ary Fat-Tree topology.

    Parameters
    ----------
    k : int
        Fat-Tree parameter. Must be even. Default from config.

    Returns
    -------
    nx.Graph
        The Fat-Tree graph with node/edge attributes.
    """
    if k is None:
        k = TOPOLOGY["k"]
    if not isinstance(k, int) or k < 2 or k % 2:
        raise ValueError(f"k must be a positive even integer >= 2, got {k}")

    G = nx.Graph()

    num_pods = k
    num_core = (k // 2) ** 2
    num_agg_per_pod = k // 2
    num_edge_per_pod = k // 2
    num_hosts_per_edge = k // 2

    # ── Create Core Switches ──────────────────────────────────────
    core_switches = []
    for i in range(num_core):
        node_id = f"core_{i}"
        G.add_node(node_id, type="core", layer=0, pod=-1, index=i)
        core_switches.append(node_id)

    # ── Create Pods ───────────────────────────────────────────────
    agg_switches = []
    edge_switches = []
    hosts = []

    for pod in range(num_pods):
        # Aggregation switches in this pod
        pod_agg = []
        for a in range(num_agg_per_pod):
            node_id = f"agg_{pod}_{a}"
            G.add_node(node_id, type="aggregation", layer=1, pod=pod, index=a)
            pod_agg.append(node_id)
        agg_switches.extend(pod_agg)

        # Edge switches in this pod
        pod_edge = []
        for e in range(num_edge_per_pod):
            node_id = f"edge_{pod}_{e}"
            G.add_node(node_id, type="edge", layer=2, pod=pod, index=e)
            pod_edge.append(node_id)
        edge_switches.extend(pod_edge)

        # ── Intra-pod links: Edge ↔ Aggregation ──────────────────
        for edge_sw in pod_edge:
            for agg_sw in pod_agg:
                _add_link(G, edge_sw, agg_sw)

        # ── Core ↔ Aggregation links ─────────────────────────────
        for a_idx, agg_sw in enumerate(pod_agg):
            # Each aggregation switch connects to k/2 core switches
            for c in range(k // 2):
                core_idx = a_idx * (k // 2) + c
                _add_link(G, agg_sw, core_switches[core_idx])

        # ── Hosts ↔ Edge switches ─────────────────────────────────
        for e_idx, edge_sw in enumerate(pod_edge):
            for h in range(num_hosts_per_edge):
                host_id = f"host_{pod}_{e_idx}_{h}"
                G.add_node(host_id, type="host", layer=3, pod=pod, edge_index=e_idx)
                _add_link(G, host_id, edge_sw, is_host_link=True)
                hosts.append(host_id)

    # Store topology metadata on the graph itself
    G.graph["k"] = k
    G.graph["core_switches"] = core_switches
    G.graph["agg_switches"] = agg_switches
    G.graph["edge_switches"] = edge_switches
    G.graph["hosts"] = hosts
    G.graph["all_switches"] = core_switches + agg_switches + edge_switches

    return G


def _add_link(G: nx.Graph, u: str, v: str, is_host_link: bool = False):
    """Add a bidirectional link with capacity and buffer attributes."""
    capacity = TOPOLOGY["link_capacity_packets_per_tick"]
    if is_host_link:
        capacity = TOPOLOGY["link_capacity_packets_per_tick"]  # Same capacity for host links

    G.add_edge(u, v,
               capacity=capacity,
               propagation_delay_us=TOPOLOGY["propagation_delay_us"],
               buffer_size=TOPOLOGY["buffer_size_packets"],
               # Dynamic state (updated during simulation)
               utilization=0.0,
               queue_depth=0,
               packet_drops=0,
               latency_us=TOPOLOGY["propagation_delay_us"],
               is_host_link=is_host_link)


def get_switch_graph(G: nx.Graph) -> nx.Graph:
    """
    Extract the switch-only subgraph (no hosts).
    This is the graph the GNN operates on.
    """
    switch_nodes = G.graph["all_switches"]
    return G.subgraph(switch_nodes).copy()


def get_switch_index_map(G: nx.Graph) -> dict:
    """
    Create a mapping from switch node IDs to integer indices for the GNN.
    """
    switches = G.graph["all_switches"]
    return {sw: idx for idx, sw in enumerate(switches)}


def get_edge_index_map(G: nx.Graph) -> dict:
    """
    Create a mapping from edge tuples to integer indices for the GNN.
    Only includes switch-to-switch edges (no host links).
    """
    switch_set = set(G.graph["all_switches"])
    edge_map = {}
    idx = 0
    for u, v, data in G.edges(data=True):
        if u in switch_set and v in switch_set:
            edge_map[(u, v)] = idx
            edge_map[(v, u)] = idx  # Undirected
            idx += 1
    return edge_map


def print_topology_summary(G: nx.Graph):
    """Print a summary of the Fat-Tree topology."""
    k = G.graph["k"]
    print(f"\n{'='*50}")
    print(f"  Fat-Tree Topology (k={k})")
    print(f"{'='*50}")
    print(f"  Core switches:        {len(G.graph['core_switches'])}")
    print(f"  Aggregation switches: {len(G.graph['agg_switches'])}")
    print(f"  Edge switches:        {len(G.graph['edge_switches'])}")
    print(f"  Hosts:                {len(G.graph['hosts'])}")
    print(f"  Total nodes:          {G.number_of_nodes()}")
    print(f"  Total links:          {G.number_of_edges()}")
    print(f"{'='*50}\n")
