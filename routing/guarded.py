"""Experimental capacity-reserving controller; no RDMA ordering guarantee.

Demand, queue and headroom are checked per direction (full-duplex link states);
risk uses the per-physical-link prediction, mapped for both directions in edge_to_idx.
"""
from collections import Counter
from itertools import islice
import numpy as np
import networkx as nx
from config import REROUTING

class GuardedRerouter:
    def __init__(self, graph, edge_to_idx, threshold=None, max_changes=4,
                 headroom=.90, min_improvement=.15, max_age=1):
        self.G, self.edge_to_idx = graph, edge_to_idx
        self.threshold = threshold if threshold is not None else REROUTING['congestion_alert_threshold']
        self.max_changes, self.headroom = max_changes, headroom
        self.min_improvement, self.max_age = min_improvement, max_age
        self.last_reroute_tick, self.reroute_log = {}, []
        self.rejections = Counter()

    def evaluate_and_reroute(self, flows, predictions, engine, tick, prediction_tick=None):
        if prediction_tick is None: prediction_tick = tick
        expected = max(self.edge_to_idx.values(), default=-1) + 1
        p = np.asarray(predictions)
        if p.shape != (expected,) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            self.rejections['invalid_prediction'] += 1
            return []
        if not 0 <= tick - prediction_tick <= self.max_age:
            self.rejections['stale_prediction'] += 1
            return []
        edges = lambda path: [engine._edge_key(u,v) for u,v in zip(path,path[1:])]
        def risk(path):
            return max((float(p[self.edge_to_idx[e]]) for e in edges(path) if e in self.edge_to_idx), default=0.)
        # Include current offered flow rates and observed arrivals; reserve moves
        # sequentially so simultaneous decisions cannot all use the same headroom.
        demand = Counter()
        active = [f for f in flows if f.active]
        for f in active:
            for e in edges(f.current_path): demand[e] += f.rate
        for e, s in engine.link_states.items(): demand[e] = max(demand[e], s.packets_in)
        events = []
        for f in sorted(active, key=lambda f: (-risk(f.current_path), f.flow_id)):
            if len(events) >= self.max_changes: break
            old = f.current_path
            if risk(old) <= self.threshold: continue
            if tick - self.last_reroute_tick.get(f.flow_id, -10**9) < REROUTING['reroute_cooldown_ticks']:
                self.rejections['cooldown'] += 1; continue
            old_edges = set(edges(old))
            def weight(u,v,d):
                e = engine._edge_key(u,v)
                return 1 + 10 * float(p[self.edge_to_idx[e]]) if e in self.edge_to_idx else 1
            candidates = nx.shortest_simple_paths(self.G, f.src, f.dst, weight=weight)
            chosen = None
            try:
                for path in islice(candidates, REROUTING['k_shortest_paths']):
                    if path == old or len(path) > len(old) + 2: continue
                    if risk(old) - risk(path) < self.min_improvement: continue
                    added = set(edges(path)) - old_edges
                    if all(demand[e] + f.rate + engine.link_states[e].queue <=
                           self.headroom * engine.link_states[e].capacity for e in added):
                        chosen = path; break
            except nx.NetworkXNoPath:
                pass
            if chosen is None:
                self.rejections['no_admissible_path'] += 1; continue
            for e in old_edges: demand[e] -= f.rate
            for e in edges(chosen): demand[e] += f.rate
            f.current_path = chosen
            self.last_reroute_tick[f.flow_id] = tick
            event = dict(tick=tick, flow_id=f.flow_id, src=f.src, dst=f.dst,
                         old_path=old, new_path=chosen, reason='guarded',
                         old_risk=risk(old), new_risk=risk(chosen))
            events.append(event); self.reroute_log.append(event)
        return events

    def get_stats(self):
        return dict(total_reroutes=len(self.reroute_log), rejections=dict(self.rejections))
