import sys, json, copy
import os; ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from config import MODEL
from simulation.topology import build_fat_tree
from simulation.traffic_generator import TrafficGenerator
from simulation.network_engine import NetworkEngine
from simulation.telemetry import TelemetryCollector, run_simulation_and_collect
from routing.guarded import GuardedRerouter
PATTERNS=['uniform','hotspot','elephant_mice','stride','llm_training','mixed']
rec=json.load(open(os.path.join(ROOT,'research','benchmark_results.json')))
R={(r['pattern'],r['seed'],r['strategy']):r for r in rec['runs']}
G=build_fat_tree()
def run(flows, strat, ticks=150):
    e=NetworkEngine(G); flows=copy.deepcopy(flows)
    for f in flows: f.current_path=[]; f.active=False
    e.set_flows(flows); c=TelemetryCollector(G,e)
    ctl=GuardedRerouter(G,c.edge_to_idx)
    for _ in range(ticks):
        c.collect_tick(e.step())
        if strat!='ecmp' and e.tick>=MODEL['window_size']:
            if strat=='guarded_persistence': p=c.edge_labels_history[-1]
            else: p=np.clip(c.edge_features_history[-1][:,0],0,1)   # current utilization as score
            ctl.evaluate_and_reroute(flows,p,e,e.tick)
    m=e.metrics(); m['reroutes']=len(ctl.reroute_log); return m
tot={k:0 for k in ['ecmp_rec','ecmp_mine','gg','gp','gu']}; rows=[]
for pat in PATTERNS:
  for s in range(3000,3003):
    fl=TrafficGenerator(G,seed=s).generate_scenario(total_ticks=150,pattern=pat)
    a=run(fl,'ecmp'); b=run(fl,'guarded_persistence'); u=run(fl,'guarded_util')
    er=R[(pat,s,'ecmp')]['total_drops']; gg=R[(pat,s,'guarded_gnn')]['total_drops']
    rows.append((pat,s,er,a['total_drops'],gg,b['total_drops'],u['total_drops'],R[(pat,s,'guarded_gnn')]['reroutes'],b['reroutes']))
    for k,v in zip(tot,[er,a['total_drops'],gg,b['total_drops'],u['total_drops']]): tot[k]+=v
print("pattern seed | ECMP(rec) ECMP(mine) | GuardedGNN(rec) GuardedPersist GuardedUtil | reroutes GG/GP")
for r in rows: print(*r)
print({k:v/18 for k,v in tot.items()})
for k in ['gg','gp','gu']: print(k,"reduction vs ECMP %.2f%%"%((1-tot[k]/tot['ecmp_rec'])*100))
# forecasting: continuous persistence on recorded test split
test=rec['splits']['test']; y=[];pb=[];pc=[];onset=[]
for i in test:
    pat=PATTERNS[i%6]; fl=TrafficGenerator(G,seed=42+i).generate_scenario(total_ticks=150,pattern=pat)
    c=run_simulation_and_collect(G,fl,150)
    for smp in c.get_windowed_samples():
        u=smp['edge_features'][-1][:,0]; y.append(smp['labels']); pb.append((u>0.85).astype(float)); pc.append(u)
y=np.concatenate(y); pb=np.concatenate(pb); pc=np.concatenate(pc)
print("test pos rate %.4f  binary persistence F1 %.4f AP %.4f AUC %.4f"%(y.mean(),f1_score(y,pb),average_precision_score(y,pb),roc_auc_score(y,pb)))
print("continuous persistence (current util as score) AP %.4f AUC %.4f"%(average_precision_score(y,pc),roc_auc_score(y,pc)))
on=pb==0; print("onset cases:",on.sum(),"became congested:",int(y[on].sum()), "share of positives that are onsets: %.3f"%(y[on].sum()/y.sum()))