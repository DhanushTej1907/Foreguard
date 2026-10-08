"""Reproducible research run: grouped training + paired held-out traffic runs."""
import argparse, copy, json, random, time, platform
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, brier_score_loss
from config import MODEL, SIMULATION, TRAINING
from simulation.topology import build_fat_tree
from simulation.traffic_generator import TrafficGenerator
from simulation.network_engine import NetworkEngine
from simulation.telemetry import TelemetryCollector, run_simulation_and_collect
from model.dataset import build_datasets, save_datasets
from model.gnn_model import CongestionPredictor
from model.train import train_model
from routing.rerouter import ProactiveRerouter, ReactiveRerouter
from routing.guarded import GuardedRerouter

PATTERNS = ['uniform', 'hotspot', 'elephant_mice', 'stride', 'llm_training', 'mixed']

def predict(model, collector):
    w=MODEL['window_size']
    with torch.no_grad():
        return model(torch.tensor(np.stack(collector.node_features_history[-w:])).unsqueeze(0),
                     torch.tensor(np.stack(collector.edge_features_history[-w:])).unsqueeze(0),
                     torch.tensor(collector.get_pyg_edge_index())).squeeze(0).numpy()

def scores(labels, probs):
    y,p=np.asarray(labels).ravel(),np.asarray(probs).ravel()
    return dict(f1=float(f1_score(y,p>.5,zero_division=0)),
                average_precision=float(average_precision_score(y,p)) if y.sum() else None,
                roc_auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
                brier=float(brier_score_loss(y,p)), positive_fraction=float(y.mean()))

def run_case(graph, flows, ticks, strategy, model):
    engine=NetworkEngine(graph); flows=copy.deepcopy(flows)
    for f in flows: f.current_path=[]; f.active=False
    engine.set_flows(flows); c=TelemetryCollector(graph,engine)
    controller = (ReactiveRerouter(graph) if strategy=='reactive' else
                  GuardedRerouter(graph,c.edge_to_idx) if strategy=='guarded_gnn' else
                  ProactiveRerouter(graph,c.edge_to_idx))
    inference=[]
    for _ in range(ticks):
        c.collect_tick(engine.step())
        if strategy=='reactive': controller.evaluate_and_reroute(flows,engine,engine.tick)
        elif strategy!='ecmp' and engine.tick>=MODEL['window_size']:
            if strategy=='persistence':
                p=c.edge_labels_history[-1]
            else:
                start=time.perf_counter(); p=predict(model,c); inference.append((time.perf_counter()-start)*1000)
            controller.evaluate_and_reroute(flows,p,engine,engine.tick)
    result=engine.metrics()
    result['reroutes']=len(controller.reroute_log)
    result['inference_ms_p50']=float(np.median(inference)) if inference else None
    result['inference_ms_p99']=float(np.percentile(inference,99)) if inference else None
    if strategy=='guarded_gnn': result['rejections']=dict(controller.rejections)
    return result

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--ticks',type=int,default=150); parser.add_argument('--seeds',type=int,default=3)
    args=parser.parse_args()
    if args.ticks<50 or args.seeds<1 or args.epochs<1: parser.error('ticks >= 50, seeds >= 1, epochs >= 1')
    torch.set_num_threads(2); torch.manual_seed(42); np.random.seed(42); random.seed(42)
    TRAINING['batch_size']=32
    graph=build_fat_tree(); samples=[]; metadata=[]
    # Independent random seeds with each family represented repeatedly.
    for i in range(30):
        pattern=PATTERNS[i%len(PATTERNS)]
        flows=TrafficGenerator(graph,seed=42+i).generate_scenario(total_ticks=args.ticks,pattern=pattern)
        c=run_simulation_and_collect(graph,flows,args.ticks)
        windows=c.get_windowed_samples()
        for s in windows: s['scenario_id']=i
        samples.extend(windows); metadata.append(dict(scenario_id=i,seed=42+i,pattern=pattern))
    train,val,test=build_datasets(samples,c.get_pyg_edge_index()); save_datasets(train,val,test)
    model=CongestionPredictor(num_edges=len(c.switch_edges),edge_pairs=[(c.switch_to_idx[u],c.switch_to_idx[v]) for u,v in c.switch_edges])
    history=train_model(model,train,val,epochs=args.epochs); model.eval()
    labels=[]; probabilities=[]; persistence=[]
    # Batched inference for the held-out windows.
    from torch.utils.data import DataLoader
    from model.dataset import collate_fn
    with torch.no_grad():
        for b in DataLoader(test,batch_size=32,collate_fn=collate_fn):
            probabilities.extend(model(b['node_features'],b['edge_features'],b['edge_index']).numpy())
            labels.extend(b['labels'].numpy())
            persistence.extend((b['edge_features'][:,-1,:,0].numpy()>MODEL['congestion_threshold']).astype(float))
    output=dict(schema_version=3, settings=vars(args), python=platform.python_version(), torch=torch.__version__,
                scope='synthetic shared-link batch simulator; no PFC/RDMA; no real-time deployment claim',
                reproducibility={'global_seed':42,'model_parameters':17985,'model_training_seed':42,'routing_seed_start':3000},
                scenarios=metadata, splits={name: sorted({s['scenario_id'] for s in ds.samples}) for name,ds in [('train',train),('val',val),('test',test)]},
                history=history, prediction={'gnn':scores(labels,probabilities),'persistence':scores(labels,persistence)},runs=[])
    for pattern in PATTERNS:
        for seed in range(3000,3000+args.seeds):
            flows=TrafficGenerator(graph,seed=seed).generate_scenario(total_ticks=args.ticks,pattern=pattern)
            for strategy in ['ecmp','reactive','persistence','gnn','guarded_gnn']:
                result=run_case(graph,flows,args.ticks,strategy,model)
                assert result['packet_conservation']
                output['runs'].append(dict(pattern=pattern,seed=seed,strategy=strategy,**result))
            print(f'Completed {pattern}, seed {seed}',flush=True)
    # Paired seed bootstrap inside each family. Three seeds are pilot evidence only.
    rng=np.random.default_rng(42); summary=[]
    for pattern in PATTERNS:
        runs=[r for r in output['runs'] if r['pattern']==pattern]
        for strategy in ['reactive','persistence','gnn','guarded_gnn']:
            base=[r for r in runs if r['strategy']=='ecmp']; alt=[r for r in runs if r['strategy']==strategy]
            diffs=np.array([a['total_drops']-b['total_drops'] for a,b in zip(alt,base)])
            means=np.mean(rng.choice(diffs,size=(2000,len(diffs)),replace=True),axis=1)
            summary.append(dict(pattern=pattern,strategy=strategy,mean_drop_difference=float(diffs.mean()),
                paired_bootstrap_95_percent_interval=[float(v) for v in np.percentile(means,[2.5,97.5])]))
    output['paired_summary']=summary
    # Overall paired routing summary across the 18 matched traffic trials.
    grouped={}
    for r in output['runs']:
        grouped.setdefault((r['pattern'], r['seed']), {})[r['strategy']]=r
    matched=[v for v in grouped.values() if all(k in v for k in ('ecmp','reactive','persistence','gnn','guarded_gnn'))]
    output['overall_summary']={
        s:{
            'mean_total_drops':float(np.mean([v[s]['total_drops'] for v in matched])),
            'mean_delivered_packets_per_tick':float(np.mean([v[s]['avg_throughput'] for v in matched])),
            'mean_delivered_latency_us':float(np.mean([v[s]['delivered_mean_latency_us'] for v in matched])),
            'mean_run_p99_latency_us':float(np.mean([v[s]['delivered_p99_latency_us'] for v in matched])),
            'mean_reroutes_per_run':float(np.mean([v[s]['reroutes'] for v in matched])),
        } for s in ('ecmp','reactive','persistence','gnn','guarded_gnn')
    }
    guarded_vs_ecmp=[v['guarded_gnn']['total_drops']-v['ecmp']['total_drops'] for v in matched]
    output['guarded_vs_ecmp']={
        'matched_trials':len(matched),
        'wins':int(sum(x<0 for x in guarded_vs_ecmp)),
        'ties':int(sum(x==0 for x in guarded_vs_ecmp)),
        'losses':int(sum(x>0 for x in guarded_vs_ecmp)),
        'mean_drop_difference':float(np.mean(guarded_vs_ecmp)),
        'mean_drop_reduction_percent':float(-np.mean(guarded_vs_ecmp)/np.mean([v['ecmp']['total_drops'] for v in matched])*100),
        'reroute_mutation_reduction_vs_reactive_percent':float((1-np.mean([v['guarded_gnn']['reroutes'] for v in matched])/np.mean([v['reactive']['reroutes'] for v in matched]))*100),
    }
    root=Path(__file__).parent/'research'; root.mkdir(exist_ok=True)
    (root/'benchmark_results.json').write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(output['prediction'],indent=2))

if __name__=='__main__': main()
