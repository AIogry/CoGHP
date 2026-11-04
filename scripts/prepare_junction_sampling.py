#!/usr/bin/env python3
"""Build a TRAIN-only junction index and audit an unchanged v0 sampler stream.

No simulator is created, and no agent/value updates are performed.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.dont_write_bytecode=True
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('junction_sampling_prepare',ROOT/'impls/utils/junction_sampling.py')
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
JunctionRouteIndex,JunctionHighSampler=module.JunctionRouteIndex,module.JunctionHighSampler


def sha(path):
    with open(path,'rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def distribution(values):
    values=np.asarray(values)
    if len(values)==0:
        return dict(n=0,mean=None,quantiles={})
    quantiles=dict(zip(('min','q10','median','q90','max'),map(float,np.quantile(values,[0,.1,.5,.9,1]))))
    return dict(n=len(values),mean=float(values.mean()),quantiles=quantiles)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,type=Path)
    p.add_argument('--flags',required=True,type=Path,help='Frozen v0 flags JSON providing sampler configuration.')
    p.add_argument('--sampler-source',type=Path,default=ROOT,help='Frozen source root with impls/utils/datasets.py.')
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--sample-count',type=int,default=1048576)
    p.add_argument('--batch-size',type=int,default=16384)
    p.add_argument('--sampler-seed',type=int,default=20261105)
    p.add_argument('--replacement-seed',type=int,default=2026110501)
    p.add_argument('--probability',type=float,default=.5)
    p.add_argument('--min-visits',type=int,default=5)
    p.add_argument('--min-episodes',type=int,default=3)
    args=p.parse_args()
    if args.sample_count < 1 or args.batch_size < 1:
        p.error('Positive sample count and batch size required.')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if (args.output_dir/'preparation_report.json').exists():
        raise FileExistsError('Refuse to replace a completed preparation report; choose another output directory.')
    dataset_sha=sha(args.dataset)
    with np.load(args.dataset,allow_pickle=False) as raw:
        observations=raw['observations'];ends=np.flatnonzero(raw['terminals'])
    index=JunctionRouteIndex.build(observations,ends)
    index.save(args.output_dir/'junction_index')
    print('INDEX',json.dumps(index.metadata),flush=True)
    # Only the legacy sampler is imported from the selected frozen source.
    sys.path[:0]=[str(args.sampler_source/'impls'),str(args.sampler_source)]
    from ogbench.utils import load_dataset
    from utils.datasets import Dataset,MultiHGCDataset
    config=json.loads(args.flags.read_text())['agent']
    if config['subgoal_steps']!=50 or config['num_subgoals'] not in (1,2,3):
        raise ValueError('This protocol requires fixed50 and M<=3.')
    data=Dataset.create(**load_dataset(args.dataset,compact_dataset=True))
    if not np.array_equal(np.flatnonzero(np.asarray(data['valids'])==0),ends):
        raise AssertionError('Raw terminal and compact invalid-final boundaries disagree.')
    legacy=MultiHGCDataset(data,config,rng=np.random.default_rng(args.sampler_seed))
    unchanged=MultiHGCDataset(data,config,rng=np.random.default_rng(args.sampler_seed))
    sampler=JunctionHighSampler(index,observations,args.probability,args.replacement_seed,args.min_visits,args.min_episodes)
    arrays={k:[] for k in ('original_start','original_goal','original_targets','start_indices','goal_indices','target_indices','replaced','triggered','supported','support_visits','support_episodes','donor_visit_ids')}
    begin=time.monotonic();sampler_seconds=0.
    for offset in range(0,args.sample_count,args.batch_size):
        n=min(args.batch_size,args.sample_count-offset)
        batch=legacy.sample(n,evaluation=True)
        base=(batch['indices'],batch['high_actor_goal_idxs'],batch['high_actor_target_idxs'])
        preserved=tuple(x.copy() for x in base)
        tick=time.monotonic();result=sampler.sample(*base);sampler_seconds+=time.monotonic()-tick
        control=unchanged.sample(n,evaluation=True)
        for key,actual,expected in zip(('indices','high_actor_goal_idxs','high_actor_target_idxs'),base,preserved):
            np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(actual,control[key])
        # Low/value draws and all legacy payload remain byte-identical to control.
        for key in ('observations','actions','value_goals','low_actor_goals','high_actor_goals','high_actor_targets','rewards','masks'):
            np.testing.assert_array_equal(batch[key],control[key])
        for key,array in zip(('original_start','original_goal','original_targets'),base):arrays[key].append(np.asarray(array))
        for key in arrays:
            if not key.startswith('original_'):arrays[key].append(result[key])
    values={k:np.concatenate(v) for k,v in arrays.items()}
    replaced=values['replaced'];source=values['original_start'];goal=values['original_goal'];new=values['start_indices'];new_goal=values['goal_indices'];original_visits=index.frame_visits[source];donors=values['donor_visit_ids'];rows=np.flatnonzero(replaced)
    checks={
        'original_low_value_sampler_stream_unchanged':True,
        'unreplaced_query_indices_unchanged':bool(np.array_equal(new[~replaced],source[~replaced]) and np.array_equal(new_goal[~replaced],goal[~replaced]) and np.array_equal(values['target_indices'][~replaced],values['original_targets'][~replaced])),
        'same_junction_cell':bool(np.array_equal(index.frame_cells[new[rows]],index.frame_cells[source[rows]])),
        'same_goal_cell':bool(np.array_equal(index.frame_cells[new_goal[rows]],index.frame_cells[goal[rows]])),
        'same_incoming':bool(np.array_equal(index.events['incoming'][donors[rows]],index.events['incoming'][original_visits[rows]])),
        'different_exit':bool(np.all(index.events['first_exit'][donors[rows]]!=index.events['first_exit'][original_visits[rows]])),
        'different_episode':bool(np.all(index.frame_episodes[new[rows]]!=index.frame_episodes[source[rows]])),
        'same_temporal_offset_bin':bool(np.array_equal(np.searchsorted(module.OFFSET_BIN_EDGES,new_goal[rows]-new[rows],side='right'),np.searchsorted(module.OFFSET_BIN_EDGES,goal[rows]-source[rows],side='right'))),
        'same_50_100_150_clipping':bool(np.array_equal((new_goal[rows]-new[rows])[:,None]<[50,100,150],(goal[rows]-source[rows])[:,None]<[50,100,150])),
        'all_donor_targets_real_within_episode':bool(np.all(index.frame_episodes[values['target_indices'][rows]]==index.frame_episodes[new[rows,None]])),
        'all_donor_goals_before_return':bool(np.all((index.events['return_index'][donors[rows]]<0)|(new_goal[rows]<index.events['return_index'][donors[rows]]))),
        'all_donor_nearest50_after_exit':bool(np.all(values['target_indices'][rows,-1]>=index.events['exit_index'][donors[rows]])),
        'support_meets_independent_threshold':bool(np.all(values['support_visits'][rows]>=args.min_visits) and np.all(values['support_episodes'][rows]>=args.min_episodes)),
        'dataset_hash_unchanged':sha(args.dataset)==dataset_sha,
    }
    assert all(checks.values()),checks
    offset_bins=np.searchsorted(module.OFFSET_BIN_EDGES,goal-source,side='right')-1
    by_bin=[]
    for b,(lo,hi) in enumerate(zip(module.OFFSET_BIN_EDGES[:-1],module.OFFSET_BIN_EDGES[1:])):
        selected=offset_bins==b
        by_bin.append(dict(bin=b,lower=int(lo),upper=int(hi),original_queries=int(selected.sum()),triggered=int((selected&values['triggered']).sum()),supported=int((selected&values['supported']).sum()),replaced=int((selected&replaced).sum())))
    dt=goal[rows]-source[rows];new_dt=new_goal[rows]-new[rows]
    report=dict(protocol='junction_soft_v1',dataset=str(args.dataset.resolve()),dataset_sha256=dataset_sha,
        frozen_sampler_source=str(args.sampler_source.resolve()),flags=str(args.flags.resolve()),index_signature=index.signature,
        index=index.metadata,geometry=index.geometry,probability=args.probability,min_visits=args.min_visits,min_episodes=args.min_episodes,
        sampler_seed=args.sampler_seed,replacement_seed=args.replacement_seed,query_count=len(source),
        triggered_count=int(values['triggered'].sum()),supported_count=int(values['supported'].sum()),replaced_count=int(replaced.sum()),
        triggered_fraction=float(values['triggered'].mean()),supported_fraction=float(values['supported'].mean()),replaced_fraction=float(replaced.mean()),
        supported_physical_visits=int(len(np.unique(original_visits[values['supported']]))),
        supported_episodes=int(len(np.unique(index.frame_episodes[source[values['supported']]]))),
        supported_junctions=int(len(np.unique(index.frame_cells[source[values['supported']]]))),
        drift=dict(start_xy=distribution(np.linalg.norm(observations[new[rows],:2]-observations[source[rows],:2],axis=-1)),
            start_full_state_l2=distribution(np.linalg.norm(observations[new[rows]]-observations[source[rows]],axis=-1)),
            start_xy_velocity=distribution(np.linalg.norm(observations[new[rows],15:17]-observations[source[rows],15:17],axis=-1)),
            original_goal_offset=distribution(dt),donor_goal_offset=distribution(new_dt),goal_offset_difference=distribution(new_dt-dt)),
        checks=checks,by_temporal_bin=by_bin,sampling_wall_seconds=time.monotonic()-begin,
        replacement_sampler_seconds=sampler_seconds,replacement_seconds_per1024=sampler_seconds/len(source)*1024,
        support_rule='clean alternative side >=5 distinct visits and >=3 raw episodes AFTER excluding source episode and source first exit; no bad-side threshold; no spatial caliper',
        trigger_rule='nearest50 label after first exit and before first return; chosen goal at/after return; its cell not reached between first exit and return',
        donor_rule='uniform eligible physical visit, then uniform eligible real core start, then uniform real goal index in same cell and temporal bin before first return',
        limitations=['same incoming and source cell do not fix identical continuous position, posture, or velocity','same temporal bin preserves clipping and coarse time difficulty, not exact goal offset','no return is a sampling preference, not a correctness label','topology inferred from training occupancy only; no shortest-path selection'])
    np.savez_compressed(args.output_dir/'preparation_queries.npz',**values)
    (args.output_dir/'preparation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output_dir/'replacement_rng_state.json').write_text(json.dumps(sampler.get_rng_state(),indent=2)+'\n')
    print('REPORT',json.dumps({k:v for k,v in report.items() if k not in ('geometry','index','by_temporal_bin')},indent=2),flush=True)


if __name__=='__main__':
    main()
