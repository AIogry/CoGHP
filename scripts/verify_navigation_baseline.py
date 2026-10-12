#!/usr/bin/env python3
"""Compare a real historical HRM checkpoint through old and new baseline paths."""
import argparse
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'impls'), str(ROOT)]
import flax
import jax
import numpy as np
from agents.hrm_action_coghp import HRMActionCoGHPAgent
from agents.navigation_coghp import NavigationCoGHPAgent, get_config
from ogbench.utils import load_dataset
from utils.datasets import Dataset, MultiHGCDataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    config=get_config().to_dict()
    config.update(json.loads((args.artifacts/'flags.json').read_text())['agent'])
    config={k:tuple(v) if isinstance(v,list) else v for k,v in config.items()}
    data=Dataset.create(**load_dataset(args.dataset,compact_dataset=True))
    b=MultiHGCDataset(data,config,rng=np.random.default_rng(281)).sample(256)
    old=HRMActionCoGHPAgent.create(0,b['observations'][:1],b['actions'][:1],config)
    config['agent_name']='navigation_coghp'
    new=NavigationCoGHPAgent.create(0,b['observations'][:1],b['actions'][:1],config)
    with (args.artifacts/'params_1000000.pkl').open('rb') as f:
        state=pickle.load(f)['agent']
    old=flax.serialization.from_state_dict(old,state)
    new=flax.serialization.from_state_dict(new,state)
    key=jax.random.PRNGKey(811)
    pairs={
        'value_loss':(old.value_loss(b,old.network.params),new.value_loss(b,new.network.params)),
        'actor_loss':(old.actor_loss(b,old.network.params,key),new.actor_loss(b,new.network.params,key)),
        'actions':(old.sample_actions(b['observations'][0],b['high_actor_goals'][0],key,0),
                   new.sample_actions(b['observations'][0],b['high_actor_goals'][0],key,0)),
    }
    report={}
    for name,(a,b) in pairs.items():
        aa,ta=jax.tree_util.tree_flatten(a)
        bb,tb=jax.tree_util.tree_flatten(b)
        assert ta==tb
        errors=[]
        for x,y in zip(aa,bb):
            np.testing.assert_array_equal(x,y)
            errors.append(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))))
        report[name]={'exact_equal':True,'max_abs_difference':max(errors)}
    args.output.write_text(json.dumps(dict(checkpoint=str(args.artifacts/'params_1000000.pkl'),
                                         dataset=str(args.dataset),results=report),indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
