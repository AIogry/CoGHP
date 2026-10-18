#!/usr/bin/env python3
"""Paired local execution diagnostic; never changes the formal evaluation.

teacher: fixed DATA waypoints, re-encoded from the current state each step.
predicted: native free-running latent planner on each step (no decoder repair).
Both start from the identical saved simulator state and use the same local
target (the data's nearest waypoint), budget and tolerance. An alternative
valid route can miss that reference waypoint: this is not a formal task score
or a universal upper bound. No fitting uses these validation trajectories.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'impls'),str(ROOT)]
import jax
import mujoco
import numpy as np
import ogbench
from agents.navigation import NavigationAgent as NavigationCoGHPAgent
from ogbench.utils import load_dataset
from utils.flax_utils import restore_agent


def restore_simulator(env, qpos, qvel, observation, goal_xy):
    env.reset(seed=173)
    base=env.unwrapped
    base.set_state(qpos.copy(),qvel.copy())
    mujoco.mj_forward(base.model,base.data)
    recovered=base.get_ob()
    error=float(np.max(np.abs(recovered-observation)))
    if not np.allclose(recovered,observation,atol=2e-4,rtol=2e-4):
        raise ValueError(f'State restoration failed observation check: max error={error}')
    base.set_goal(goal_xy=np.asarray(goal_xy))
    return recovered,error


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--step',type=int,required=True)
    p.add_argument('--dataset-root',type=Path,default=Path('/data/qijunrong/06-RL/offline-rl/data/raw_ogbench'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=24)
    p.add_argument('--budget',type=int,default=100)
    p.add_argument('--seed',type=int,default=727)
    args=p.parse_args()
    if args.output.exists():
        raise ValueError('Diagnostic output already exists; choose a new path.')
    flags=json.loads((args.artifacts/'flags.json').read_text())
    config={k:tuple(v) if isinstance(v,list) else v for k,v in flags['agent'].items()}
    if config['command_content']!='position':
        raise ValueError('This diagnostic requires the position command interface.')
    data_path=args.dataset_root/(flags['env_name']+'-val.npz')
    data=load_dataset(data_path,compact_dataset=True,add_info=True)
    if not {'qpos','qvel'}<=data.keys():
        raise ValueError('Dataset lacks simulator state; no rollout results can be produced.')
    env=ogbench.make_env_and_datasets(flags['env_name'],env_only=True)
    a=NavigationCoGHPAgent.create(flags['seed'],data['observations'][:1],data['actions'][:1],config)
    metadata=json.loads((args.artifacts/'checkpoint_metadata.json').read_text())
    a=restore_agent(a,str(args.artifacts),args.step,metadata,True)
    ends=np.flatnonzero(data['valids']==0)
    ids=np.flatnonzero(data['valids']>0)
    offsets=np.arange(config['num_subgoals'],0,-1)*config['subgoal_steps']
    terminal=ends[np.searchsorted(ends,ids)]
    ids=ids[ids+offsets[0]<=terminal]
    rng=np.random.default_rng(args.seed)
    rng.shuffle(ids)
    chosen=[]
    for t in ids:
        near=data['observations'][t+offsets[-1],:2]
        if np.linalg.norm(near-data['observations'][t,:2])>2*env.unwrapped._goal_tol:
            chosen.append(int(t))
        if len(chosen)==args.episodes:
            break
    if len(chosen)!=args.episodes:
        raise ValueError('Insufficient nontrivial local validation commands.')
    rows=[]
    for t in chosen:
        end=ends[np.searchsorted(ends,t)]
        goal=data['observations'][min(t+4*config['subgoal_steps'],end)]
        positions=data['observations'][t+offsets,:2]
        target=positions[-1]
        for mode in ('teacher','predicted'):
            ob,error=restore_simulator(env,data['qpos'][t],data['qvel'][t],data['observations'][t],target)
            initial_distance=float(np.linalg.norm(ob[:2]-target))
            success=False
            for step in range(args.budget):
                key=jax.random.fold_in(jax.random.PRNGKey(args.seed+t),step)
                action=(a.sample_with_commands(ob,goal,positions,key) if mode=='teacher'
                        else a.sample_actions(ob,goal,key,0.0))
                ob,_,terminated,truncated,info=env.step(np.asarray(action))
                success=bool(np.linalg.norm(ob[:2]-target)<=env.unwrapped._goal_tol)
                if success or terminated or truncated:
                    break
            rows.append(dict(index=t,mode=mode,success=success,steps=step+1,
                             restore_observation_max_error=error,
                             initial_distance=initial_distance,
                             final_distance=float(np.linalg.norm(ob[:2]-target))))
    summary={mode:dict(success_rate=float(np.mean([r['success'] for r in rows if r['mode']==mode])),
                       mean_steps=float(np.mean([r['steps'] for r in rows if r['mode']==mode])))
             for mode in ('teacher','predicted')}
    result=dict(protocol='fixed_data_waypoints_vs_native_replanning; common_reference_near_waypoint',
                caveat='Local diagnostic, not official task success. Alternate predicted routes may miss the reference waypoint.',
                artifacts=str(args.artifacts),checkpoint_step=args.step,dataset=str(data_path),
                seed=args.seed,budget=args.budget,tolerance=env.unwrapped._goal_tol,
                indices=chosen,summary=summary,episodes=rows)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    env.close()


if __name__=='__main__':
    main()
