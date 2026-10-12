#!/usr/bin/env python3
"""Read-only value/waypoint probes using maze geometry ONLY as diagnostics.

Cell BFS distance is a geometric proxy, not the robot's optimal temporal
distance. No graph information enters learning or action selection.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'impls'),str(ROOT)]
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
from agents.navigation_coghp import NavigationCoGHPAgent, stable_awr
from ogbench.utils import load_dataset
from utils.datasets import Dataset, MultiHGCDataset
from utils.flax_utils import restore_agent


def distance_grid(grid, goal):
    distance=np.full(grid.shape,-1,np.int32)
    if not (0<=goal[0]<grid.shape[0] and 0<=goal[1]<grid.shape[1]) or grid[goal]!=0:
        return distance
    distance[goal]=0
    pending=deque([goal])
    while pending:
        r,c=pending.popleft()
        for a,b in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if 0<=a<grid.shape[0] and 0<=b<grid.shape[1] and grid[a,b]==0 and distance[a,b]<0:
                distance[a,b]=distance[r,c]+1
                pending.append((a,b))
    return distance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--step',type=int,required=True)
    p.add_argument('--dataset-root',type=Path,default=Path('/data/qijunrong/06-RL/offline-rl/data/raw_ogbench'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=2048)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Diagnostic output already exists.')
    flags=json.loads((args.artifacts/'flags.json').read_text())
    c={k:tuple(v) if isinstance(v,list) else v for k,v in flags['agent'].items()}
    data=Dataset.create(**load_dataset(args.dataset_root/(flags['env_name']+'-val.npz'),compact_dataset=True))
    b=MultiHGCDataset(data,c,rng=np.random.default_rng(442)).sample(args.samples,evaluation=True)
    env=ogbench.make_env_and_datasets(flags['env_name'],env_only=True); base=env.unwrapped
    agent=NavigationCoGHPAgent.create(flags['seed'],b['observations'][:1],b['actions'][:1],c)
    agent=restore_agent(agent,str(args.artifacts),args.step,json.loads((args.artifacts/'checkpoint_metadata.json').read_text()),True)
    adv=np.asarray(jax.jit(agent.high_advantages)(b))
    weights=np.asarray(stable_awr(adv,c['high_alpha']))
    grids={}
    progress=[]; signs=[]; distances=[]
    def at(grid,xy):
        ij=base.xy_to_ij(xy)
        return grid[ij] if 0<=ij[0]<grid.shape[0] and 0<=ij[1]<grid.shape[1] else -1
    for i in range(args.samples):
        goal=base.xy_to_ij(b['high_actor_goals'][i,:2])
        if goal not in grids:grids[goal]=distance_grid(base.maze_map,goal)
        grid=grids[goal]; before=at(grid,b['observations'][i,:2])
        for j in range(c['num_subgoals']):
            after=at(grid,b['high_actor_targets'][i,j,:2])
            if min(before,after)>=0 and before!=after:
                progress.append(before-after);signs.append(float(adv[i,j]));distances.append(before)
    progress,signs,distances=map(np.asarray,(progress,signs,distances))
    def agreement(mask):
        return dict(count=int(mask.sum()),sign_agreement=float(np.mean(np.sign(progress[mask])==np.sign(signs[mask]))) if mask.any() else None)
    result=dict(protocol='BFS-cell geometric progress vs learned high-level advantage; diagnostics only',
                caveat='Cell distances are not dynamics-optimal temporal distances. Tied cells and invalid cells excluded.',
                checkpoint_step=args.step,artifacts=str(args.artifacts),samples=args.samples,
                all_pairs=agreement(np.ones(len(progress),bool)),
                distant_pairs=agreement(distances>=10),
                advantage_quantiles=np.quantile(adv,[0,.1,.5,.9,1]).tolist(),
                weight_cap_fraction=np.mean(weights>=99.999,axis=0).tolist(),
                weight_mean=weights.mean(axis=0).tolist())
    # A fixed-start, on-trajectory goal slice probes short-horizon plateaus.
    # These are BEHAVIOR time offsets, not optimal or shortest-path distances.
    starts=np.asarray(b['indices'])
    ends=np.flatnonzero(data['valids']==0)
    terminal=ends[np.searchsorted(ends,starts)]
    value=agent.network.select('global_value' if agent.global_mode else 'value')
    query=jax.jit(lambda s,g:jnp.mean(jnp.asarray(value(s,g)),axis=0))
    slices={}
    for offset in (0,5,10,25,50,100):
        future=data['observations'][np.minimum(starts+offset,terminal)]
        goals=future[:,:2] if agent.global_mode else future
        values=np.asarray(query(b['observations'],goals))
        slices[str(offset)]=dict(mean=float(values.mean()),std=float(values.std()),
                                quantiles=np.quantile(values,[0,.1,.5,.9,1]).tolist(),
                                mean_xy_distance=float(np.linalg.norm(future[:,:2]-b['observations'][:,:2],axis=-1).mean()),
                                complete_fraction=float(np.mean(starts+offset<=terminal)))
    result['behavior_goal_offset_value_slices']=slices
    if agent.position_mode:
        pos=agent.predicted_positions
        decoded=np.asarray(jax.jit(jax.vmap(pos,in_axes=(0,0,None)))(b['observations'],b['high_actor_goals'],jax.random.PRNGKey(6)))
        invalid=[]
        for q in decoded.reshape(-1,2):
            ij=base.xy_to_ij(q)
            invalid.append(not (0<=ij[0]<base.maze_map.shape[0] and 0<=ij[1]<base.maze_map.shape[1]) or base.maze_map[ij]!=0)
        result['decoded_waypoint_wall_cell_fraction']=float(np.mean(invalid))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));env.close()


if __name__=='__main__':main()
