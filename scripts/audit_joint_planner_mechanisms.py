#!/usr/bin/env python3
"""Read-only gradient routing and shared-head H-cycle probes on frozen models."""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=256)
    parser.add_argument('--batch-size',type=int,default=64)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ['JAX_PLATFORMS']='cpu'
    manifest=json.loads((args.analysis/'snapshot/campaign.json').read_text())
    source=Path(manifest['source_dir'])
    for relative,expected in manifest['source']['files'].items():
        if hashlib.sha256((source/relative).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Frozen source mismatch: {relative}')
    sys.path[:0]=[str(source/'impls'),str(source)]
    import jax
    import jax.numpy as jnp
    from agents.navigation import NavigationAgent
    from ogbench.utils import load_dataset
    from utils.datasets import Dataset,MultiHGCDataset
    from utils.flax_utils import restore_agent
    assert jax.default_backend()=='cpu'
    assert Path(inspect.getfile(NavigationAgent)).is_relative_to(source)
    reference=args.analysis/'snapshot/runs/hrm_h1l1_m3'
    flags=json.loads((reference/'flags.json').read_text())
    cfg={k:tuple(v) if isinstance(v,list) else v for k,v in flags['agent'].items()}
    path=Path(manifest['datasets'][flags['env_name']+'-val.npz']['path'])
    data=Dataset.create(**load_dataset(path,compact_dataset=True))
    query=MultiHGCDataset(data,cfg,rng=np.random.default_rng(20261030)).sample(2048,evaluation=True)
    saved=np.load(args.analysis/'checkpoint_probes_extended.queries.npz')
    assert np.array_equal(query['indices'],saved['indices'])
    assert np.array_equal(query['high_actor_goal_idxs'],saved['goal_indices'])
    assert np.array_equal(query['high_actor_target_idxs'],saved['target_indices'])
    query={k:v[:args.samples] for k,v in query.items()}
    assert args.samples%args.batch_size==0
    result=dict(protocol='frozen-mechanism-audit-v1',checkpoint=1000000,backend='cpu',complete=False,
        source_signature=manifest['source']['code_signature'],samples=args.samples,batch_size=args.batch_size,
        query_description=f'First {args.samples} of the exact 2048 paired raw-state queries used in the final analysis.',
        caveats=['Intermediate H states use the final shared head, which was trained only at the final H state.',
                 'Gradient statistics are four fixed validation batches at one checkpoint, not training history.',
                 'Normalized-command probes are offline action agreement only, not policy evaluation.',
                 'Action agreement uses raw deterministic Gaussian means before environment clipping.',
                 'No parameters, training code, or optimizer state are updated.'],runs={})
    def vec(tree):
        leaves=jax.tree_util.tree_leaves(tree)
        return np.concatenate([np.asarray(x).astype(np.float64).ravel() for x in leaves]) if leaves else np.zeros(1)
    def groups(gradient):
        actor=gradient['modules_actor_mixer']
        return dict(goal_rep=vec(gradient['modules_goal_rep']),value=vec(gradient['modules_value']),
            planner=vec({k:v for k,v in actor.items() if k in ('planner_core','planner_query_tokens','high_actor_head')}),
            executor=vec({k:v for k,v in actor.items() if k=='low_actor_head' or k.startswith('action_mixer_blocks_')}),
            feature_embed=vec(actor['feature_embed']),target_value=vec(gradient['modules_target_value']))
    methods=['hrm_h1l1_m1','hrm_h1l1_m3','hrm_h2l1_m1','hrm_h4l1_m1','hrm_h4l1_m3','hrm_h4l2_m3']
    for name in methods:
        directory=args.analysis/'snapshot/runs'/name
        flags=json.loads((directory/'flags.json').read_text())
        state=json.loads((directory/'run.json').read_text())
        config={k:tuple(v) if isinstance(v,list) else v for k,v in flags['agent'].items()}
        m,h,ell=config['num_subgoals'],config['planner_h_cycles'],config['planner_l_cycles']
        batch=dict(query)
        if m==1:
            for key in ('high_actor_targets','high_actor_target_idxs','high_actor_target_offsets','high_actor_target_final_idxs'):
                batch[key]=batch[key][:,-1:]
        agent=NavigationAgent.create(flags['seed'],batch['observations'][:1],batch['actions'][:1],config)
        metadata=json.loads((directory/'checkpoint_metadata.json').read_text())
        artifacts=Path(state['run_dir'])/'artifacts'
        agent=restore_agent(agent,str(artifacts),1000000,metadata,True)

        def stages(module,obs,goals,states):
            policy=module.modules['actor_mixer']
            core=policy.planner_core
            features=policy._encode_inputs(obs,goals)
            queries=jnp.tile(policy.planner_query_tokens,(obs.shape[0],1,1))
            inputs=jnp.concatenate([features,queries],axis=1)
            carry=core.initial_carry((obs.shape[0],))
            zH,zL=carry['z_H'],carry['z_L']
            labels=module.modules['goal_rep'](jnp.concatenate([jnp.broadcast_to(obs[:,None],states.shape),states],axis=-1))
            def execute(commands):
                return policy(obs,goals,jax.random.PRNGKey(0),subgoal_reps=commands,
                              temperature=0.,action_chain_source='teacher')[1].mode()
            teacher=execute(labels)
            outputs=[];actions=[];normalized_actions=[];hidden_changes=[]
            for _ in range(h):
                old=zH
                for _ in range(ell):zL=core.L_level(zL+zH+inputs)
                zH=core.H_level(zH+zL)
                means=jnp.stack([policy.high_actor_head(zH[:,2+j],temperature=1.).mode() for j in range(m)],axis=1)
                outputs.append(means);actions.append(execute(means))
                norm=jnp.linalg.norm(means,axis=-1,keepdims=True)
                normalized=means*jnp.sqrt(means.shape[-1])/jnp.maximum(norm,1e-8)
                normalized_actions.append(execute(normalized))
                hidden_changes.append(jnp.mean(jnp.linalg.norm(zH-old,axis=-1),axis=-1))
            native=policy(obs,goals,jax.random.PRNGKey(0),temperature=0.,return_latents=True)[3]
            return (labels,jnp.stack(outputs),teacher,jnp.stack(actions),jnp.stack(normalized_actions),
                    jnp.stack(hidden_changes),native,inputs.shape[1])

        @jax.jit
        def stage_forward(agent,b):
            variables={'params':agent.network.params,**agent.network.model_state}
            return agent.network.model_def.apply(variables,b['observations'],b['high_actor_goals'],b['high_actor_targets'],method=stages)

        outputs=stage_forward(agent,batch)
        labels,means,teacher,actions,normalized_actions,changes,native,token_count=[np.asarray(x) for x in outputs]
        max_error=float(np.max(np.abs(means[-1]-native)))
        assert np.allclose(means[-1],native,atol=2e-5,rtol=2e-5),max_error
        adv=np.asarray(agent.high_advantages(batch))
        weights=np.exp(np.minimum(config['high_alpha']*adv,np.log(config['awr_max_weight'])))
        record=dict(h=h,l=ell,M=m,planner_tokens=int(token_count),actor_parameter_keys=list(agent.network.params['modules_actor_mixer']),
            shared_head_final_matches_native_max_error=max_error,stages=[],gradients=[])
        for k in range(h):
            mse=((means[k]-labels)**2).mean(axis=-1)
            record['stages'].append(dict(H=k+1,mse_by_waypoint=mse.mean(axis=0).tolist(),
                weighted_mse_by_waypoint=(np.sum(mse*weights,axis=0)/np.sum(weights,axis=0)).tolist(),
                predicted_norm_by_waypoint=np.linalg.norm(means[k],axis=-1).mean(axis=0).tolist(),
                hidden_change_mean=float(changes[k].mean()),
                action_data_mse=float(np.mean((actions[k]-batch['actions'])**2)),
                action_teacher_mse=float(np.mean((actions[k]-teacher)**2)),
                normalized_command_action_data_mse=float(np.mean((normalized_actions[k]-batch['actions'])**2)),
                normalized_command_action_teacher_mse=float(np.mean((normalized_actions[k]-teacher)**2))))
        record['teacher_action_data_mse']=float(np.mean((teacher-batch['actions'])**2))
        def component_grad(objective):
            @jax.jit
            def calculate(agent,b):
                def loss(params):
                    if objective=='value':return agent.value_loss(b,params)[0]
                    _,hi,lo,_=agent.actor_loss(b,params,jax.random.PRNGKey(19))
                    return hi['joint_plan_loss'] if objective=='high' else lo['actor_loss']
                return jax.value_and_grad(loss)(agent.network.params)
            return calculate
        grad_functions={objective:component_grad(objective) for objective in ('value','high','low')}
        for start in range(0,args.samples,args.batch_size):
            b={k:v[start:start+args.batch_size] for k,v in batch.items()}
            gradients={};values={}
            for objective,fn in grad_functions.items():
                loss,g=fn(agent,b);gradients[objective]=groups(g);values[objective]=float(loss)
            norm={o:{part:float(np.linalg.norm(v)) for part,v in g.items()} for o,g in gradients.items()}
            cosine={}
            for part in ('goal_rep','feature_embed'):
                for a,c in (('value','high'),('value','low'),('high','low')):
                    u,v=gradients[a][part],gradients[c][part]
                    den=np.linalg.norm(u)*np.linalg.norm(v)
                    cosine[f'{part}:{a}:{c}']=float(u@v/den) if den>0 else None
            record['gradients'].append(dict(batch_start=start,loss=values,l2_norm_by_group=norm,cosine=cosine))
        result['runs'][name]=record
        (args.output/'mechanism_probes.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(method=name,planner_tokens=int(token_count),stages=record['stages'],gradient_norms_first=record['gradients'][0]['l2_norm_by_group'])),flush=True)
        jax.clear_caches()
    result['complete']=True
    (args.output/'mechanism_probes.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Completed:',args.output,flush=True)


if __name__=='__main__':main()
