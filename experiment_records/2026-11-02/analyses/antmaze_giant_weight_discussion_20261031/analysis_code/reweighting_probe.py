import os,sys,json,hashlib
from pathlib import Path
import numpy as np
os.environ['JAX_PLATFORMS']='cpu'
previous=Path('/home/eai/Research/CoGHP/impls/exp/antmaze_giant_v0_final_analysis_20261030')
root=Path('/home/eai/Research/CoGHP/impls/exp/antmaze_giant_weight_discussion_20261031')
root.mkdir(exist_ok=True)
manifest=json.loads((previous/'snapshot/campaign.json').read_text())
source=Path(manifest['source_dir'])
sys.path[:0]=[str(source/'impls'),str(source)]
import jax
import jax.numpy as jnp
from agents.navigation import NavigationAgent
from ogbench.utils import load_dataset
from utils.datasets import Dataset,MultiHGCDataset
from utils.flax_utils import restore_agent
reference=json.loads((previous/'snapshot/runs/hrm_h1l1_m3/flags.json').read_text())
config={k:tuple(v) if isinstance(v,list) else v for k,v in reference['agent'].items()}
data=Dataset.create(**load_dataset(manifest['datasets'][reference['env_name']+'-val.npz']['path'],compact_dataset=True))
q=MultiHGCDataset(data,config,rng=np.random.default_rng(20261030)).sample(2048,evaluation=True)
saved=np.load(previous/'checkpoint_probes_extended.queries.npz')
assert np.array_equal(saved['target_indices'],q['high_actor_target_idxs'])
result=dict(protocol='readonly-value-difference-reweighting',samples=2048,checkpoint=1000000,
    order='near-to-far (50,100,150 nominal steps; endpoints may be clipped)',
    caveat='Weights re-evaluated under fixed existing critics; no training or performance evaluation.',
    source_signature=manifest['source']['code_signature'],runs={})
for name in ('hrm_h1l1_m3','hrm_h4l2_m3'):
    folder=previous/'snapshot/runs'/name
    flags=json.loads((folder/'flags.json').read_text())
    config={k:tuple(v) if isinstance(v,list) else v for k,v in flags['agent'].items()}
    model=NavigationAgent.create(0,q['observations'][:1],q['actions'][:1],config)
    run=json.loads((folder/'run.json').read_text())
    agent=restore_agent(model,str(Path(run['run_dir'])/'artifacts'),1000000,json.loads((folder/'checkpoint_metadata.json').read_text()),True)
    @jax.jit
    def values(agent,obs,targets,goals):
        v=agent.network.select('value')
        return (jnp.mean(v(obs,goals),0),jnp.stack([jnp.mean(v(targets[:,i],goals),0) for i in range(3)],1),jnp.mean(v(goals,goals),0))
    initial,far_to_near,goal_value=[np.asarray(x) for x in values(agent,q['observations'],q['high_actor_targets'],q['high_actor_goals'])]
    future=far_to_near[:,::-1]
    cumulative=future-initial[:,None]
    segment=future-np.concatenate([initial[:,None],future[:,:-1]],1)
    assert np.array_equal(cumulative[:,0],segment[:,0])
    assert np.allclose(segment.cumsum(1),cumulative,atol=1e-5)
    forms={'cumulative':cumulative,'adjacent_segment':segment,'negative_remaining':future-goal_value[:,None]}
    output={}
    for kind,a in forms.items():
        output[kind]={}
        for beta in (3.,1.,.3):
            w=np.exp(np.minimum(beta*a,np.log(100)))
            output[kind][str(beta)]=dict(cap_fraction=(w>=100*(1-1e-5)).mean(0).tolist(),
                mean_weight=w.mean(0).tolist(),ess_fraction=(w.sum(0)**2/(len(w)*(w*w).sum(0))).tolist(),
                positive_advantage_fraction=(a>0).mean(0).tolist())
    result['runs'][name]=output
    np.savez_compressed(root/(name+'_value_samples.npz'),initial=initial,future_near_to_far=future,goal_self=goal_value,
        query_indices=q['indices'],goal_indices=q['high_actor_goal_idxs'],target_indices_near_to_far=q['high_actor_target_idxs'][:,::-1])
    print(name,json.dumps(output['cumulative']),json.dumps(output['adjacent_segment']),flush=True)
(root/'weight_forms.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(root/'weight_forms.json',flush=True)
