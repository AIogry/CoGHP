import json,sys,time
import numpy as np
sys.path.insert(0,'/home/eai/Research/CoGHP/impls')
from utils.junction_sampling import JunctionRouteIndex,JunctionHighSampler,OFFSET_BIN_EDGES
root='/tmp/coghp_junction_preparation_verified/'
idx=JunctionRouteIndex.load(root+'junction_index')
zfile=np.load(root+'preparation_queries.npz');z={k:zfile[k] for k in zfile.files};zfile.close();report=json.load(open(root+'preparation_report.json'))
with np.load(report['dataset']) as raw:
 obs=raw['observations'];ends=np.flatnonzero(raw['terminals'])
assert np.array_equal(ends,idx.episode_ends)
# Every donor interval is bounded by its original physical episode and return.
for seg,(visit,cell,goals) in enumerate(zip(idx.segment_visits,idx.segment_cells,idx.segment_goals)):
 assert len(goals)>0 and np.all(np.diff(goals)>0)
 e=idx.events;v=int(visit)
 assert goals.min()>=e['exit_index'][v] and goals.max()<=e['episode_end'][v]-1
 assert e['return_index'][v]<0 or goals.max()<e['return_index'][v]
 assert np.all(idx.frame_episodes[goals]==e['episode'][v])
 assert np.all(idx.frame_cells[goals]==cell)
for v,starts in enumerate(idx.donor_starts):
 if len(starts):
  assert np.all(idx.frame_visits[starts]==v) and np.all(idx.frame_episodes[starts]==idx.events['episode'][v])
  assert np.all(starts+50>=idx.events['exit_index'][v])
  assert np.all(starts<=idx.events['episode_end'][v]-1)
assert len(set(zip(idx.candidate_segments.tolist(),idx.candidate_bins.tolist())))==len(idx.candidate_segments)
# Re-derive physical donor support independently, including episode exclusion.
sampler=JunctionHighSampler(idx,obs,probability=.5,seed=31)
trigger_rows=np.flatnonzero(z['triggered']);rng=np.random.default_rng(929)
for row in rng.choice(trigger_rows,size=min(100,len(trigger_rows)),replace=False):
 s=int(z['original_start'][row]);g=int(z['original_goal'][row]);v=int(idx.frame_visits[s]);e=idx.events
 J=int(e['junction'][v]);incoming=int(e['incoming'][v]);exit=int(e['first_exit'][v]);ep=int(e['episode'][v]);gc=int(idx.frame_cells[g]);b=int(np.searchsorted(OFFSET_BIN_EDGES,g-s,side='right')-1)
 expected=[]
 for seg in np.flatnonzero(idx.segment_cells==gc):
  dv=int(idx.segment_visits[seg]);starts=idx.donor_starts[dv];goals=idx.segment_goals[seg]
  if e['junction'][dv]!=J or e['incoming'][dv]!=incoming or e['first_exit'][dv]==exit or e['episode'][dv]==ep:continue
  lo,hi=OFFSET_BIN_EDGES[b:b+2]
  if any(np.any((goals-start>=lo)&(goals-start<hi)) for start in starts):expected.append(dv)
 actual,nv,ne=sampler._supported_pool(J,incoming,gc,b,exit,ep)
 actualv=idx.segment_visits[idx.candidate_segments[actual]]
 assert set(expected)==set(actualv.tolist())
 assert nv==len(set(expected)) and ne==len(set(idx.events['episode'][expected].tolist()))
# All triggered queries protect a goal cell already encountered before return.
for row in trigger_rows:
 s=int(z['original_start'][row]);g=int(z['original_goal'][row]);v=int(idx.frame_visits[s]);e=idx.events
 assert idx.frame_cells[g] not in set(idx.frame_cells[e['exit_index'][v]:e['return_index'][v]].tolist())
# Real train batch-256 speed, finite cache and no external RNG.
t=time.perf_counter();n=262144
for offset in range(0,n,256):
 sampler.sample(z['original_start'][offset:offset+256],z['original_goal'][offset:offset+256],z['original_targets'][offset:offset+256])
elapsed=time.perf_counter()-t
out=dict(passed=True,donor_segments_checked=len(idx.segment_goals),physical_visits_checked=len(idx.donor_starts),independent_support_queries_checked=100,trigger_goal_protection_queries_checked=len(trigger_rows),benchmark_queries=n,benchmark_batch=256,seconds=elapsed,milliseconds_per_batch=elapsed/(n/256)*1000,cache=str(sampler._supported_pool.cache_info()))
open('/tmp/coghp_independent_sampler_review.json','w').write(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
