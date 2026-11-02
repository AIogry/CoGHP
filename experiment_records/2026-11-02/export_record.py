from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv,gzip,hashlib,io,json,math,shutil,tarfile

repo=Path('/home/eai/Research/CoGHP')
dest=repo/'experiment_records/2026-11-02'
dest.mkdir(parents=True,exist_ok=False)
base=Path('/data/qijunrong/06-RL/offline-rl/exp')
campaigns=[
 ('v0_grid','joint_planner_v0_antmaze_giant_width32_seed0_20261029'),
 ('v1_beta05_10_20','joint_planner_v1_antmaze_giant_predmean_valueonly_beta_seed0_20261031'),
 ('v1_beta25_30_35','joint_planner_v1_antmaze_giant_predmean_valueonly_beta25_30_35_seed0_20261031'),
]
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
inventory={};blob_paths={};summary=[];checkpoints=[];metadata=[]
def copy(p,target):
 target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
 assert sha(p)==sha(target)
 inventory[str(target.relative_to(dest))]={'original':str(p),'sha256':sha(target),'bytes':target.stat().st_size}
for alias,name in campaigns:
 root=base/name;out=dest/'campaigns'/alias;out.mkdir(parents=True)
 manifest=json.loads((root/'campaign.json').read_text());source=Path(manifest['source_dir'])
 for name in ['campaign.json','study.json','results.txt','launcher_runtime.json','launcher_process.json','launch_verification.json','deferred_launch_request.json']:
  if (root/name).exists():copy(root/name,out/name)
 modes={}
 for rel,h in manifest['source']['files'].items():
  p=source/rel;assert p.is_file() and not p.is_symlink() and sha(p)==h,rel
  blob_paths.setdefault(h,p);modes[rel]=p.stat().st_mode & 0o777
 dump(out/'source_modes.json',modes)
 metadata.append({'alias':alias,'original':str(root),'source_signature':manifest['source']['code_signature'],'git_base':manifest['source']['git_commit'],'source_files':len(modes)})
 for method,m in sorted(manifest['study']['methods'].items()):
  run=root/'runs'/method/'antmaze-giant-navigate-v0/seed_000/attempt_000'
  state=json.loads((run/'run.json').read_text());assert state['status']=='completed' and state['returncode']==0
  assert state['source_signature']==manifest['source']['code_signature']
  target=out/'runs'/method;copy(run/'run.json',target/'run.json')
  art=run/'artifacts'
  for filename in ['train.csv','eval.csv','flags.json','reproducibility.json','checkpoint_metadata.json','model_parameter_counts.json','position_statistics.json']:
   if (art/filename).exists():copy(art/filename,target/filename)
  for p in art.glob('training_state_*.json'):copy(p,target/p.name)
  evaluations=list(csv.DictReader((art/'eval.csv').open()))
  chosen={int(row['step']):row for row in evaluations if int(row['step']) in (800000,900000,1000000)}
  assert set(chosen)=={800000,900000,1000000}
  score=sum(float(chosen[s]['evaluation/overall_success']) for s in chosen)/3
  assert math.isclose(score,state['metric']['mean'],abs_tol=1e-12)
  cfg=json.loads((art/'flags.json').read_text())['agent']
  summary.append(dict(campaign=alias,method=method,h=cfg['planner_h_cycles'],l=cfg['planner_l_cycles'],M=cfg['num_subgoals'],
   high_alpha=cfg['high_alpha'],seed=0,score=score,
   checkpoint_scores={str(s):float(chosen[s]['evaluation/overall_success']) for s in sorted(chosen)},
   per_task_score=[sum(float(chosen[s][f'evaluation/task{i}_success']) for s in chosen)/3 for i in range(1,6)],
   command_source=cfg['train_action_chain_source'],actor_goal_rep_grad=cfg.get('actor_goal_rep_grad',True)))
  for step in sorted(chosen):
   p=art/f'params_{step}.pkl';assert p.is_file()
   checkpoints.append(dict(campaign=alias,method=method,step=step,path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
 print('Archived records:',alias,flush=True)

# One content-addressed archive preserves all three exact frozen source trees
# without triplicating shared simulation assets. Campaign manifests map paths
# to these SHA256 blob names; source_modes.json records executable permissions.
bundle=dest/'frozen_source_blobs.tar.gz'
with bundle.open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as gz:
  with tarfile.open(fileobj=gz,mode='w') as archive:
   for h,p in sorted(blob_paths.items()):
    info=tarfile.TarInfo('blobs/'+h);info.size=p.stat().st_size;info.mode=0o444;info.mtime=0
    with p.open('rb') as f:archive.addfile(info,f)
with tarfile.open(bundle,'r:gz') as archive:
 for member in archive:
  data=archive.extractfile(member).read();assert hashlib.sha256(data).hexdigest()==member.name.split('/')[-1]

analyses=['antmaze_giant_v0_final_analysis_20261030','antmaze_giant_v0_theory_audit_20261030',
          'antmaze_giant_weight_discussion_20261031','antmaze_giant_v1_attribution_20261031']
for name in analyses:
 folder=repo/'impls/exp'/name
 if not folder.exists():continue
 for p in sorted(folder.rglob('*')):
  if p.is_file() and p.suffix in ('.json','.csv','.html','.png','.py','.log'):
   copy(p,dest/'analyses'/name/p.relative_to(folder))
dump(dest/'scores.json',summary);dump(dest/'checkpoint_index.json',checkpoints);dump(dest/'file_inventory.json',inventory)
dump(dest/'manifest.json',dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),campaigns=metadata,runs=len(summary),
 checkpoint_count=len(checkpoints),checkpoint_bytes=sum(x['bytes'] for x in checkpoints),
 checkpoint_storage='Binary checkpoints and raw datasets remain at their original /data paths; this Git archive contains their provenance and checksum indexes, not the binaries.',
 source_archive=dict(path=bundle.name,sha256=sha(bundle),bytes=bundle.stat().st_size,unique_blobs=len(blob_paths),
  restore='Read campaigns/<alias>/campaign.json source.files; each value h maps to archive member blobs/h. Copy each blob to its corresponding relative path and restore permissions from source_modes.json.'),
 analysis_note='Historical analysis snapshots are preserved with their original capture dates; they may describe partial results. scores.json contains the verified complete results for all 24 runs.'))
print('Archive:',dest,'runs',len(summary),'checkpoints',len(checkpoints),'source bundle MiB',round(bundle.stat().st_size/2**20,2),flush=True)
