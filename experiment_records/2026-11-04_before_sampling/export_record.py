from pathlib import Path
import hashlib,json,shutil,tarfile,io
R=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
EXP=R/'impls/exp';CP=Path('/data/qijunrong/06-RL/offline-rl/exp/joint_planner_v2_antmaze_giant_predmean_detached_jointencoder_beta3_seed0_20261103')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
manifest={'purpose':'Preserve completed v2 code/records and November 2-4 diagnostics before high-only sampling changes. Full binary checkpoints remain in local frozen experiment archives.','campaign':str(CP),'files':[],'checkpoint_index':[]}
for p in CP.rglob('*'):
 if not p.is_file() or 'source' in p.relative_to(CP).parts:continue
 rel=p.relative_to(CP)
 if p.name.startswith('params_') and p.suffix=='.pkl':
  manifest['checkpoint_index'].append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size});continue
 if p.suffix not in ('.json','.csv','.txt'):continue
 to=OUT/'v2_campaign'/rel;to.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,to)
 manifest['files'].append({'original':str(p),'saved':str(to.relative_to(OUT)),'sha256':sha(to)})
analyses=['antmaze_giant_v1_review_20261102','antmaze_giant_v2_diagnosis_20261103','antmaze_giant_map_atlas_20261103','antmaze_giant_backtracking_audit_20261103','antmaze_giant_value_branch_diagnostics_20261103','v0_junction_weight_audit_20261104']
# Archive diagnostics in <=24MiB input shards so every individual Git object is small.
parts=[];chunk=[];size=0
for name in analyses:
 for p in sorted((EXP/name).rglob('*')):
  if not p.is_file() or '__pycache__' in p.parts or 'inputs' in p.relative_to(EXP/name).parts:continue
  n=p.stat().st_size
  if chunk and size+n>24*2**20:parts.append(chunk);chunk=[];size=0
  chunk.append(p);size+=n
if chunk:parts.append(chunk)
for i,files in enumerate(parts):
 out=OUT/f'analysis_materials_{i:02d}.tar.gz'
 with tarfile.open(out,'w:gz') as t:
  for p in files:t.add(p,arcname=str(p.relative_to(EXP)),recursive=False)
 manifest['files'].append({'saved':out.name,'sha256':sha(out),'bytes':out.stat().st_size,'members':[{ 'path':str(p.relative_to(EXP)),'sha256':sha(p),'bytes':p.stat().st_size} for p in files]})
full=Path('/home/eai/Research/docs/CoGHP/11-4/CoGHP_v0_junction_weight_audit_2026-11-04.zip')
manifest['portable_full_audit']={'path':str(full),'bytes':full.stat().st_size,'sha256':sha(full)}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(OUT/'README.md').write_text('# Pre-sampling backup, 2026-11-04\n\nThis commit preserves v2 code/config/tests, completed training/evaluation records, and six diagnostic bundles (including numerical queries, plots, reports and analysis scripts). Extract every analysis_materials_*.tar.gz into the same directory to restore the analyses.\n\nThe manifest records every archived input hash. Large raw train/validation datasets and binary checkpoints remain in the frozen /data campaigns and the complete portable local archive listed in manifest.json; they are not duplicated into Git. Original source/data paths inside reports are provenance, not portable Git-bundle links.\n')
print('backup shards',len(parts),'bytes',sum(p.stat().st_size for p in OUT.glob('*.tar.gz')))
