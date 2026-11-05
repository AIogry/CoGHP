from pathlib import Path
import json,hashlib,tarfile,shutil
R=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
C=Path('/data/qijunrong/06-RL/offline-rl/exp/joint_planner_v0_junction_sampling_hlm_seed01_20261104')
A=R/'impls/exp/junction_sampling_seed0_analysis_20261105'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
archive=OUT/'analysis_materials.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
 for p in sorted(A.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts:t.add(p,arcname=str(Path(A.name)/p.relative_to(A)),recursive=False)
files=[];checkpoints=[]
for p in C.glob('runs/**/seed_001/**/artifacts/*'):
 if not p.is_file():continue
 if p.name.startswith('params_') and p.suffix=='.pkl':checkpoints.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)});continue
 if p.suffix not in ('.csv','.json'):continue
 target=OUT/'paused_seed1'/p.relative_to(C);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);files.append({'saved':str(target.relative_to(OUT)),'original':str(p),'sha256':sha(target)})
for filename in ['pause_record_20261105.json','launcher_process.json','launcher_runtime.json','campaign.json','study.json','launch_verification.json','results.txt']:
 p=C/filename
 if p.exists():shutil.copy2(p,OUT/filename);files.append({'saved':filename,'original':str(p),'sha256':sha(OUT/filename)})
for p in C.glob('runs/**/seed_001/**/run.json'):
 target=OUT/'paused_seed1'/p.relative_to(C);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
meta={'purpose':'Preserve completed sampling study analysis and paused seed1 records before separate-high-value changes.','source_commit':'27f8b0db04642a6bee6fb80c6f408eeb32f272f5','analysis_archive':{'path':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive)},'files':files,'paused_checkpoint_index':checkpoints,'note':'Paused run.json can retain running because processes are SIGSTOP. pause_record is authoritative. No process resumed or terminated. Rawdata and fullcheckpoints remain in frozen campaigns.'}
(OUT/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
(OUT/'README.md').write_text('# Before independent high value, 2026-11-05\n\nComplete seed0 report, numerical controls, counterfactual traces, figures and analysis code are preserved in analysis_materials.tar.gz. Paused seed1 logs/metadata and checkpoint hashes are recorded separately. Existing raw datasets/checkpoints remain in /data; live paused optimizer state remains in its original processes. This backup does not resume or terminate them.\n')
with tarfile.open(archive,'r:gz') as t:
 for m in t:
  rel=Path(m.name).relative_to(A.name);assert hashlib.sha256(t.extractfile(m).read()).hexdigest()==sha(A/rel)
print('Verified backup',archive.stat().st_size,'bytes')
