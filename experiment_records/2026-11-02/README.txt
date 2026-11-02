CoGHP experiment archive — 2026-11-02 (Asia/Shanghai)

This archive preserves the completed v0 grid and both v1 beta sweeps before
any proposed change to encoder gradients or planner/executor coupling.

Coverage:
- 3 campaigns, 24 completed seed-0 runs, all with return code 0.
- Complete train.csv and eval.csv logs, resolved flags, run metadata,
  reproducibility records, checkpoint signatures and parameter counts.
- Verified scores at exactly 800k, 900k and 1M in scores.json.
- Earlier analysis reports and their original dated diagnostic snapshots.
- All three exact frozen source snapshots, deduplicated into one source bundle.

The 72 binary checkpoints (about 4 GiB) and raw datasets remain at their
original /data locations. They are NOT uploaded by this archive. Checkpoint
paths, sizes and SHA256 hashes are recorded in checkpoint_index.json.

To reconstruct a frozen source snapshot, read source.files in the chosen
campaigns/<alias>/campaign.json. Each path maps to a SHA256 hash h, whose exact
bytes are stored as blobs/h in frozen_source_blobs.tar.gz. Restore file modes
from campaigns/<alias>/source_modes.json. The bundle was decompressed and every
member hash verified during export.

The historical analysis reports retain their original capture dates and may
describe unfinished runs. Use scores.json for the complete 24-run results.
Snapshot logs and generated figures are preserved rather than rewritten.

manifest.json documents archive scope and provenance. file_inventory.json
maps copied records to their original paths and hashes. archive_sha256.json
covers every archived file except itself.
