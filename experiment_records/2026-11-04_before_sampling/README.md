# Pre-sampling backup, 2026-11-04

This commit preserves v2 code/config/tests, completed training/evaluation records, and six diagnostic bundles (including numerical queries, plots, reports and analysis scripts). Extract every analysis_materials_*.tar.gz into the same directory to restore the analyses.

The manifest records every archived input hash. Large raw train/validation datasets and binary checkpoints remain in the frozen /data campaigns and the complete portable local archive listed in manifest.json; they are not duplicated into Git. Original source/data paths inside reports are provenance, not portable Git-bundle links.
