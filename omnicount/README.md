# OmniCount Project Folder

This folder contains the research code for OmniCount: preprocessing, counting pipelines, metrics, visualization, and helper scripts.

## Key Pieces

- `core/`: counting and refinement pipelines
- `preprocessing/`: depth and mask cleanup, patch extraction
- `metrics/`: evaluation scripts
- `visualization/`: overlays and inspection utilities
- `baselines/`: CLIPSeg baseline experiments
- `scripts/project_wrappers/`: repo-local launch wrappers for external submodules

## Setup

Install the outer dependencies and submodule dependencies from the repository root, then run the scripts from that root so relative paths resolve correctly.

## Notes

Some scripts retain historical experiment blocks for traceability. Generated outputs are not committed and are expected to be recreated locally.