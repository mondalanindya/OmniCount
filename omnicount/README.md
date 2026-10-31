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

## Paper evaluation settings

The paper reports SAN ViT-Large and SAM ViT-Base. The executable counting
pipelines use `sam_vit_b_01ec64.pth`; the ViT-H checkpoint is not the
published OmniCount configuration. Reference-point refinement uses the
reported 10-pixel depth window, depth threshold `0.3`, `K=16`, `C=256`,
Gaussian sigma `0.4`, and a four-pixel local-maxima spacing.

Counts are the number of retained SAM instance masks after empty or
insignificantly small masks have been discarded. `metrics/metric_multi.py`
computes mRMSE as the mean of the per-category RMSE values, matching the
paper's category-averaged definition.