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
Gaussian sigma `0.4`, and a four-pixel local-maxima spacing. The paper does
not specify the automatic-mask grid size; this implementation makes its
fallback explicit as 32 points per side with a minimum retained region area
of 25 pixels.

Counts are the number of retained SAM instance masks after empty or
insignificantly small masks have been discarded. `metrics/metric_multi.py`
computes mRMSE as the mean of the per-category RMSE values over all images,
matching the paper's category-averaged definition. Missing labels are
treated as zero for both ground truth and prediction.

The original Marigold release's academic depth settings are used here:
checkpoint `prs-eth/marigold-depth-v1-0`, 50 denoising steps, and ensemble
size 10. The OmniCount paper names Marigold but does not state its exact
checkpoint or inference settings.
The OmniCount-191 zero-shot split contains 118 training classes and 73
test classes (26,978 and 3,252 images); its manifest is distributed with
the dataset rather than this code repository.