## Method At A Glance

1. Depth prediction: Marigold estimates depth maps for each image.
2. Open-vocabulary mask proposal: SAN produces class-conditioned binary masks.
3. Mask refinement: depth and morphology-based filtering improves mask quality.
4. Counting pipeline: SAM-backed region selection and CLIP-based scoring aggregate counts.
5. Evaluation: RMSE/relRMSE style metrics on dataset-specific outputs.

## Repository Layout

```
omniCount/
├── core/                      counting and inference pipelines
├── preprocessing/             depth/mask refinement and patch extraction
├── metrics/                   evaluation scripts
├── visualization/             overlays and analysis utilities
├── baselines/                 CLIPSeg baseline experiments
├── dataset_stat/              dataset summary helpers
├── scripts/                   utility scripts and project wrappers
├── external/                  upstream submodules (SAM, SAN, Marigold, etc.)
├── paths.py                   repo-relative path helpers
├── requirements.txt           outer-repo dependencies
└── README.md
```

`scripts/project_wrappers/` contains OmniCount-specific launch wrappers layered over upstream repos.

## Setup

```bash
#Clone the repo
cd omnicount
cd external/sam && pip install -e . && cd ../..
cd external/GroundingDINO && pip install -e . && cd ../..
cd external/SAN && pip install -r requirements.txt && cd ../..
cd external/Marigold && pip install -r requirements.txt && cd ../..

bash scripts/download_checkpoints.sh ./checkpoints
```

## Usage (Research Workflow)

Run from repo root.

```bash
# 1) Depth maps
python scripts/project_wrappers/marigold_run.py --input_rgb_dir outputs_sota/animals/images --output_dir outputs_sota/animals/depth
python scripts/project_wrappers/marigold_run_mod.py --input_rgb_dir outputs_sota/animals/images --output_dir outputs_sota/animals/depth

# 2) Binary masks
bash scripts/project_wrappers/san_multi.sh

# 3) Refinement and patches
python preprocessing/extract_bin_mask.py
python preprocessing/extract_zero.py
python preprocessing/extract_patch.py

# 4) Counting
python core/sam_mod_multi.py
python core/sam_mod_single.py
python core/sam_mod_box_multi.py
python core/sam_mod_box.py

# 5) Evaluation
python metrics/metric_multi.py
python metrics/metric_single.py
```
