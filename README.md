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

# OmniCount: Multi-label Object Counting with Semantic-Geometric Priors

## [AAAI 2025](https://aaai.org/Conferences/AAAI-25/)

[Download](https://github.com/mondalanindya/OmniCount/raw/refs/heads/main/OmniCount-191.zip) | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/34151) | [Code](https://github.com/mondalanindya/OmniCount/)

TL;DR: OmniCount introduces a training-free framework and the OmniCount-191 dataset for counting multiple object categories in a single pass using pre-trained semantic and geometric priors.

## Object Counting Paradigms

![OmniCount Teaser](https://raw.githubusercontent.com/mondalanindya/OmniCount/main/assets/figs/omnicount_teaser.png)

Typical single-label object counting methods usually process one category at a time. OmniCount instead targets multi-label, open-vocabulary counting in a single pass.

## Abstract

OmniCount is a practical framework for simultaneous counting of multiple object categories without extra training. It combines semantic and geometric priors from pre-trained models, produces precise masks, and supports varied prompts through SAM to improve counting accuracy.

## Video

## OmniCount: Model Design

![OmniCount Pipeline](https://raw.githubusercontent.com/mondalanindya/OmniCount/main/assets/figs/pipeline.png)

OmniCount starts from an input image and target object classes, uses semantic and geometric estimation modules to generate class-specific masks and depth maps, refines these priors, and then passes patches and reference points into SAM for final counting.

## Improving Counting using Priors

![OmniCount Pipeline](https://raw.githubusercontent.com/mondalanindya/OmniCount/main/assets/figs/refinement.png)

Reference point selection is improved by combining semantic priors, local maxima, and Gaussian refinement. Depth-guided recovery helps reduce over-segmentation and improves object recovery for occluded or distant instances.

## Results

Representative examples from the OmniCount-191 benchmark and related datasets show consistent multi-label counts across agriculture, birds, fruits, pets, urban scenes, and wildlife.

## OmniCount-191 Benchmark

![OmniCount-191 Benchmark](https://raw.githubusercontent.com/mondalanindya/OmniCount/main/assets/figs/omnicount191.png)

OmniCount-191 is a benchmark for multi-label object counting with 30,230 images and annotations for point, box, and VQA-style supervision.

## BibTeX

```bibtex
@inproceedings{mondal2025omnicount,
	title={Omnicount: Multi-label object counting with semantic-geometric priors},
	author={Mondal, Anindya and Nag, Sauradip and Zhu, Xiatian and Dutta, Anjan},
	booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
	volume={39},
	number={18},
	pages={19537--19545},
	year={2025}
}
```

## License

Open RAIL-S License.

Inspired by [Nerfies](https://nerfies.github.io/). Thanks to Manisha for UI/UX design insights.
