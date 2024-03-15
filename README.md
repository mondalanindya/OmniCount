# OmniCount

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/omnicount-multi-label-object-counting-with/object-counting-on-pascal-voc-2007-count-test)](https://paperswithcode.com/sota/object-counting-on-pascal-voc-2007-count-test?p=omnicount-multi-label-object-counting-with) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/omnicount-multi-label-object-counting-with/training-free-object-counting-on-fsc147)](https://paperswithcode.com/sota/training-free-object-counting-on-fsc147?p=omnicount-multi-label-object-counting-with) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/omnicount-multi-label-object-counting-with/training-free-object-counting-on-omnicount)](https://paperswithcode.com/sota/training-free-object-counting-on-omnicount?p=omnicount-multi-label-object-counting-with) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/omnicount-multi-label-object-counting-with/object-counting-on-omnicount-191)](https://paperswithcode.com/sota/object-counting-on-omnicount-191?p=omnicount-multi-label-object-counting-with)

## Authors
[Anindya Mondal*](https://scholar.google.com/citations?user=qjQmNJMAAAAJ&hl=en), [Sauradip Nag*](https://sauradip.github.io/), [Xiatian Zhu](https://surrey-uplab.github.io/), [Anjan Dutta](https://sites.google.com/site/2adutta/).

The code and the Omnicount-191 dataset will be released after publication. Please stay tuned! 

[[ArXiv]](https://arxiv.org/abs/2403.05435)
# Abstract

Object counting is pivotal for understanding the composition of scenes. Previously, this task was dominated by class-specific methods, which have gradually evolved into more adaptable class-agnostic strategies. However, these strategies come with their own set of limitations, such as the need for manual exemplar input and multiple passes for multiple categories, resulting in significant inefficiencies. This paper introduces a new, more practical approach enabling simultaneous counting of multiple object categories using an open vocabulary framework. Our solution, OmniCount, stands out by using semantic and geometric insights from pre-trained models to count multiple categories of objects as specified by users, all without additional training. OmniCount distinguishes itself by generating precise object masks and leveraging point prompts via the Segment Anything Model for efficient counting. To evaluate OmniCount, we created the OmniCount-191 benchmark, a first-of-its-kind dataset with multi-label object counts, including points, bounding boxes, and VQA annotations. Our comprehensive evaluation in OmniCount-191, alongside other leading benchmarks, demonstrates OmniCount's exceptional performance, significantly outpacing existing solutions and heralding a new era in object counting technology.


![image](https://github.com/mondalanindya/OmniCount/blob/main/assets/figs/pipeline_v2.png)




