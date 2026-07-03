# SAGE

**SAGE: Semantic-Guided Framework with Decoupled Optimization for Open-Vocabulary Video Visual Relationship Detection**

SAGE has been accepted by *Neural Networks*. This repository provides the baseline implementation built on our AAAI 2024 work, "Multi-modal Prompting for Open-vocabulary Video Visual Relationship Detection".

## Overview

Open-vocabulary video visual relationship detection aims to recognize relationships beyond annotated categories, including unseen interactions between seen and unseen objects. SAGE addresses two key challenges in dynamic video scenes: the visual-semantic gap caused by weak spatio-temporal cues, and optimization instability caused by noisy instance-level visual representations. The framework introduces multimodal LLM-guided semantic reasoning to enrich visual representations with structured interaction descriptions, and a decoupled class-aware prompting strategy that adapts classifiers from stable class-level textual knowledge. Experiments on VidVRD and VidOR demonstrate strong generalization, especially on novel relationship categories.

## Prerequisites

- pytorch=2.0.1
- python=3.8.17
- torchvision=0.15.2
- tqdm
- pillow
- ftfy
- regex

You can also run the following commands to prepare the conda environment.

```bash
bash conda.sh
conda activate MMP_OV_VidVRD
```

## Data And Checkpoints

VidVRD frame annotations can be downloaded from the official VidVRD page:
https://xdshang.github.io/docs/imagenet-vidvrd.html

The category information, trajectory information, and testing ground-truth relation files used by this project are under `dataset/vidvrd/data`.

Pre-extracted features are required for evaluation. The original feature download link is:
https://pan.baidu.com/s/1h1A2Qfcj6oEW8VJDYKyRlA?pwd=a8s6

Trained checkpoints are required for evaluation. The original checkpoint download link is:
https://pan.baidu.com/s/1is8cNDm0_Ni3XeQawGQRwg?pwd=9pe2

Expected local layout after downloading external assets:

```text
dataset/
  vidvrd/
    anno/
      train/
      test/
    data/
    feature/
    model/
```

## Evaluation

Put the downloaded checkpoint under `dataset/vidvrd/model`, or pass its path with `CKPT_PATH`.

```bash
cd scripts
CKPT_PATH=../dataset/vidvrd/model/stage2_tcp.pth bash test_vidvrd_openvoc.sh
```

## Citation

If you find this repository useful, please cite our SAGE paper:

```bibtex
@article{wang2026sage,
  title={SAGE: Semantic-Guided Framework with Decoupled Optimization for Open-Vocabulary Video Visual Relationship Detection},
  author={Wang, Shiqi and Xue, Weiying and Hu, Shuyi and Li, Haowen and Liu, Qi},
  journal={Neural Networks},
  volume={203},
  pages={109183},
  year={2026},
  doi={10.1016/j.neunet.2026.109183}
}
```
