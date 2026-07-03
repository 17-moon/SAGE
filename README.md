# MMP_OV_VidVRD

Implementation for the paper "Multi-modal Prompting for Open-vocabulary Video Visual Relationship Detection" (AAAI 2024).

## Public Release

This GitHub-clean copy keeps the lightweight evaluation/model code and omits large or private artifacts:

- Model checkpoints and pretrained weights are not included.
- Image files and generated plots are not included.
- Feature extraction scripts are not included.
- Training scripts are not included.
- Visualization scripts are not included.

The public `scripts/model_zoo` folder intentionally keeps only:

- `model_stage1.py`
- `stage2_tcp.py`

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

```bibtex
@inproceedings{yang2024multi,
  title={Multi-Modal Prompting for Open-Vocabulary Video Visual Relationship Detection},
  author={Yang, Shuo and Wang, Yongqi and Ji, Xiaofeng and Wu, Xinxiao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={7},
  pages={6513--6521},
  year={2024}
}
```
