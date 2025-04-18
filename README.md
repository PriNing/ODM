# ODM

![method](https://github.com/PriNing/ODM/blob/main/img/1.png)

This repository is the official implementation for the following paper:

[ODM: A Text-Image Further Alignment Pre-training Approach for Scene Text Detection and Spotting](https://arxiv.org/abs/2403.00303)

Chen Duan, Pei Fu, Shan Guo, Qianyi Jiang, Xiaoming Wei, CVPR 2024

Part of the code is inherited from [oCLIP](https://github.com/bytedance/oclip).


## Data
Download [SynthText](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)

## ODM model
Download [ODM](https://drive.google.com/file/d/1vzjFhy6LcGB7hXt548qJZby9s1vOSWJB/view?usp=sharing), and extract [RN50](https://drive.google.com/file/d/1ari7YD0qZ6JejCn3IYc-wu0PO4CCXncR/view?usp=sharing) from ODM.

We provide a script for converting model parameter names：
```Bash
python tools/convert2mmocr.py
```


## Train
Single-GPU:

```Bash
python -u src/training/main.py     \
--save-frequency 20     \
--report-to tensorboard     \
--train-data /path/to/data      \
--char-dict-pth /path/to/char    \
--gt_dir /path/to/gt \
--csv-img-key filepath     \
--csv-caption-key title     \
--warmup 10000     \
--batch-size=32  \
--lr=1e-4    \
--wd=0.1     \
--epochs=100     \
--workers=4    \
--model RN50_Seg_Clip  \
--gpu 0 \
--logs=/path/to/save \
--prefix demo \
```

Multi-GPU

```Bash
python -u src/training/main.py     \
--save-frequency 20     \
--report-to tensorboard     \
--train-data /path/to/data      \
--char-dict-pth /path/to/char    \
--gt_dir /path/to/gt \
--csv-img-key filepath     \
--csv-caption-key title     \
--warmup 10000     \
--batch-size=32  \
--lr=1e-4    \
--wd=0.1     \
--epochs=100     \
--workers=4    \
--model RN50_Seg_Clip  \
--logs=/path/to/save \
--prefix demo \
```


# Visualization

```Bash
sh infer.sh
```

![Visualization](https://github.com/PriNing/ODM/blob/main/img/2.png)


# Citation
```Text
@inproceedings{duan2024odm,
  title={ODM: A Text-Image Further Alignment Pre-training Approach for Scene Text Detection and Spotting},
  author={Duan, Chen and Fu, Pei and Guo, Shan and Jiang, Qianyi and Wei, Xiaoming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15587--15597},
  year={2024}
}
```
