

### Training
Use `train.py` to train ESANet on NYUv2, SUNRGB-D, Cityscapes, or SceneNet RGB-D
(or implement your own dataset by following the implementation of the provided 
datasets).
The arguments default to training ESANet-R34-NBt1D on NYUv2 with the 
hyper-parameters from our paper. Thus, they could be omitted but are presented 
here for clarity.

> Note that training ESANet-R34-NBt1D requires the pretrained weights for the 
encoder backbone ResNet-34 NBt1D. You can download our pretrained weights on 
ImageNet from [Link](https://drive.google.com/uc?id=1neUb6SJ87dIY1VvrSGxurVBQlH8Pd_Bi). 
Otherwise, you can use `imagenet_pretraining.py` to create your own pretrained weights.


- Dependency
    - Ubuntu20.04
    - Pytorch1.12.1
    - python 3.8 
    - RTX3090



- Train on NYUv2:
    ```bash
    modified the path to NYUV2 dataset via parameter --dataset_dir
    bash train.sh
    ```
- Eval
    ```bash
    cd test

    python train_missing.py   --dataset nyuv2     --dataset_dir /home/quick_data/shicaiwei/NYUDV2    --pretrained_dir ./trained_models/imagenet --results_dir /home/ssd/results/nyuv2/rgbd_r50_dul_auxi_kl3_auxi_0.5  --gpu 1   --modality rgbd 
    ```

- File note
    - train_missing_dul_auxi_amp.py: use amp to accelerating training. However, it may lead to gradient NaN
    - train_missing.py: vanilla method without DMRNet.


- Code base
    - [MMANet](https://github.com/shicaiwei123/MMANET-CVPR2023)
    - [ESANet](https://github.com/TUI-NICR/ESANet)