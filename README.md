# Important

My work here is only part of the project of my team. This work is still incomplete and is only for the use of my research group. I would try to make the code more complete and write a better readme before September. Sorry for the inconvenience if you are just someone very interested in my work.

I modified the original code of Facebook's [Pyslowfast](https://github.com/facebookresearch/SlowFast). For any problem you encountered in the process of setting environment and running the demo, you can find the solutions in stack overflow, chatGPT, the issue section of [Pyslowfast](https://github.com/facebookresearch/SlowFast), etc..
This code is specific to the Demo of Slowfast, the Kinetics dataset in particular. 
For convenience, I uploaded all the code including the original README underneath.

## Download files for Kinetics
- [Kinetics class names json file](https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json) this one is needed (I renamed this fileName.json)
- [`SLOWFAST_8x8_R50.pkl`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWONLY_8x8_R50.pkl)
- [Kinetics parent-child class mapping](https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/parents.json) I didn't use this one.

# PySlowFast

PySlowFast is an open source video understanding codebase from FAIR that provides state-of-the-art video classification models with efficient training. This repository includes implementations of the following methods:

- [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982)
- [Non-local Neural Networks](https://arxiv.org/abs/1711.07971)
- [A Multigrid Method for Efficiently Training Video Models](https://arxiv.org/abs/1912.00998)
- [X3D: Progressive Network Expansion for Efficient Video Recognition](https://arxiv.org/abs/2004.04730)
- [Multiscale Vision Transformers](https://arxiv.org/abs/2104.11227)
- [A Large-Scale Study on Unsupervised Spatiotemporal Representation Learning](https://arxiv.org/abs/2104.14558)
- [MViTv2: Improved Multiscale Vision Transformers for Classification and Detection](https://arxiv.org/abs/2112.01526)
- [Masked Feature Prediction for Self-Supervised Visual Pre-Training](https://arxiv.org/abs/2112.09133)
- [Masked Autoencoders As Spatiotemporal Learners](https://arxiv.org/abs/2205.09113)
- [Reversible Vision Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Mangalam_Reversible_Vision_Transformers_CVPR_2022_paper.pdf)

<div align="center">
  <img src="demo/ava_demo.gif" width="600px"/>
</div>

## Introduction

The goal of PySlowFast is to provide a high-performance, light-weight pytorch codebase provides state-of-the-art video backbones for video understanding research on different tasks (classification, detection, and etc). It is designed in order to support rapid implementation and evaluation of novel video research ideas. PySlowFast includes implementations of the following backbone network architectures:

- SlowFast
- Slow
- C2D
- I3D
- Non-local Network
- X3D
- MViTv1 and MViTv2
- Rev-ViT and Rev-MViT

## Updates
 - We now [Reversible Vision Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Mangalam_Reversible_Vision_Transformers_CVPR_2022_paper.pdf). Both Reversible ViT and MViT models released. See [`projects/rev`](./projects/rev/README.md).
 - We now support [MAE for Video](https://arxiv.org/abs/2104.11227.pdf). See [`projects/mae`](./projects/mae/README.md) for more information.
 - We now support [MaskFeat](https://arxiv.org/abs/2112.09133). See [`projects/maskfeat`](./projects/maskfeat/README.md) for more information.
 - We now support [MViTv2](https://arxiv.org/abs/2104.11227.pdf) in PySlowFast. See [`projects/mvitv2`](./projects/mvitv2/README.md) for more information.
 - We now support [A Large-Scale Study on Unsupervised Spatiotemporal Representation Learning](https://arxiv.org/abs/2104.14558). See [`projects/contrastive_ssl`](./projects/contrastive_ssl/README.md) for more information.
 - We now support [Multiscale Vision Transformers](https://arxiv.org/abs/2104.11227.pdf) on Kinetics and ImageNet. See [`projects/mvit`](./projects/mvit/README.md) for more information.
 - We now support [PyTorchVideo](https://github.com/facebookresearch/pytorchvideo) models and datasets. See [`projects/pytorchvideo`](./projects/pytorchvideo/README.md) for more information.
 - We now support [X3D Models](https://arxiv.org/abs/2004.04730). See [`projects/x3d`](./projects/x3d/README.md) for more information.
 - We now support [Multigrid Training](https://arxiv.org/abs/1912.00998) for efficiently training video models. See [`projects/multigrid`](./projects/multigrid/README.md) for more information.
 - PySlowFast is released in conjunction with our [ICCV 2019 Tutorial](https://alexander-kirillov.github.io/tutorials/visual-recognition-iccv19/).

## License

PySlowFast is released under the [Apache 2.0 license](LICENSE).

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the PySlowFast [Model Zoo](MODEL_ZOO.md).

## Installation

Please find installation instructions for PyTorch and PySlowFast in [INSTALL.md](INSTALL.md). You may follow the instructions in [DATASET.md](slowfast/datasets/DATASET.md) to prepare the datasets.

## Quick Start

Follow the example in [GETTING_STARTED.md](GETTING_STARTED.md) to start playing video models with PySlowFast.

## Visualization Tools

We offer a range of visualization tools for the train/eval/test processes, model analysis, and for running inference with trained model.
More information at [Visualization Tools](VISUALIZATION_TOOLS.md).

## Contributors
PySlowFast is written and maintained by [Haoqi Fan](https://haoqifan.github.io/), [Yanghao Li](https://lyttonhao.github.io/), [Bo Xiong](https://www.cs.utexas.edu/~bxiong/), [Wan-Yen Lo](https://www.linkedin.com/in/wanyenlo/), [Christoph Feichtenhofer](https://feichtenhofer.github.io/).

## Citing PySlowFast
If you find PySlowFast useful in your research, please use the following BibTeX entry for citation.
```BibTeX
@misc{fan2020pyslowfast,
  author =       {Haoqi Fan and Yanghao Li and Bo Xiong and Wan-Yen Lo and
                  Christoph Feichtenhofer},
  title =        {PySlowFast},
  howpublished = {\url{https://github.com/facebookresearch/slowfast}},
  year =         {2020}
}
```
