# Joint Optimization Framework for Learning with Noisy Labels
This repository contains the code for the paper [Joint Optimization Framework for Learning with Noisy Labels](https://arxiv.org/abs/1803.11364).

## Requirements
- Python 3.6
- Chainer 4.0.0
- CuPy 4.0.0
- ChainerCV 0.9.0

## Training
To train the network on the Symmmetric Noise CIFAR-10 dataset (noise rate = 0.7):

    $ python first_step_train.py --gpu 0 --out first_sn07 --learnrate 0.08 --alpha 1.2 --beta 0.8 --percent 0.7
    $ python second_step_train.py --gpu 0 --out second_sn07 --label first_sn07

To train the network on the Asymmmetric Noise CIFAR-10 dataset (noise rate = 0.4):

    $ python first_step_train.py --gpu 0 --out first_an04 --learnrate 0.03 --alpha 0.8 --beta 0.4 --percent 0.4 --asym
    $ python second_step_train.py --gpu 0 --out second_an04 --label first_an04

## Citation
    @inproceedings{tanaka2018joint,
        title = {Joint Optimization Framework for Learning with Noisy Labels},
        author = {Tanaka, Daiki and Ikami, Daiki and Yamasaki, Toshihiko and Aizawa, Kiyoharu},
        booktitle = {CVPR},
        year = {2018}
    }
