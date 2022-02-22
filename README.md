# DL-boilderplate

This is the template :heart_eyes: use for deep learning project.

## Folder Structure
- **data**: The input and label dataset.
- **experiments**: For other experiments you want to compare.
- **net**: refer to [pytorch-book](https://github.com/chenyuntc/pytorch-book/blob/master/chapter06-best_practice/PyTorch%E5%AE%9E%E6%88%98%E6%8C%87%E5%8D%97.md).
```angular2html
������ checkpoints/
��   ������ epochs/                 # save checkpoints for every epoch
������ data/
��   ������ __init__.py
��   ������ dataset.py              # encapsulated in the 'DatasetLoad' object
��   ������ preprocess.py           # preprocess data in the first directory
������ models/
��   ������ __init__.py
��   ������ xxNet.py                # one model corresponds to one file
������ runs/                       # output directory of tensorboard
������ utils/
��   ������ start_tensorboard.sh    # running at server side.
������ main.py
������ test.py
������ train.py
```
And thanks to the enthusiastic friends.

## Quick Start

Click "Use this template" button on the top right.