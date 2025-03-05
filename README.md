# Pytorch-GAN

This is an implementation of the papers [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) and [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) in PyTorch, done for personal enrichment. To get started, follow the setup instructions and then the run instructions.

## Setup
1. Install PyTorch from https://pytorch.org/ - Note: they removed conda support, but you can still install it using pip in a conda environment (that's what I did)
2. `pip install tqdm pillow pandas numpy matplotlib torchsummary`

## Run
For GAN:
1. Adjust the hyperparameters near the top of train.py as needed
2. `python train.py`

For Conditional GAN:
1. Adjust the hyperparameters near the top of train_conditional.py as needed
2. `python train_conditional.py`