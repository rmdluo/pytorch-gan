import torch

def mnist_to_one_hot(digits):
    assert len(digits.shape) == 1
    one_hot = torch.zeros((len(digits), 10))
    one_hot[torch.arange(len(digits)), digits] = 1
    return one_hot