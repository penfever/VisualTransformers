import torch
import torchvision

DL_PATH = "/data/bf996/omniglot_merge/" # Use your own path
omniglot = torchvision.datasets.ImageFolder(root=DL_PATH, transform=None)
holdout_idx = [i for i in range(0, 100)]
holdout_subset = torch.utils.data.Subset(omniglot, holdout_idx)
print(omniglot.imgs[:][0][1])