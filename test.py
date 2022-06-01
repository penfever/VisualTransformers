import PIL
import time
import timm
import torch
from torchsummary import summary
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init

if __name__ == '__main__':
    BS = 9
    DL_PATH = "/data/bf996/omniglot_merge/" # Use your own path

    transform = torchvision.transforms.Compose(
        [
        torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


    omniglot = torchvision.datasets.ImageFolder(root=DL_PATH, transform=transform)

    train_set_size = int(len(omniglot) * 0.7)
    valid_set_size = len(omniglot) - train_set_size
    train_dataset, test_dataset = torch.utils.data.random_split(omniglot, [train_set_size, valid_set_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BS,
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BS,
                                            shuffle=False)
    labels = torch.unique(torch.tensor(omniglot.targets))
    print(labels, len(labels))