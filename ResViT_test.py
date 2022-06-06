
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 11:37:52 2020
@author: mthossain
"""
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

decoder_layer = nn.TransformerDecoderLayer(d_model=64, nhead=8, dropout=.1)
transformer = nn.TransformerDecoder(decoder_layer, num_layers=12)
dec_test = nn.ModuleList(nn.Sequential(*list(decoder_layer.children())[:5]))
print(dec_test)
print(decoder_layer)
# for layer in torch.nn.Sequential(*list(decoder_layer.children())):
#     print(layer)