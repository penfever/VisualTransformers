import PIL
import time
import timm
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init

#dim -> embedding dimension

class seqTrans(nn.Module):
    def __init__(self, conv_model, num_classes=10, dim = 64, num_tokens = 64, mlp_dim = 256, heads = 8, depth = 12, emb_dropout = 0.1, dropout= 0.1):
        super(seqTrans, self).__init__()
        self.dim = dim
        self.conv_model = conv_model
        self.in_planes = 64 #controls how many channels the model expects
        self.label_embed = torch.nn.Embedding(num_classes, dim)
        self.num_classes = num_classes
        self.apply(_weights_init)
        
        self.pos_embedding = nn.Parameter(torch.empty(num_tokens * 2, dim))        
        torch.nn.init.normal_(self.pos_embedding, std = .02) # Initialize to normal distribution. Based on the paper
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()  #TODO: consider using linear, tanh for this a la BERT
        self.nn1 = nn.Linear(dim * num_tokens * 2, self.num_classes)  # if finetuning, just use a linear layer without further hidden layers (paper)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        
    def forward(self, seq_x, seq_y, mask = None):
        x = conv_model(seq_x)
        x += self.pos_embedding
        if len(x[0]) > 1:
            seq_y[-1] = self.num_classes-1 #CLS_TOKEN
            y = self.label_embed(seq_y)
            y = y.view(len(seq_y), self.dim, self.in_planes)
            x = build_seq(x, y)
        x = self.dropout(x)
        x = self.transformer(x, mask)
        x = self.to_cls_token(x[:, 0])
        x = x.flatten()
        x = self.nn1(x)
        return x

conv_model = timm.create_model('resnet50', pretrained=True)
conv_model.fc = torch.nn.Linear(2048, 64)