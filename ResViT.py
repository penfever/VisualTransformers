import PIL
import time
import timm
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        self.af1 = nn.GELU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hidden_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std = 1e-6)
        self.do2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.do1(x)
        x = self.nn2(x)
        x = self.do2(x)
        
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias = True) # Wq,Wk,Wv for each vector, thats why *3
        torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        torch.nn.init.zeros_(self.to_qkv.bias)
        
        self.nn1 = nn.Linear(dim, dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)        
        self.do1 = nn.Dropout(dropout)
        

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x) #gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h) # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1) #follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v) #product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)') #concat heads into one matrix, ready for next encoder block
        out =  self.nn1(out)
        out = self.do1(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads = heads, dropout = dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attention, mlp in self.layers:
            x = attention(x, mask = mask) # go to attention
            x = mlp(x) #go to MLP_Block
        return x

def build_seq(image_vec, label_vec):
    embed_full = image_vec[0]
    # print(label_vec[0].size())
    embed_full = torch.stack((embed_full, label_vec[0]))
    # print(embed_full.size())
    for i in range(1, len(image_vec)):
        # print(embed_full.size())
        embed_full = torch.cat((embed_full, image_vec[i].unsqueeze(0)))
        embed_full = torch.cat((embed_full, label_vec[i].unsqueeze(0)))
    return embed_full
    # embed_full = image_vec[0, :, :].unsqueeze(0)
    # embed_full = torch.cat((embed_full, label_vec[0].unsqueeze(0)))
    # for i in range(1, len(image_vec)):
    #     embed_full = torch.cat((embed_full, image_vec[i].unsqueeze(0)))
    #     embed_full = torch.cat((embed_full, label_vec[i].unsqueeze(0)))
    # return embed_full

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

        # self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        # self.transformer = torch.nn.Transformer(d_model=dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=heads)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.to_cls_token = nn.Identity()  #TODO: consider using linear, tanh for this a la BERT
        self.nn1 = nn.Linear(dim, self.num_classes)  # if finetuning, just use a linear layer without further hidden layers (paper)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        
    def forward(self, seq_x, seq_y, mask = None):
        x = conv_model(seq_x)
        print(x.size())
        x = rearrange(x, 'b c h w -> b (h w) c') # nXn convolution output reshaped to [batch_size, (n^2), c]
        print(x.size(), self.pos_embedding.size())
        if len(x[0]) > 1:
            seq_y[-1] = self.num_classes-1 #CLS_TOKEN
            y = self.label_embed(seq_y)
            # y = y.view(len(seq_y), self.dim, self.in_planes)
            x = build_seq(x, y)
        # print(x.size(), self.pos_embedding.size())
        x += self.pos_embedding
        x = self.dropout(x)
        # mask = torch.ones_like(torch.tensor([len(x),len(x)])).bool()
        tgt = torch.rand_like(x)
        tgt = self.transformer(tgt, x)
        # print(x.size())
        tgt = self.to_cls_token(tgt[-1, :])
        # x = x.flatten()
        x = self.nn1(tgt)
        x = self.nn1(x)
        return x

BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 32
N_EPOCHS = 10
N_TOKENS = 8
LR = 0.0006
DL_PATH = "/data/bf996/omniglot_merge/" # Use your own path
SUBSET_SIZE = 100
transform = torchvision.transforms.Compose(
     [
     torchvision.transforms.Grayscale(num_output_channels=3),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


omniglot = torchvision.datasets.ImageFolder(root=DL_PATH, transform=transform)
idx = [i for i in range(len(omniglot)) if omniglot.imgs[i][1] < SUBSET_SIZE]
# build the appropriate subset
subset = torch.utils.data.Subset(omniglot, idx)
DATASET = subset
labels = torch.unique(torch.tensor(omniglot.targets))
if DATASET == subset:
    labels = torch.tensor([i for i in range(SUBSET_SIZE)])
NUM_CLASSES = len(labels)
print("num classes is {}".format(NUM_CLASSES))
train_set_size = int(len(DATASET) * 0.7)
valid_set_size = len(DATASET) - train_set_size
train_dataset, test_dataset = torch.utils.data.random_split(DATASET, [train_set_size, valid_set_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST,
                                         shuffle=False)

def train(model, optimizer, data_loader, loss_history, scheduler=None):
    total_samples = len(data_loader.dataset)
    model.train()
    for i, (data, target) in enumerate(data_loader):
        if len(target) < BATCH_SIZE_TRAIN:
          continue
        data = data.cuda()
        # print(data.size())
        data = data.reshape(BATCH_SIZE_TRAIN//N_TOKENS, N_TOKENS, 3, 105, 105)
        target = target.cuda()
        target = target.reshape(BATCH_SIZE_TRAIN//N_TOKENS, N_TOKENS)
        final_idx = [N_TOKENS-1 for i in range(BATCH_SIZE_TRAIN//N_TOKENS)]
        # print(final_idx)
        ids = torch.Tensor(final_idx).long().cuda()
        # print(target, ids)
        true_target = target.gather(1, ids.view(-1,1)).clone()
        # print(target, true_target)
        output = F.log_softmax(model(data, target), dim=1)
        # output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, true_target)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if i % 100 == 0:
            print("LOGSOFT: min = {:1.3f}, max = {:1.3f}, mean = {:1.3f} ".format(torch.min(output).item(), torch.max(output).item(), torch.mean(output).item()))
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())
            
def evaluate(model, data_loader, loss_history):
    model.eval()    
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            data = data.cuda()
            target = target.cuda()
            # seq = data
            if len(target) < BATCH_SIZE_TRAIN:
              continue
            # true_target = target[-1].clone().unsqueeze(0)
            output = F.log_softmax(model(data, target), dim=1)
            # output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target)
            _, pred = torch.max(output, dim=1)
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')

# conv_model = timm.create_model('resnet50', pretrained=True)
# conv_model = torch.nn.Sequential(*list(conv_model.children())[:-3])
# new_out = torch.nn.Conv2d(1024, N_TOKENS, kernel_size=(2,2), stride=(1,1), padding=(1,1), bias=False)
# conv_model = torch.nn.Sequential(*list(conv_model.children())).append(new_out)

conv_model = timm.create_model('resnet50', pretrained=True)
conv_model.fc = torch.nn.Linear(2048, 64)

conv_model_direct = timm.create_model('resnet50', pretrained=True)
conv_model_direct.fc = torch.nn.Linear(2048, NUM_CLASSES)

model = seqTrans(conv_model=conv_model, num_classes=NUM_CLASSES, num_tokens=N_TOKENS).cuda()
# model = conv_model_direct.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=.9)

train_loss_history, test_loss_history = [], []
for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:', epoch)
    start_time = time.time()
    train(model, optimizer, train_loader, train_loss_history)
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
    evaluate(model, test_loader, test_loss_history)

print('Execution time')

PATH = "./ViTRes.pt" # Use your own path
torch.save(model.state_dict(), PATH)

# =============================================================================
# model = ViT()
# model.load_state_dict(torch.load(PATH))
# model.eval()            
# =============================================================================