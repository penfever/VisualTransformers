
import PIL
import time
import timm
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
import math
from torch.nn import MultiheadAttention
import wandb

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

"""
NOTE:
For simple image classification, the transformer appears to degrade performance, at least judging by test set accuracy.

for d in range(self.depth):
    identity = x
    x = self.attn(x, x, x)
    x = self.layer_norm(x[0]) + identity
    identity = x
    x = self.nn1(x)
    x = self.af1(x)
    x = self.do1(x)
    x = self.nn2(x)
    x = self.do2(x)
    x = self.layer_norm(x) + identity

performs worse than

    identity = x
    x = self.attn(x, x, x)
    x = self.layer_norm(x[0]) + identity

performs worse than

    return x
"""

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.attn = MultiheadAttention(embed_dim=dim, num_heads = heads, dropout = dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(dim)
        self.nn1 = nn.Linear(dim, mlp_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        self.af1 = nn.GELU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(mlp_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std = 1e-6)
        self.do2 = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        identity = x
        x = self.attn(x, x, x)
        x = self.layer_norm(x[0]) + identity
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class seqTrans(nn.Module):
    def __init__(self, conv_model, num_classes=10, dim = 64, num_tokens = 64, mlp_dim = 256, heads = 8, depth = 12, emb_dropout = 0.1, dropout= 0.1):
        super(seqTrans, self).__init__()
        self.dim = dim
        self.depth = depth
        self.conv_model = conv_model
        self.in_planes = 64 #controls how many channels the model expects
        self.num_classes = num_classes
        self.apply(_weights_init)   
        self.positional_encoder = PositionalEncoding(
            dim_model=dim, dropout_p=dropout, max_len=5000
        )
        self.dropout = nn.Dropout(emb_dropout)
        # self.transformer = nn.Transformer(d_model=dim, batch_first=True, norm_first=True)
        # self.transformer_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=heads)
        # self.transformer = nn.TransfomerDecoder(d_model=dim, batch_first=True, norm_first=True)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, self.num_classes)  # if finetuning, just use a linear layer without further hidden layers (paper)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        
    def forward(self, img, mask = None):
        # print(img.size())
        x = conv_model(img)
        x = x.unsqueeze(dim=1)
        # print(x.size())
        #x = rearrange(x, 'b c h w -> b c (h w)')
        x = self.positional_encoder(x)
        x = self.transformer(x, mask)
        # print(x.size())
        x = self.to_cls_token(x[:, 0]) 
        x = self.nn1(x)
        return x

BATCH_SIZE_TRAIN = BATCH_SIZE_TEST = 32
N_TOKENS = 1
DL_PATH = "/data/bf996/omniglot_merge/" # Use your own path
SUBSET_SIZE = 1000
MODEL_DIM = 128

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
LR = .001 if DATASET == subset else .0006
labels = torch.unique(torch.tensor(omniglot.targets))
NUM_DATASET_CLASSES = len(labels)
if DATASET == subset:
    labels = torch.tensor([i for i in range(SUBSET_SIZE)])
NUM_CLASSES = len(labels)+1 #add 1 for CLS token
print("num classes is {}".format(NUM_CLASSES))
train_set_size = int(len(DATASET) * 0.8)
valid_set_size = len(DATASET) - train_set_size
train_dataset, test_dataset = torch.utils.data.random_split(DATASET, [train_set_size, valid_set_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN,
                                          shuffle=True, num_workers=8)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST,
                                         shuffle=False, num_workers=8)

def train_log(output, loss, epoch, example_ct):
    # Where the magic happens
    pass

def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.
    """
    with torch.no_grad():
        wandb.watch(model, criterion, log="all", log_freq=1000)

        maxk = topk
        batch_size = target.size(0)

        # st()
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # st()
        correct = (pred == target.unsqueeze(dim=0)).expand_as(pred)

        res = []
        correct_k = correct[:maxk].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
        return res

def train(model, optimizer, criterion, data_loader, loss_history, scheduler=None):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10000)
    example_ct = 0  # number of examples seen
    total_samples = len(data_loader.dataset)
    model.train()
    for i, (data, target) in enumerate(data_loader):
        if len(target) < BATCH_SIZE_TRAIN:
          continue
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = criterion(output, target)
        example_ct += len(data)//N_TOKENS
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if i % 10 == 0:
            train_log(output, loss, epoch, example_ct)
            min_logprob = torch.min(output).item()
            max_logprob = torch.max(output).item()
            avg_logprob = torch.mean(output).item()
            wandb.log({"min_logprob": min_logprob, "max_logprob": max_logprob, "avg_logprob": avg_logprob, "epoch": epoch, "avg_train_loss": loss})
            print("LOGSOFT: min = {:1.3f}, max = {:1.3f}, mean = {:1.3f} ".format(min_logprob, max_logprob, avg_logprob))
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                '{:6.4f}'.format(loss.item()))
            
def evaluate(model, data_loader, loss_history, criterion):
    model.eval()    
    total_samples = len(data_loader.dataset)
    topk_samples = []
    total_loss = 0
    correct_samples = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.cuda()
            target = target.cuda()
            if len(target) < BATCH_SIZE_TRAIN:
              continue
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()
            topk_samples.append(accuracy(output, target, 5))
    avg_loss = total_loss / total_samples
    top1_acc = 100.0 * (correct_samples / total_samples)
    top5_acc = 100 * torch.mean(torch.tensor(topk_samples))
    wandb.log({"avg_loss": avg_loss, "top1_test_accuracy": top1_acc, "top5_test_accuracy": top5_acc})
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) + '\n' +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(top1_acc) + '%)\n' +
          'Top 5 Accuracy: ' + '{:.2f}%\n'.format(top5_acc))

N_EPOCHS = 10 + (NUM_DATASET_CLASSES // NUM_CLASSES) #Need more epochs for smaller subsets

config = dict(
    epochs=N_EPOCHS,
    classes=NUM_CLASSES,
    batch_size=BATCH_SIZE_TEST,
    learning_rate=LR,
    model_dim=MODEL_DIM,
    dataset="Omniglot",
    architecture="RN50-flatRepTransNoSeq")

with wandb.init(project="RN50-flatRepTransNoSeq-Omniglot", config=config):
    conv_model = timm.create_model('resnet50', pretrained=True)
    conv_model.fc = torch.nn.Linear(2048, MODEL_DIM)
    model = seqTrans(conv_model=conv_model, dim=MODEL_DIM, num_classes=NUM_CLASSES).cuda()
    criterion = F.nll_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    train_loss_history, test_loss_history = [], []
    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch)
        start_time = time.time()
        train(model, optimizer, criterion, train_loader, train_loss_history)
        print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
        evaluate(model, test_loader, test_loss_history, criterion)

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