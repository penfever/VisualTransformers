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

def build_seq(image_vec, label_vec):
    embed_full = image_vec[0]
    embed_full = torch.stack((embed_full, label_vec[0]))
    for i in range(1, len(image_vec)):
        embed_full = torch.cat((embed_full, image_vec[i].unsqueeze(0)))
        embed_full = torch.cat((embed_full, label_vec[i].unsqueeze(0)))
    return embed_full

class seqTrans(nn.Module):
    def __init__(self, conv_model, label_embedding, all_labels, num_classes=10, dim = 64, num_tokens = 64, mlp_dim = 256, heads = 8, depth = 12, emb_dropout = 0.1, dropout= 0.1):
        super(seqTrans, self).__init__()
        self.dim = dim
        self.label_embed = label_embedding
        self.all_labels = all_labels
        self.num_tokens = num_tokens
        self.n_seq = BATCH_SIZE_TRAIN // self.num_tokens
        self.conv_model = conv_model
        self.in_planes = 64 #controls how many channels the model expects
        self.num_classes = num_classes
        self.apply(_weights_init)
        self.positional_encoder = PositionalEncoding(
            dim_model=dim, dropout_p=dropout, max_len=5000
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()  #TODO: consider using linear, tanh for this a la BERT
        self.nn1 = nn.Linear(dim, self.num_classes)  # if finetuning, just use a linear layer without further hidden layers (paper)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        
    def forward(self, seq_x, seq_y, mask = None):
        x = conv_model(seq_x.view(BATCH_SIZE_TRAIN, 3, 105, 105))
        dup_x = x
        idx = [self.num_tokens * i + self.num_tokens - 1 for i in range(self.n_seq)]
        seq_y = seq_y.reshape(BATCH_SIZE_TRAIN)
        seq_y[idx] = self.num_classes-1 #CLS_TOKEN
        y = self.label_embed(seq_y) * (self.dim ** -0.5)
        if self.num_tokens > 1:
            x = build_seq(x, y)
            # print(x.size())
            # x = rearrange(x, 'b c h w -> b (h w) c') # nXn convolution output reshaped to [batch_size, (n^2), c]
            x = x.reshape(self.n_seq, self.num_tokens * 2, self.dim)
            #TESTS
            assert(torch.equal(x[0, 0, :], dup_x[0, :]))
            assert(torch.equal(x[0, 2, :], dup_x[1, :]))
            assert(torch.equal(x[0, 1, :], y[0, :]))
            assert(torch.equal(x[0, 3, :], y[1, :]))
        else:
            x = x.unsqueeze(dim=1)
        x = self.positional_encoder(x)
        x = self.transformer(x, mask)
        x = self.to_cls_token(x[:, -1])
        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(x, self.all_labels)
        return dists

BATCH_SIZE_TRAIN = BATCH_SIZE_TEST = 64
N_TOKENS = 2
DL_PATH = "/data/bf996/omniglot_merge/" # Use your own path
SUBSET_SIZE = 100
MODEL_DIM = 256
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
LR = .002 if DATASET == subset else .0005
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
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST,
                                         shuffle=False)

def train(model, optimizer, criterion, data_loader, loss_history, scheduler=None):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=1000)
    example_ct = 0  # number of examples seen
    total_samples = len(data_loader.dataset)//N_TOKENS
    model.train()
    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        if len(target) < BATCH_SIZE_TRAIN:
          continue
        data = data.cuda()
        data_s = data.reshape(BATCH_SIZE_TRAIN//N_TOKENS, N_TOKENS, 3, 105, 105)
        assert(torch.equal(data, data_s.view(BATCH_SIZE_TRAIN, 3, 105, 105)))
        data = data_s
        target = target.cuda()
        target = target.reshape(BATCH_SIZE_TRAIN//N_TOKENS, N_TOKENS)
        final_idx = [N_TOKENS-1 for i in range(BATCH_SIZE_TRAIN//N_TOKENS)]
        ids = torch.Tensor(final_idx).long().cuda()
        true_target = target.gather(1, ids.view(-1,1)).clone()
        output = F.log_softmax(model(data, target), dim=1)
        loss = criterion(output, true_target.squeeze(dim=1))
        example_ct += len(data)//N_TOKENS
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if i % 5 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                '{:6.4f}'.format(loss.item()))
            train_log(output, loss, epoch, example_ct)

def train_log(output, loss, epoch, example_ct):
    # Where the magic happens
    min_logprob = torch.min(output).item()
    max_logprob = torch.max(output).item()
    avg_logprob = torch.mean(output).item()
    wandb.log({"min_logprob": min_logprob, "max_logprob": max_logprob, "avg_logprob": avg_logprob, "epoch": epoch, "avg_train_loss": loss}, step=example_ct)
    print("LOGSOFT: min = {:1.3f}, max = {:1.3f}, mean = {:1.3f} ".format(min_logprob, max_logprob, avg_logprob))

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

def evaluate(model, data_loader, loss_history, criterion):
    model.eval()    
    total_samples = len(data_loader.dataset)//N_TOKENS
    topk_samples = []
    total_loss = 0
    correct_samples = 0
    with torch.no_grad():
        for data, target in data_loader:
            if len(target) < BATCH_SIZE_TRAIN:
                continue
            data = data.cuda()
            data = data.reshape(BATCH_SIZE_TRAIN//N_TOKENS, N_TOKENS, 3, 105, 105)
            target = target.cuda()
            target = target.reshape(BATCH_SIZE_TRAIN//N_TOKENS, N_TOKENS)
            final_idx = [N_TOKENS-1 for i in range(BATCH_SIZE_TRAIN//N_TOKENS)]
            ids = torch.Tensor(final_idx).long().cuda()
            true_target = target.gather(1, ids.view(-1,1)).clone()
            output = F.log_softmax(model(data, target), dim=1)
            loss = criterion(output, true_target.squeeze(dim=1), reduction='sum')
            _, pred = torch.max(output, dim=1)
            total_loss += loss.item()
            correct_samples += pred.eq(true_target.squeeze(dim=1)).sum()
            topk_samples.append(accuracy(output, true_target.squeeze(dim=1), 5))
    avg_loss = total_loss / total_samples
    wandb.log({"avg_test_loss": avg_loss, "top1_test_accuracy": 100.0 * correct_samples / total_samples, "top5_test_accuracy": 100 * torch.mean(torch.tensor(topk_samples))})
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) + '\n' +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n' +
          'Top 5 Accuracy: ' + '{:.2f}%\n'.format(100 * torch.mean(torch.tensor(topk_samples))))

N_EPOCHS = 100 + (NUM_DATASET_CLASSES // NUM_CLASSES) #Need more epochs for smaller subsets

config = dict(
    epochs=N_EPOCHS,
    classes=NUM_CLASSES,
    batch_size=BATCH_SIZE_TEST,
    learning_rate=LR,
    model_dim=MODEL_DIM,
    seq_len=N_TOKENS,
    dataset="Omniglot",
    architecture="RN50-flatRepTransNoSeq")

with wandb.init(project="RN50-SeqTrans-Omniglot", config=config):
    conv_model = timm.create_model('resnet50', pretrained=True)
    conv_model.fc = torch.nn.Linear(2048, MODEL_DIM)

    label_embed = torch.nn.Embedding(NUM_CLASSES, MODEL_DIM).cuda()
    num_tensor = torch.tensor([i for i in range(NUM_CLASSES)]).cuda().detach()
    all_labels = label_embed(num_tensor).cuda().detach()

    model = seqTrans(conv_model=conv_model, label_embedding = label_embed, all_labels = all_labels, dim=MODEL_DIM, num_classes=NUM_CLASSES, num_tokens=N_TOKENS).cuda()
    criterion = F.nll_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    train_loss_history, test_loss_history = [], []
    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch)
        start_time = time.time()
        train(model, optimizer, criterion, train_loader, train_loss_history)
        print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
        evaluate(model, test_loader, test_loss_history, criterion)

    print('Execution time')

PATH = "./ViTRes.pt" # Use your own path
torch.save(model.state_dict(), PATH)

# =============================================================================
# model = ViT()
# model.load_state_dict(torch.load(PATH))
# model.eval()            
# =============================================================================