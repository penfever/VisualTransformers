# import PIL
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
import math
import time
from datetime import datetime, date
import logging
from shutil import copyfile
import pathlib
from pathlib import Path
import numpy as np
import os
# import timm
import torch
import torchvision
import torch.nn.functional as F
# from einops import rearrange
from torch import nn
import torch.nn.init as init
# from torch.nn import MultiheadAttention
from transformers.optimization import get_cosine_schedule_with_warmup
from model.transformer import Transformer as VAS_Transformer
import wandb

# FUNCTIONS AND CLASS DEFINITIONS

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

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
    def __init__(self, label_embedding, all_labels, num_classes=10, dim = 64, mlp_dim = 256, heads = 8, depth = 12, emb_dropout = 0.1, dropout= 0.1):
        super(seqTrans, self).__init__()
        self.dim = dim
        self.label_embed = label_embedding
        self.all_labels = all_labels
        self.conv_model = torchvision.models.resnet18()
        # self.conv_model = timm.create_model('resnet18', pretrasined=False)
        # self.conv1_weight = self.conv_model.state_dict()['conv1.weight']
        # self.conv_model.state_dict()['conv1.weight'] = self.conv1_weight.sum(dim=1, keepdim=True)
        self.conv_model.fc = torch.nn.Linear(512, MODEL_DIM)
        self.num_classes = num_classes
        self.apply(_weights_init)
        self.positional_encoder = PositionalEncoding(
            dim_model=dim, dropout_p=dropout, max_len=5000
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = VAS_Transformer(num_classes, num_classes, n_layers=12, hidden_size=dim, dropout_rate=dropout, src_pad_idx=0, trg_pad_idx=0)
        self.to_cls_token = nn.Identity()

    def build_seq(self, image_vec, label_vec):
        embed_full = image_vec[0]
        embed_full = torch.stack((embed_full, label_vec[0]))
        for i in range(1, len(image_vec)):
            embed_full = torch.cat((embed_full, image_vec[i].unsqueeze(0)))
            embed_full = torch.cat((embed_full, label_vec[i].unsqueeze(0)))
        return embed_full

    def forward(self, seq_x, seq_y, bs, nt, mask = None):
        # NOTE: bs expects to be divisible by 2
        n_seq = bs // nt
        x = self.conv_model(seq_x.view(bs, 3, 105, 105))
        dup_x = x
        idx = [nt * i + nt - 1 for i in range(n_seq)]
        seq_y = seq_y.reshape(bs)
        seq_y[idx] = self.num_classes-1 #CLS_TOKEN
        y = self.label_embed(seq_y) * (self.dim ** -0.5)
        if nt > 1:
            # print("x is {}, y is {}".format(x.size(), y.size()))
            x = self.build_seq(x, y)
            x = x.reshape(n_seq, nt * 2, self.dim)
            #TESTS
            assert(torch.equal(x[0, 0, :], dup_x[0, :]))
            assert(torch.equal(x[0, 2, :], dup_x[1, :]))
            assert(torch.equal(x[0, 1, :], y[0, :]))
            assert(torch.equal(x[0, 3, :], y[1, :]))
        else:
            x = x.unsqueeze(dim=1)
        x = self.positional_encoder(x)
        x = self.transformer(x, x)
        x = self.to_cls_token(x[:, -1])
        return x

def train(model, optimizer, criterion, data_loader, epoch, scheduler=None):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    #wandb.watch(model, criterion, log="all", log_freq=25000)
    example_ct = 0  # number of examples seen
    total_samples = len(data_loader.dataset)//N_TOKENS
    model.train()
    logging.debug("LR: {:.6f}".format(optimizer.param_groups[0]['lr']))
    losses = []
    min_logprobs = []
    max_logprobs = []
    avg_logprobs = []
    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        data = data.to(device=device)
        data_s = data.reshape(BATCH_SIZE_TRAIN//N_TOKENS, N_TOKENS, 3, 105, 105)
        assert(torch.equal(data, data_s.view(BATCH_SIZE_TRAIN, 3, 105, 105)))
        data = data_s
        target = target.to(device=device)
        target = target.reshape(BATCH_SIZE_TRAIN//N_TOKENS, N_TOKENS)
        final_idx = [N_TOKENS-1 for i in range(BATCH_SIZE_TRAIN//N_TOKENS)]
        ids = torch.Tensor(final_idx).long().to(device=device)
        true_target = target.gather(1, ids.view(-1,1)).clone().squeeze(dim=1)
        output = model(data, target, BATCH_SIZE_GPU, N_TOKENS).squeeze(dim=1)
        loss = criterion(output, true_target)
        losses.append(loss.cpu())
        example_ct += len(data)//N_TOKENS
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        min_logprob = torch.min(output).cpu().item()
        min_logprobs.append(min_logprob)
        max_logprob = torch.max(output).cpu().item()
        max_logprobs.append(max_logprob)
        avg_logprob = torch.mean(output).cpu().item()
        avg_logprobs.append(avg_logprob)
    avg_train_loss = torch.mean(torch.tensor(losses))
    avg_train_min = torch.mean(torch.tensor(min_logprobs))
    avg_train_max = torch.mean(torch.tensor(max_logprobs))
    avg_train_avg = torch.mean(torch.tensor(avg_logprobs))
    if epoch % 50 == 1:
        wandb.log({"min_logprob": avg_train_min, "max_logprob": avg_train_max, "avg_logprob": avg_train_avg, "epoch": epoch, "avg_train_loss": avg_train_loss, "lr": optimizer.param_groups[0]['lr']})
    return

def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.
    """
    with torch.no_grad():
        maxk = topk
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = (pred == target.unsqueeze(dim=0)).expand_as(pred)
        res = []
        correct_k = correct[:maxk].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
        return res

def evaluate(model, data_loader, loss_history, criterion, scheduler):
    model.eval()    
    total_samples = 0
    topk_samples = []
    total_loss = 0
    correct_samples = 0
    with torch.no_grad():
        for data, target in data_loader:
            if len(target) < BATCH_SIZE_TRAIN:
                continue
            total_samples += (len(data) // N_TOKENS)
            data = accelerator.gather(data)
            data = data.reshape(BATCH_SIZE_TRAIN//N_TOKENS, N_TOKENS, 3, 105, 105)
            target = accelerator.gather(target)
            true_target = target[-1].unsqueeze(dim=0).clone()
            target = target.reshape(BATCH_SIZE_TRAIN//N_TOKENS, N_TOKENS)
            final_idx = [N_TOKENS-1 for i in range(BATCH_SIZE_TRAIN//N_TOKENS)]
            ids = torch.Tensor(final_idx).long().to(device=device)
            true_target = target.gather(1, ids.view(-1,1)).clone()
            output = F.log_softmax(model(data, target, BATCH_SIZE_GPU, N_TOKENS), dim=1)
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

def fsl_eval(model, data_loader, criterion, scheduler):
    #TODO: check if sequence should be balanced between classes
    model.eval()  
    total_samples = 0
    correct_samples = 0
    with torch.no_grad():
        for data, target in data_loader:
            if len(target) < N_TOKENS:
                continue
            total_samples += (len(data) // N_TOKENS)
            data = accelerator.gather(data)
            data = data.reshape(1, N_TOKENS, 3, 105, 105)
            target = accelerator.gather(target)
            true_target = target[-1].unsqueeze(dim=0).clone()
            target = target.reshape(1, N_TOKENS)

            final_idx = [N_TOKENS-1 for i in range(N_TOKENS)]
            ids = torch.Tensor(final_idx).long()
            output = F.log_softmax(model(data, target, N_TOKENS, N_TOKENS), dim=1)
            loss = criterion(output, true_target, reduction='sum')
            _, pred = torch.max(output[:, :2], dim=1)
            correct_samples += pred.eq(true_target.flatten()).sum()
    # if scheduler is not None:
    #     scheduler.step(avg_loss)
    wandb.log({"top1_fsl_accuracy": 100.0 * correct_samples / total_samples})
    print('\n fsl Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')
    # device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

def burst_loader(ds, nc):
    weights = np.array([float((1/3)/(nc-2)) for i in range(nc)])
    burst_indices = np.random.randint(nc, size=2)
    weights[burst_indices] = np.array([float(1/3)])
    if not np.isclose(np.sum(weights), 1):
        logging.debug("Weight imbalance: ")
        logging.debug(np.abs(np.sum(weights) - 1))
    train_sampler = torch.utils.data.WeightedRandomSampler(weights, SAMPLING_SIZE, replacement=True)
    train_loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE_TRAIN, num_workers=DEV_CT*8, sampler=train_sampler)
    return train_loader

# LOGGING

# Set logger

# logging.basicConfig(level=logging.CRITICAL)

# HYPERPARAMETERS

LR = 3e-4
kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=kwargs)
device = accelerator.device
# device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
BURSTY = float(0.9)
#TODO: when N_TOKENS = 9, training fails to converge and LR seems off
N_TOKENS = 8
# SCALER increases the GPU and CPU load by scaling batch sizes
SCALER = 40
DEV_CT = max(1, int(torch.cuda.device_count()))
BATCH_SIZE_GPU = BATCH_SIZE_TRAIN = BATCH_SIZE_TEST = SCALER * N_TOKENS * DEV_CT
# DISTRIBUTED = True if DEV_CT > 1 else False
# print("distributed: {}".format(DISTRIBUTED))
# if DISTRIBUTED:
#     BATCH_SIZE_GPU = BATCH_SIZE_TEST // DEV_CT
#     BS_FSL = N_TOKENS // DEV_CT
# else:
BS_FSL = N_TOKENS
SAMPLING_SIZE = SCALER * BATCH_SIZE_GPU * DEV_CT
DL_PATH = "/data/bf996/omniglot_merge/" # Use your own path
HOLDOUT_PATH = "/data/bf996/omniglot_merge_holdout/" # Use your own path
SUBSET_SIZE = 100
MODEL_DIM = 128
N_EPOCHS = 10000
START = 1
SAVE_FREQ = 50
WARMUP = 4000
TOTAL_STEPS = 10000
WORKERS = min(os.cpu_count() * 2, DEV_CT * 8)

# TRANSFORMS

transform = torchvision.transforms.Compose(
    [
    torchvision.transforms.Grayscale(num_output_channels=3),
    torchvision.transforms.ToTensor(),
    # TODO: get mean, stdev of omniglot
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

# MAIN DATASET

print("begin loading data: ")

omniglot = torchvision.datasets.ImageFolder(root=DL_PATH, transform=transform)

# SUBSET HANDLING

if SUBSET_SIZE > 0:
    main_idx = [i for i in range(len(omniglot)) if omniglot.imgs[:][i][1] < SUBSET_SIZE]
# build the appropriate subset
    subset = torch.utils.data.Subset(omniglot, main_idx)
    DATASET = subset
else:
    DATASET = omniglot
    subset = None
labels = torch.unique(torch.tensor(omniglot.targets))
NUM_DATASET_CLASSES = len(labels)
if DATASET == subset:
    labels = torch.tensor([i for i in range(SUBSET_SIZE)])
NUM_CLASSES = len(labels)+1 #add 1 for CLS token

# CONFIG DICT FOR WANDB

config = dict(
    epochs=N_EPOCHS,
    classes=NUM_CLASSES,
    batch_size=BATCH_SIZE_TEST,
    learning_rate=LR,
    model_dim=MODEL_DIM,
    seq_len=N_TOKENS,
    dataset="Omniglot",
    architecture="RN18-SeqTrans128")

# TRANSFORMS
with wandb.init(project="{}-{}-burst{}".format(config['architecture'], config['dataset'], BURSTY), config=config, group="DDP"):

    # HOLDOUT DATASET FOR FSL
    omniglot_holdout = torchvision.datasets.ImageFolder(root=HOLDOUT_PATH, transform=transform)
    holdout_loader = accelerator.prepare(torch.utils.data.DataLoader(omniglot_holdout, batch_size=N_TOKENS, num_workers=WORKERS, sampler=torch.utils.data.RandomSampler(omniglot_holdout, replacement=True, num_samples=SAMPLING_SIZE), drop_last=True))

    print("num classes is {}".format(NUM_CLASSES))

    train_set_size = int(len(DATASET) * 0.8)
    valid_set_size = len(DATASET) - train_set_size
    train_dataset, test_dataset = torch.utils.data.random_split(DATASET, [train_set_size, valid_set_size])

    train_loader = accelerator.prepare(torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN,
                                            shuffle=True, num_workers=WORKERS, drop_last=True))
    test_sampler = torch.utils.data.RandomSampler(test_dataset, replacement=True, num_samples=SAMPLING_SIZE)
    test_loader = accelerator.prepare(torch.utils.data.DataLoader(omniglot, batch_size=BATCH_SIZE_TEST,
                                            shuffle=False, num_workers=WORKERS, sampler=test_sampler, drop_last=True))

    TOTAL_SAMPLES = len(train_dataset)//N_TOKENS
    print("total samples in training: {}".format(TOTAL_SAMPLES))
    NUM_TRAINING_STEPS = TOTAL_SAMPLES // BATCH_SIZE_TEST * N_EPOCHS
    HOLD_RATE_STEPS = 10000


    # MAIN LOOP
    LATEST = "./ViTRes_Latest.pt"
    label_embed = torch.nn.Embedding(NUM_CLASSES, MODEL_DIM).to(device=device)
    num_tensor = torch.tensor([i for i in range(NUM_CLASSES)]).to(device=device).detach()
    all_labels = label_embed(num_tensor).to(device=device).detach()
    model = accelerator.prepare(seqTrans(label_embedding = label_embed, all_labels = all_labels, dim=MODEL_DIM, num_classes=NUM_CLASSES).to(device=device))
    EPOCH_RES = 0
    if Path(LATEST).exists():
        print("Loading latest pretrained weights from {}".format(LATEST))
        try:
            model.load_state_dict(torch.load(LATEST))
        except:
            print("\n Try loading using dataparallel")          
        epoch_file = list(Path(".").glob('*.epoch'))
        START = EPOCH_RES = int(epoch_file[0].stem)
        print("Resuming from epoch {}".format(EPOCH_RES))

    criterion = F.cross_entropy
    optimizer = accelerator.prepare(torch.optim.AdamW(model.parameters(), lr=LR))
    scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=WARMUP, 
    num_training_steps=TOTAL_STEPS)
    for i in range(EPOCH_RES):
        scheduler.step()
    train_loss_history, test_loss_history = [], []
    burst_count = 0
    start_time = time.time()
    for epoch in range(START, N_EPOCHS + START):
        if epoch % 10 == 1:
            print('Epoch: {}'.format(epoch))
        pr = np.random.random_sample(size=1)
        if pr[0] < BURSTY:
            burst_count += 1
            train_loader = burst_loader(train_dataset, NUM_CLASSES)
        train(model, optimizer, criterion, train_loader, epoch, scheduler)
        if epoch % SAVE_FREQ == 1 and epoch != START:
            print('Avg execution time: {:5.2f} seconds'.format((time.time() - start_time) / max(1, SAVE_FREQ)))
            start_time = time.time()
            evaluate(model, test_loader, test_loss_history, criterion, scheduler)
            fsl_eval(model, holdout_loader, criterion, scheduler)
            print("saving model weights for epoch {}".format(epoch))
            now = datetime.now()
            today = date.today()
            current_time = now.strftime("%H-%M-%S")
            d4 = today.strftime("%b-%d-%Y")
            strepoch = str(epoch)
            PATH = Path("./ViTRes_{}_{}_burst_{}_epoch_{}.pt".format(d4, current_time, str(int(100*BURSTY)), strepoch)) # Use your own path
            accelerator.save(accelerator.unwrap_model(model).state_dict(), PATH)
            copyfile(PATH, LATEST)
            LATEST_PARENT = Path(LATEST).parents[0]
            epochs = LATEST_PARENT.glob('*.epoch')
            if epochs:
                for item in epochs:
                    item.unlink()
            NEXT_PATH = Path(pathlib.PurePath(LATEST_PARENT, Path('{}.epoch'.format(strepoch))))
            NEXT_PATH.touch()