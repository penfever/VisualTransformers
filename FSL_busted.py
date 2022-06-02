import torch
from torch import nn, optim
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.models import resnet50
from tqdm import tqdm

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden)
        self.input_sublayer = SublayerConnection(size=hidden)
        self.output_sublayer = SublayerConnection(size=hidden)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return x

class VisEmbed(nn.Module):
    """
    Sequence-wise embedding
    """

    def __init__(self, backbone: nn.Module, embed_size):
        """
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.backbone = backbone
        self.embed_size = embed_size

    def forward(self, support_images: torch.Tensor, support_labels: torch.Tensor):
        support_images = support_images.cuda()
        support_labels = support_labels.cuda()
        embedding = self.backbone(support_images)
        embed_full = embedding[0]
        embed_full = torch.cat((embed_full, support_labels[0].unsqueeze(0)))
        for i in range(1, len(embedding)):
          embed_full = torch.cat((embed_full, embedding[i]))
          if i != len(embedding) - 1:
            embed_full = torch.cat((embed_full, support_labels[i].unsqueeze(0)))
        return embed_full

class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """
    def __init__(self, backbone: nn.Module, hidden, n_layers, attn_heads, classes):
        """
        :param hidden: BERT model hidden size (64x9 images + 1x8 labels)
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        """

        super().__init__()
        self.backbone = backbone
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.classes = classes
        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding (image/label sequence)
        # self.embedding = VisEmbed(backbone=backbone, embed_size=hidden)
        self.embedding = backbone

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, self.feed_forward_hidden) for _ in range(n_layers)])
        self.linear1 = torch.nn.Linear(self.hidden, self.classes)
        self.linear2 = torch.nn.Linear(self.feed_forward_hidden, self.classes)
        self.logsoftmax = torch.nn.LogSoftmax()

    def forward(self, support_images: torch.Tensor, support_labels: torch.Tensor) -> torch.Tensor:
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        # embedding the indexed sequence to sequence of vectors
        # print("pre-embedding")
        # x = self.embedding(support_images=support_images, support_labels=support_labels)
        x = self.embedding(support_images.cuda())
        # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        mask = None
        # running over multiple transformer blocks
        # for transformer in self.transformer_blocks:
        #    x = transformer.forward(x, mask)

        x = self.linear1(x)
        # x = self.linear2(x)
        x = self.logsoftmax(x)
        return x

DL_PATH = "/data/bf996/omniglot_merge/"
BS = 1
IMAGE_SIZE = 28
LR = 0.01

convolutional_network = resnet50(pretrained=True)
convolutional_network.fc = nn.Linear(2048,64)

transform = torchvision.transforms.Compose(
     [
     torchvision.transforms.Grayscale(num_output_channels=3),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

omniglot = torchvision.datasets.ImageFolder(root=DL_PATH, transform=transform)
labels = torch.unique(torch.tensor(omniglot.targets))
NUM_CLASSES = len(labels)
train_set_size = int(len(omniglot) * 0.7)
valid_set_size = len(omniglot) - train_set_size
train_dataset, test_dataset = torch.utils.data.random_split(omniglot, [train_set_size, valid_set_size])

full_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=BS,
                                          shuffle=True)

full_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=BS,
                                         shuffle=False)

criterion = nn.NLLLoss()
net = BERT(backbone=convolutional_network, attn_heads=8, classes=NUM_CLASSES, hidden=64, n_layers=12).cuda()
trainable_parameters = []
for param in net.named_parameters():
  assert param[1].requires_grad  # finetune all LM parameters
  trainable_parameters.append(param[1])

optimizer = optim.AdamW(params=trainable_parameters, lr=LR)
optimizer.zero_grad()

def train(epochs=10):
  for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(full_loader_train, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if len(labels) < BS:
          continue
        labels = labels.cuda()
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(support_images=inputs, support_labels=labels)
        # print(outputs.size(), labels.size())
        # print(torch.argmax(outputs), labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0
    net.eval()
    with torch.no_grad():
        total_samples = 0
        correct_samples = 0
        for i, data in enumerate(full_loader_test, 0):
            inputs, labels = data
            if len(labels) < BS:
                continue
            total_samples += 1
            labels = labels.cuda()
            inputs = inputs.cuda()
            outputs = net(support_images=inputs, support_labels=labels)
            pred = torch.argmax(outputs)
            if pred == labels[-1]:
                correct_samples += 1
        print('  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
            '{:5}'.format(total_samples) + ' (' +
            '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')

if __name__ == '__main__':
    train(25)