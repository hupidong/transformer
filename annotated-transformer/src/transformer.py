import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn

seaborn.set_context(context="talk")

#1. Embeddings and Softmax
# https://zhuanlan.zhihu.com/p/107889011
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        # d_model=512,词嵌入的维度大小； vocab_size=当前语言的词表大小
        super(Embeddings, self).__init__()
        self.d_model = d_model  # 512
        self.lut = nn.Embedding(vocab_size, d_model)
        # one-hot转词嵌入，这里有一个待训练的矩阵E，大小是vocab_size*d_model

    def forward(self, x):
        # x ~ (batch.size, sequence.length, one-hot),
        # one-hot大小=vocab_size，当前语言的词表大小
        return self.lut(x) * math.sqrt(self.d_model)
        # 得到的10*512词嵌入矩阵，主动乘以sqrt(512)=22.6，
        # 这里我做了一些对比，感觉这个乘以sqrt(512)没啥用… 求反驳。
        # 这里的输出的tensor大小类似于(batch.size, sequence.length, 512)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        # d_model=512, 词嵌入维度；dropout=0.1,
        # max_len=5000代表事先准备好长度为5000的序列的位置编码，其实没必要，
        # 一般100或者200足够了。
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        # (5000,512)矩阵，保持每个位置的位置编码，一共5000个位置，
        # 每个位置用一个512维度向量来表示其位置编码
        position = torch.arange(0, max_len).unsqueeze(1)
        # (5000) -> (5000,1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        # (0,2,…, 4998)一共准备2500个值，供sin, cos调用
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数下标的位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数下标的位置
        pe = pe.unsqueeze(0)
        # (5000, 512) -> (1, 5000, 512) 为batch.size留出位置
        self.register_buffer('pe', pe)
        # 注意，位置编码不会更新，是写死的，所以这个class里面没有可训练的参数。部分其它实现是可训练的，比如tensorflow源码。

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        # 接受1.Embeddings的词嵌入结果x，
        # 然后把自己的位置编码pe，封装成torch的Variable(不需要梯度)，加上去。
        # 例如，假设x是(30,10,512)的一个tensor，
        # 30是batch.size, 10是该batch的序列长度, 512是每个词的词嵌入向量；
        # 则该行代码的第二项是(1, min(10, 5000), 512)=(1,10,512)，
        # 在具体相加的时候，会扩展(1,10,512)为(30,10,512)，
        # 保证一个batch中的30个序列，都使用（叠加）一样的位置编码。
        return self.dropout(x)  # 增加一次dropout操作

#2. attension
# https://zhuanlan.zhihu.com/p/107889011
def attention(query, key, value, mask=None, dropout=None):
    # query, key, value的形状类似于(30, 8, 10, 64), (30, 8, 11, 64),
    # (30, 8, 11, 64)，例如30是batch.size，即当前batch中有多少一个序列；
    # 8=head.num，注意力头的个数；
    # 10=目标序列中词的个数，64是每个词对应的向量表示；
    # 11=源语言序列传过来的memory中，当前序列的词的个数，
    # 64是每个词对应的向量表示。
    # 类似于，这里假定query来自target language sequence；
    # key和value都来自source language sequence.
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)  # 64=d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 先是(30,8,10,64)和(30, 8, 64, 11)相乘，

    #（注意是最后两个维度相乘）得到(30,8,10,11)，
    # 代表10个目标语言序列中每个词和11个源语言序列的分别的“亲密度”。
    # 然后除以sqrt(d_k)=8，防止过大的亲密度。
    # 这里的scores的shape是(30, 8, 10, 11)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
       # 使用mask，对已经计算好的scores，按照mask矩阵，填-1e9，
       # 然后在下一步计算softmax的时候，被设置成-1e9的数对应的值~0,被忽视
    p_attn = F.softmax(scores, dim=-1)
    #    对scores的最后一个维度执行softmax，得到的还是一个tensor,
    # (30, 8, 10, 11)
    if dropout is not None:
       p_attn = dropout(p_attn)  # 执行一次dropout
    return torch.matmul(p_attn, value), p_attn
    # 返回的第一项，是(30,8,10, 11)乘以（最后两个维度相乘）
    # value=(30,8,11,64)，得到的tensor是(30,8,10,64)，
    #    和query的最初的形状一样。另外，返回p_attn，形状为(30,8,10,11).
    # 注意，这里返回p_attn主要是用来可视化显示多头注意力机制。


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        # h=8, d_model=512
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  # We assume d_v always equals d_k 512%8=0
        self.d_k = d_model // h  # d_k=512//8=64
        self.h = h  # 8
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        # 定义四个Linear networks, 每个的大小是(512, 512)的，
        # 每个Linear network里面有两类可训练参数，Weights，
        # 其大小为512*512，以及biases，其大小为512=d_model。
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # 注意，输入query的形状类似于(30, 10, 512)，
        # key.size() ~ (30, 11, 512),
        # 以及value.size() ~ (30, 11, 512)
        if mask is not None:  # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # mask下回细细分解。
        nbatches = query.size(0)  # e.g., nbatches=30
        # 1) Do all the linear projections in batch from
        # d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k)
                                 .transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        # 这里是前三个Linear Networks的具体应用，
        # 例如query=(30,10, 512) -> Linear network -> (30, 10, 512)
        # -> view -> (30,10, 8, 64) -> transpose(1,2) -> (30, 8, 10, 64)
        # ，其他的key和value也是类似地，
        # 从(30, 11, 512) -> (30, 8, 11, 64)。
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        # 调用上面定义好的attention函数，输出的x形状为(30, 8, 10, 64)；
        # attn的形状为(30, 8, 10=target.seq.len, 11=src.seq.len)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # x ~ (30, 8, 10, 64) -> transpose(1,2) ->
        # (30, 10, 8, 64) -> contiguous() and view ->
        # (30, 10, 8*64) = (30, 10, 512)
        return self.linears[-1](x)
        # 执行第四个Linear network，把(30, 10, 512)经过一次linear network，
        # 得到(30, 10, 512).

#4. EncoderLayer
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        # features=d_model=512, eps=epsilon 用于分母的非0化平滑
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        # a_2 是一个可训练参数向量，(512)
        self.b_2 = nn.Parameter(torch.zeros(features))
        # b_2 也是一个可训练参数向量, (512)
        self.eps = eps

    def forward(self, x):
        # x 的形状为(batch.size, sequence.len, 512)
        mean = x.mean(-1, keepdim=True)
        # 对x的最后一个维度，取平均值，得到tensor (batch.size, seq.len)
        std = x.std(-1, keepdim=True)
        # 对x的最后一个维度，取标准方差，得(batch.size, seq.len)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        # 本质上类似于（x-mean)/std，不过这里加入了两个可训练向量
        # a_2 and b_2，以及分母上增加一个极小值epsilon，用来防止std为0
        # 的时候的除法溢出


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        # size=d_model=512; dropout=0.1
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)  # (512)，用来定义a_2和b_2
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer_connection with the same size."
        # x is alike (batch.size, sequence.len, 512)
        # sublayer是一个具体的MultiHeadAttention
        # 或者PositionwiseFeedForward对象
        return x + self.dropout(sublayer(self.norm(x)))
        # x (30, 10, 512) -> norm (LayerNorm) -> (30, 10, 512)
        # -> sublayer (MultiHeadAttention or PositionwiseFeedForward)
        # -> (30, 10, 512) -> dropout -> (30, 10, 512)
        # 然后输入的x（没有走sublayer) + 上面的结果，
        # 即实现了残差相加的功能


# Position-wise Feed-Forward Networks
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        # d_model = 512
        # d_ff = 2048 = 512*4
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        # 构建第一个全连接层，(512, 2048)，其中有两种可训练参数：
        # weights矩阵，(512, 2048)，以及
        # biases偏移向量, (2048)
        self.w_2 = nn.Linear(d_ff, d_model)
        # 构建第二个全连接层, (2048, 512)，两种可训练参数：
        # weights矩阵，(2048, 512)，以及
        # biases偏移向量, (512)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape = (batch.size, sequence.len, 512)
        # 例如, (30, 10, 512)
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
        # x (30, 10, 512) -> self.w_1 -> (30, 10, 2048)
        # -> relu -> (30, 10, 2048)
        # -> dropout -> (30, 10, 2048)
        # -> self.w_2 -> (30, 10, 512)是输出的shape


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        # size=d_model=512
        # self_attn = MultiHeadAttention对象, first sublayer
        # feed_forward = PositionwiseFeedForward对象，second sublayer
        # dropout = 0.1 (e.g.)
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size, dropout), 2)
        # 使用深度克隆方法，完整地复制出来两个SublayerConnection
        self.size = size # 512

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        # x shape = (30, 10, 512)
        # mask 是(batch.size, 10,10)的矩阵，类似于当前一个词w，有哪些词是w可见的
        # 源语言的序列的话，所有其他词都可见，除了"<blank>"这样的填充；
        # 目标语言的序列的话，所有w的左边的词，都可见。
        x = self.sublayer_connection[0](x,
                                        lambda x: self.self_attn(x, x, x, mask))
        # x (30, 10, 512) -> self_attn (MultiHeadAttention)
        # shape is same (30, 10, 512) -> SublayerConnection
        # -> (30, 10, 512)
        return self.sublayer_connection[1](x, self.feed_forward)
        # x 和feed_forward对象一起，给第二个SublayerConnection


#5. Encoder
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        # layer = one EncoderLayer object, N=6
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        # 深copy，N=6，
        self.norm = LayerNorm(layer.size)
        # 定义一个LayerNorm，layer.size=d_model=512
        # 其中有两个可训练参数a_2和b_2

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        # x is alike (30, 10, 512)
        # (batch.size, sequence.len, d_model)
        # mask是类似于(batch.size, 10, 10)的矩阵
        for layer in self.layers:
            x = layer(x, mask)
            # 进行六次EncoderLayer操作
        return self.norm(x)
        # 最后做一次LayerNorm，最后的输出也是(30, 10, 512) shape


#6. DecoderLayer
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn,
      feed_forward, dropout):
      # size = d_model=512,
      # self_attn = one MultiHeadAttention object，目标语言序列的
      # src_attn = second MultiHeadAttention object, 目标语言序列
      # 和源语言序列之间的
      # feed_forward 一个全连接层
      # dropout = 0.1
        super(DecoderLayer, self).__init__()
        self.size = size # 512
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size, dropout), 3)
        # 需要三个SublayerConnection, 分别在
        # self.self_attn, self.src_attn, 和self.feed_forward
        # 的后边

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory # (batch.size, sequence.len, 512)
        # 来自源语言序列的Encoder之后的输出，作为memory
        # 供目标语言的序列检索匹配：（类似于alignment in SMT)
        x = self.sublayer_connection[0](x,
                                        lambda x: self.self_attn(x, x, x, tgt_mask))
        # 通过一个匿名函数，来实现目标序列的自注意力编码
        # 结果扔给sublayer[0]:SublayerConnection
        x = self.sublayer_connection[1](x,
                                        lambda x: self.src_attn(x, m, m, src_mask))
        # 通过第二个匿名函数，来实现目标序列和源序列的注意力计算
        # 结果扔给sublayer[1]:SublayerConnection
        return self.sublayer_connection[2](x, self.feed_forward)
        # 走一个全连接层，然后
        # 结果扔给sublayer[2]:SublayerConnection

# Decoder
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        # layer = DecoderLayer object
        # N = 6
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        # 深度copy六次DecoderLayer
        self.norm = LayerNorm(layer.size)
        # 初始化一个LayerNorm

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
            # 执行六次DecoderLayer
        return self.norm(x)
        # 执行一次LayerNorm

#7. Generator
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        # d_model=512
        # vocab = 目标语言词表大小
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        # 定义一个全连接层，可训练参数个数是(512 * trg_vocab_size) +
        # trg_vocab_size

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
        # x 类似于 (batch.size, sequence.length, 512)
        # -> proj 全连接层 (30, 10, trg_vocab_size) = logits
        # 对最后一个维度执行log_soft_max
        # 得到(30, 10, trg_vocab_size)


#8. model Architecture: EncoderDecoder
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture.
    Base for this and many other models.
    """
    def __init__(self, encoder, decoder,
      src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        # Encoder对象
        self.decoder = decoder
        # Decoder对象
        self.src_embed = src_embed
        # 源语言序列的编码，包括词嵌入和位置编码
        self.tgt_embed = tgt_embed
        # 目标语言序列的编码，包括词嵌入和位置编码
        self.generator = generator
        # 生成器

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
        # 先对源语言序列进行编码，
        # 结果作为memory传递给目标语言的编码器

    def encode(self, src, src_mask):
        # src = (batch.size, seq.length)
        # src_mask 负责对src加掩码
        return self.encoder(self.src_embed(src), src_mask)
        # 对源语言序列进行编码，得到的结果为
        # (batch.size, seq.length, 512)的tensor

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt),
          memory, src_mask, tgt_mask)
        # 对目标语言序列进行编码，得到的结果为
        # (batch.size, seq.length, 512)的tensor


## ------------------------------------ ##
## ------------------------------------ ##

# 生成mask掩模矩阵
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

# 生成EncoderDecoder整体模型
def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


if __name__ == "__main__":
    # first example

    def data_gen(V, batch, nbatches):
        "Generate random data for a src-tgt copy task."
        for i in range(nbatches):
            data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
            data[:, 0] = 1
            src = Variable(data, requires_grad=False)
            tgt = Variable(data, requires_grad=False)
            yield Batch(src, tgt, 0)


    class SimpleLossCompute:
        "A simple loss compute and train function."

        def __init__(self, generator, criterion, opt=None):
            self.generator = generator
            self.criterion = criterion
            self.opt = opt

        def __call__(self, x, y, norm):
            x = self.generator(x)
            loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                                  y.contiguous().view(-1)) / norm
            loss.backward()
            if self.opt is not None:
                self.opt.step()
                self.opt.optimizer.zero_grad()
            return loss.data.item() * norm


    # Train the simple copy task.
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        model.train()
        run_epoch(data_gen(V, 30, 20), model,
                  SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(V, 30, 5), model,
                        SimpleLossCompute(model.generator, criterion, None)))

    # real world  example
    # For data loading.
    from torchtext import data, datasets

    if True:
        import spacy

        spacy_de = spacy.load('de')
        spacy_en = spacy.load('en')


        def tokenize_de(text):
            return [tok.text for tok in spacy_de.tokenizer(text)]


        def tokenize_en(text):
            return [tok.text for tok in spacy_en.tokenizer(text)]


        BOS_WORD = '<s>'
        EOS_WORD = '</s>'
        BLANK_WORD = "<blank>"
        SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
        TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                         eos_token=EOS_WORD, pad_token=BLANK_WORD)

        MAX_LEN = 100
        train, val, test = datasets.IWSLT.splits(
            exts=('.de', '.en'), fields=(SRC, TGT),
            filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                                  len(vars(x)['trg']) <= MAX_LEN)
        MIN_FREQ = 2
        SRC.build_vocab(train.src, min_freq=MIN_FREQ)
        TGT.build_vocab(train.trg, min_freq=MIN_FREQ)


    class MyIterator(data.Iterator):
        def create_batches(self):
            if self.train:
                def pool(d, random_shuffler):
                    for p in data.batch(d, self.batch_size * 100):
                        p_batch = data.batch(
                            sorted(p, key=self.sort_key),
                            self.batch_size, self.batch_size_fn)
                        for b in random_shuffler(list(p_batch)):
                            yield b

                self.batches = pool(self.data(), self.random_shuffler)

            else:
                self.batches = []
                for b in data.batch(self.data(), self.batch_size,
                                    self.batch_size_fn):
                    self.batches.append(sorted(b, key=self.sort_key))


    def rebatch(pad_idx, batch):
        "Fix order in torchtext to match ours"
        src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
        return Batch(src, trg, pad_idx)


    # Skip if not interested in multigpu.
    class MultiGPULossCompute:
        "A multi-gpu loss compute and train function."

        def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
            # Send out to different gpus.
            self.generator = generator
            self.criterion = nn.parallel.replicate(criterion,
                                                   devices=devices)
            self.opt = opt
            self.devices = devices
            self.chunk_size = chunk_size

        def __call__(self, out, targets, normalize):
            total = 0.0
            generator = nn.parallel.replicate(self.generator,
                                              devices=self.devices)
            out_scatter = nn.parallel.scatter(out,
                                              target_gpus=self.devices)
            out_grad = [[] for _ in out_scatter]
            targets = nn.parallel.scatter(targets,
                                          target_gpus=self.devices)

            # Divide generating into chunks.
            chunk_size = self.chunk_size
            for i in range(0, out_scatter[0].size(1), chunk_size):
                # Predict distributions
                out_column = [[Variable(o[:, i:i + chunk_size].data,
                                        requires_grad=self.opt is not None)]
                              for o in out_scatter]
                gen = nn.parallel.parallel_apply(generator, out_column)

                # Compute loss.
                y = [(g.contiguous().view(-1, g.size(-1)),
                      t[:, i:i + chunk_size].contiguous().view(-1))
                     for g, t in zip(gen, targets)]
                loss = nn.parallel.parallel_apply(self.criterion, y)

                # Sum and normalize loss
                l = nn.parallel.gather(loss,
                                       target_device=self.devices[0])
                l = l.sum()[0] / normalize
                total += l.data[0]

                # Backprop loss to output of transformer
                if self.opt is not None:
                    l.backward()
                    for j, l in enumerate(loss):
                        out_grad[j].append(out_column[j][0].grad.data.clone())

            # Backprop all loss through transformer.
            if self.opt is not None:
                out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
                o1 = out
                o2 = nn.parallel.gather(out_grad,
                                        target_device=self.devices[0])
                o1.backward(gradient=o2)
                self.opt.step()
                self.opt.optimizer.zero_grad()
            return total * normalize


    # GPUs to use
    devices = [0, 1, 2, 3]
    if True:
        pad_idx = TGT.vocab.stoi["<blank>"]
        model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
        model.cuda()
        criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
        criterion.cuda()
        BATCH_SIZE = 12000
        train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
                                repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                                batch_size_fn=batch_size_fn, train=True)
        valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
                                repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                                batch_size_fn=batch_size_fn, train=False)
        model_par = nn.DataParallel(model, device_ids=devices)
    None

    # !wget https://s3.amazonaws.com/opennmt-models/iwslt.pt

    if False:
        model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        for epoch in range(10):
            model_par.train()
            run_epoch((rebatch(pad_idx, b) for b in train_iter),
                      model_par,
                      MultiGPULossCompute(model.generator, criterion,
                                          devices=devices, opt=model_opt))
            model_par.eval()
            loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                             model_par,
                             MultiGPULossCompute(model.generator, criterion,
                                                 devices=devices, opt=None))
            print(loss)
    else:
        model = torch.load("iwslt.pt")

    for i, batch in enumerate(valid_iter):
        src = batch.src.transpose(0, 1)[:1]
        src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask,
                            max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
        print("Translation:", end="\t")
        for i in range(1, out.size(1)):
            sym = TGT.vocab.itos[out[0, i]]
            if sym == "</s>": break
            print(sym, end=" ")
        print()
        print("Target:", end="\t")
        for i in range(1, batch.trg.size(0)):
            sym = TGT.vocab.itos[batch.trg.data[i, 0]]
            if sym == "</s>": break
            print(sym, end=" ")
        print()
        break
