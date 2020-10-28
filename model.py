import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math, copy
import numpy as np
from basemodel_1D import TemporalConvNet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_mask_bidirectional(size, atten_len_a, atten_len_b):
    attn_shape = (1, size, size)
    past_all_mask = np.triu(np.ones(attn_shape), k=atten_len_b).astype('uint8')
    past_all_mask = torch.from_numpy(past_all_mask)
    past_all_mask = past_all_mask == 0
    no_need_mask = np.triu(np.ones(attn_shape), k=-atten_len_a + 1).astype('uint8')
    no_need_mask = torch.from_numpy(no_need_mask)
    gene_mask = no_need_mask * past_all_mask

    return gene_mask.to(device)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features)).to(device)
        self.b_2 = nn.Parameter(torch.zeros(features)).to(device)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Encoder(nn.Module):

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = layer
        self.norm = LayerNorm(layer[0].size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class MultiModalEncoder(nn.Module):

    def __init__(self, layer, N, modal_num):
        super(MultiModalEncoder, self).__init__()
        self.modal_num = modal_num
        self.layers = layer
        self.norm = nn.ModuleList()
        for i in range(self.modal_num):
            self.norm.append(LayerNorm(layer[0].size))


    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        _x = torch.chunk(x, self.modal_num, dim=-1)
        _x_list = []
        for i in range(self.modal_num):
            _x_list.append(self.norm[i](_x[i]))

        x = torch.cat(_x_list, dim=-1)

        return x


class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):

        return x + self.dropout(sublayer(self.norm(x)))


class MultiModalSublayerConnection(nn.Module):

    def __init__(self, size, modal_num, dropout):
        super(MultiModalSublayerConnection, self).__init__()
        self.modal_num = modal_num

        self.norm = nn.ModuleList()
        for i in range(self.modal_num):
            self.norm.append(LayerNorm(size))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        residual = x

        _x_list = []
        _x = torch.chunk(x, self.modal_num, -1)
        for i in range(self.modal_num):
            _x_list.append(self.norm[i](_x[i]))
        x = torch.cat(_x_list, dim=-1)

        return self.dropout(sublayer(x)) + residual


class EncoderLayer(nn.Module):

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList()
        self.sublayer.append(SublayerConnection(size, dropout))
        self.sublayer.append(SublayerConnection(size, dropout))

        self.size = size

    def forward(self, x, mask):

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class MultiModalEncoderLayer(nn.Module):

    def __init__(self, size, modal_num, mm_atten, mt_atten, feed_forward, dropout):
        super(MultiModalEncoderLayer, self).__init__()
        self.modal_num = modal_num

        self.mm_atten = mm_atten
        self.mt_atten = mt_atten
        self.feed_forward = feed_forward

        mm_sublayer = MultiModalSublayerConnection(size, modal_num, dropout)
        mt_sublayer = nn.ModuleList()
        for i in range(modal_num):
            mt_sublayer.append(SublayerConnection(size, dropout))
        ff_sublayer = nn.ModuleList()
        for i in range(modal_num):
            ff_sublayer.append(SublayerConnection(size, dropout))

        self.sublayer = nn.ModuleList()
        self.sublayer.append(mm_sublayer)
        self.sublayer.append(mt_sublayer)
        self.sublayer.append(ff_sublayer)

        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.mm_atten(x, x, x))

        _x = torch.chunk(x, self.modal_num, dim=-1)
        _x_list = []
        for i in range(self.modal_num):
            feature = self.sublayer[1][i](_x[i], lambda x: self.mt_atten[i](x, x, x, mask[i]))
            feature = self.sublayer[2][i](feature, self.feed_forward[i])
            _x_list.append(feature)
        x = torch.cat(_x_list, dim=-1)

        return x


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):

        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):

        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, _ = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class MultiModalAttention(nn.Module):
    def __init__(self, h, d_model, modal_num, dropout=0.1):

        super(MultiModalAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.modal_num = modal_num
        self.mm_linears = nn.ModuleList()
        for i in range(self.modal_num):
            linears = clones(nn.Linear(d_model, d_model), 4)
            self.mm_linears.append(linears)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        query = torch.chunk(query, self.modal_num, dim=-1)
        key   = torch.chunk(key, self.modal_num, dim=-1)
        value = torch.chunk(value, self.modal_num, dim=-1)

        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query[0].size(0)

        _query_list = []
        _key_list = []
        _value_list = []
        for i in range(self.modal_num):
            _query_list.append(self.mm_linears[i][0](query[i]).view(nbatches, -1, self.h, self.d_k))
            _key_list.append(self.mm_linears[i][1](key[i]).view(nbatches, -1, self.h, self.d_k))
            _value_list.append(self.mm_linears[i][2](value[i]).view(nbatches, -1, self.h, self.d_k))

        mm_query = torch.stack(_query_list, dim=-2)
        mm_key = torch.stack(_key_list, dim=-2)
        mm_value = torch.stack(_value_list, dim=-2)
        x, _ = attention(mm_query, mm_key, mm_value, mask=mask, dropout=self.dropout)

        x = x.transpose(-2, -3).contiguous().view(nbatches, -1, self.modal_num, self.h * self.d_k)
        _x = torch.chunk(x, self.modal_num, dim=-2)
        _x_list = []
        for i in range(self.modal_num):
            _x_list.append(self.mm_linears[i][-1](_x[i].squeeze()))
        x = torch.cat(_x_list, dim=-1)

        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SEmbeddings(nn.Module):
    def __init__(self, d_model, dim):
        super(SEmbeddings, self).__init__()
        self.lut = nn.Linear(dim, d_model)
        self.d_model = d_model

    def forward(self, x):
        x = self.lut(x)
        x = x * math.sqrt(self.d_model)
        return x


class TEmbeddings(nn.Module):
    def __init__(self, opts, dim):
        super(TEmbeddings, self).__init__()
        self.levels = opts.levels
        self.ksize = opts.ksize
        self.d_model = opts.d_model
        self.dropout = opts.dropout

        self.channel_sizes = [self.d_model] * self.levels
        self.lut = TemporalConvNet(dim, self.channel_sizes, kernel_size=self.ksize, dropout=self.dropout)

    def forward(self, x):
        x = self.lut(x.transpose(1, 2)).transpose(1, 2) * math.sqrt(self.d_model)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        v = torch.arange(0, d_model, 2).type(torch.float)
        v = v * -(math.log(1000.0) / d_model)
        div_term = torch.exp(v)
        pe[:, 0::2] = torch.sin(position.type(torch.float) * div_term)
        pe[:, 1::2] = torch.cos(position.type(torch.float) * div_term)
        pe = pe.unsqueeze(0).to(device)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class ProcessInput(nn.Module):
    def __init__(self, opts, dim):
        super(ProcessInput, self).__init__()

        if opts.embed == 'spatial':
            self.Embeddings = SEmbeddings(opts.d_model, dim)
        elif opts.embed == 'temporal':
            self.Embeddings = TEmbeddings(opts.d_model, dim)
        self.PositionEncoding = PositionalEncoding(opts.d_model, opts.dropout_position, max_len=5000)

    def forward(self, x):
        return self.PositionEncoding(self.Embeddings(x))


class TE(nn.Module):

    def __init__(self, opts, num_features):
        super(TE, self).__init__()

        self.modal_num = opts.modal_num
        assert self.modal_num == 1, 'TE model is only used for single feature streams ...'

        self.mask_a_length = int(opts.mask_a_length)
        self.mask_b_length = int(opts.mask_b_length)

        self.N = opts.block_num
        self.dropout = opts.dropout
        self.h = opts.h
        self.d_model = opts.d_model
        self.d_ff = opts.d_ff

        self.input = ProcessInput(opts, num_features)
        self.regress = nn.Linear(self.d_model, 1)
        self.dropout_embed = nn.Dropout(p=opts.dropout_embed)

        encoder_layer = nn.ModuleList()
        for i in range(self.N):
            atten = MultiHeadedAttention(self.h, self.d_model, self.dropout)
            ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
            encoder_layer.append(EncoderLayer(self.d_model, atten, ff, self.dropout))
        self.te = Encoder(encoder_layer, self.N)

        for p in self.te.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for p in self.input.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for p in self.regress.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.input(x)
        x = self.dropout_embed(x)

        mask = generate_mask_bidirectional(x.shape[1], self.mask_a_length, self.mask_b_length)
        x = self.te(x, mask)

        return self.regress(x)


class TEMMA(nn.Module):

    def __init__(self, opts, num_features):
        super(TEMMA, self).__init__()

        self.modal_num = opts.modal_num
        assert self.modal_num > 1, 'TEMMA model is only used for multiple feature streams ...'

        self.mask_a_length = [int(l) for l in opts.mask_a_length.split(',')]
        self.mask_b_length = [int(l) for l in opts.mask_b_length.split(',')]

        self.num_features = num_features
        self.modal_num = opts.modal_num
        self.N = opts.block_num
        self.dropout_mmatten = opts.dropout_mmatten
        self.dropout_mtatten = opts.dropout_mtatten
        self.dropout_ff = opts.dropout_ff
        self.dropout_subconnect = opts.dropout_subconnect
        self.h = opts.h
        self.h_mma = opts.h_mma
        self.d_model = opts.d_model
        self.d_ff = opts.d_ff

        self.input = nn.ModuleList()
        for i in range(self.modal_num):
            self.input.append(ProcessInput(opts, num_features // self.modal_num))
        self.dropout_embed = nn.Dropout(p=opts.dropout_embed)

        multimodal_encoder_layer = nn.ModuleList()
        for i in range(self.N):
            mm_atten = MultiModalAttention(self.h_mma, self.d_model, self.modal_num, self.dropout_mmatten)
            mt_atten = nn.ModuleList()
            ff = nn.ModuleList()
            for j in range(self.modal_num):
                mt_atten.append(MultiHeadedAttention(self.h, self.d_model, self.dropout_mtatten))
                ff.append(PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout_ff))
            multimodal_encoder_layer.append(MultiModalEncoderLayer(self.d_model, self.modal_num, mm_atten, mt_atten, ff, self.dropout_subconnect))

        self.temma = MultiModalEncoder(multimodal_encoder_layer, self.N, self.modal_num)
        self.regress = nn.Linear(self.d_model * self.modal_num, 1)

        for p in self.temma.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for p in self.input.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for p in self.regress.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):

        _x = torch.chunk(x, self.modal_num, dim=-1)
        _x_list = []
        for i in range(self.modal_num):
            _x_list.append(self.input[i](_x[i]))
        x = torch.cat(_x_list, dim=-1)

        x = self.dropout_embed(x)

        mask = []
        for i in range(self.modal_num):
            mask.append(generate_mask_bidirectional(x.shape[1], self.mask_a_length[i], self.mask_b_length[i]))
        x = self.temma(x, mask)

        return self.regress(x)
