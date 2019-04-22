# !pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl numpy matplotlib spacy torchtext seaborn 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")
from transformer_functions import clones, subsequent_mask, attention
# %matplotlib inline
from torchdiffeq import odeint_adjoint as odeint

TOL = 1e-3

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class ODE_Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, ode_layer):
        super(ODE_Encoder, self).__init__()
        #self.layers = clones(layer, N)
        self.ode_layer = ode_layer
        self.norm = LayerNorm(ode_layer.size)
        self.integration_time = torch.tensor([0, 1]).float()
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
#         for layer in self.layers:
#             x = layer(x, mask)
        self.integration_time = self.integration_time.type_as(x)
        self.ode_layer.set_mask(mask)
        out = odeint(self.ode_layer, x, self.integration_time, rtol=TOL, atol=TOL)
        return self.norm(out[1])
    

class ODE_EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(ODE_EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.mask = None
        
        self.sublayer = clones(SublayerRoutine(size, dropout), 2)        
        # self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        
        self.nfe = 0
        
    def set_mask(self, mask):
        "Mandatory before calling forward, in order to set the mask for that operation"
        self.mask = mask

    def forward(self, t, x):
        "Follow Figure 1 (left) for connections."
        self.nfe += 1
        # print(t)
        self_attn = lambda t, x: self.self_attn(t, x, x, x, self.mask)
        feed_forward = self.feed_forward
        
        mh = self.sublayer[0](t, x, self_attn)
        
        dx = mh + self.sublayer[1](t, x + mh, feed_forward)
        # x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # return self.sublayer[1](x, self.feed_forward)
        return dx

    
class ODE_Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, ode_layer):
        super(ODE_Decoder, self).__init__()
        self.ode_layer = ode_layer
        self.norm = LayerNorm(ode_layer.size)
        self.integration_time = torch.tensor([0, 1]).float()
        
    def forward(self, x, memory, src_mask, tgt_mask):
#         for layer in self.layers:
#             x = layer(x, memory, src_mask, tgt_mask)
#         return self.norm(x)
        x_len = x.shape[1]
        # Concatenate x and memory to pass into forward together
        x = torch.cat([x, memory], dim=1)
        self.ode_layer.set_forward_attributes(src_mask, tgt_mask, x_len)

        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.ode_layer, x, self.integration_time, rtol=TOL, atol=TOL)
        
        x = out[1][:, :x_len, :self.ode_layer.size]
        return self.norm(x)

    
class ODE_DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(ODE_DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.src_mask = None
        self.tgt_mask = None
        
        self.sublayer = clones(SublayerRoutine(size, dropout), 3)
        self.size = size
        self.nfe = 0
        self.x_len = None
        
    def set_forward_attributes(self, src_mask, tgt_mask, x_len):
        "Mandatory before calling forward, in order to set the mask for that operation"
        # self.memory = memory
        self.src_mask = src_mask
        self.tgt_mask = tgt_mask
        self.x_len = x_len

    def forward(self, t, x):
        "Follow Figure 1 (left) for connections."
        # print(t)
        self.nfe += 1
        # Extract memory and x
        m = x[:, self.x_len:, :]
        x = x[:, :self.x_len, :]
        
        mh_masked = self.sublayer[0](t, x, lambda t, x: self.self_attn(t, x, x, x, self.tgt_mask))
        q = x + mh_masked
        
        # m = self.memory
        mh = self.sublayer[1](t, q, lambda t, x: self.src_attn(t, x, m, m, self.src_mask))
        a = q + mh
        
        ff = self.sublayer[2](t, a, self.feed_forward)
        dx = mh_masked + mh + ff
        
        # Because m stays constant, its derivative during the transformation is set to 0
        dm = torch.zeros_like(m)
        out = torch.cat([dx, dm], dim=1)
        return out
    
    
class SublayerRoutine(nn.Module):
    """
    Applies norm to input and dropout to output
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerRoutine, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, t, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return self.dropout(sublayer(t, self.norm(x)))
    
    
# class MemorySublayerRoutine(nn.Module):
#     """
#     Applies norm to input and dropout to output
#     Note for code simplicity the norm is first as opposed to last.
#     """
#     def __init__(self, size, dropout):
#         super(MemorySublayerRoutine, self).__init__()
#         self.norm = LayerNorm(size)
#         self.dropout = nn.Dropout(dropout)
#         self.size = size

#     def forward(self, t, x, sublayer):
#         "Apply residual connection to any sublayer with the same size."
#         m = x[:, :, self.size:]
#         x = self.norm(x[:, :, :self.size])
#         x = torch.cat([x, m], dim=2)
#         return self.dropout(sublayer(t, self.norm(x)))

    
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

    
class ConcatMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(ConcatMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.linear_V = nn.Linear(d_model + 1, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, t, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key))]
        
        # Concat t to the back of value
        tt = torch.ones_like(value[:, :, :1]) * t
        value = torch.cat([tt, value], 2)
        value = self.linear_V(value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

    
class ConcatPositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(ConcatPositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model + 1, d_ff)
        self.w_2 = nn.Linear(d_ff + 1, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, t, x):
        tt_1 = torch.ones_like(x[:, :, :1]) * t
        ttx = torch.cat([tt_1, x], 2)
        h = F.relu(self.w_1(ttx))
        
        tt_2 = torch.ones_like(h[:, :, :1]) * t
        tth = torch.cat([tt_2, h], 2)
        
        return self.w_2(self.dropout(tth))

    
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

    
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


"""Make full model with given classes & functions"""
def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = ConcatMultiHeadedAttention(h, d_model)
    ff = ConcatPositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    enc_layer = ODE_EncoderLayer(d_model, c(attn), c(ff), dropout)
    dec_layer = ODE_DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
    model = EncoderDecoder(
        ODE_Encoder(enc_layer),
        ODE_Decoder(dec_layer),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model, enc_layer, dec_layer


if __name__ == '__main__':
	transformer = make_model(src_vocab=10, tgt_vocab=10)




