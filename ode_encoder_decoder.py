import os
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import math, copy, time
from torch.autograd import Variable
import seaborn
seaborn.set_context(context="talk")

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from transformer import EncoderDecoder, EncoderLayer, DecoderLayer, LayerNorm, \
MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding, Embeddings, Generator
from transformer_functions import clones, subsequent_mask, attention
from torchdiffeq import odeint_adjoint as odeint


# parser = argparse.ArgumentParser()
# parser.add_argument('--network', type=str, choices=['transformer', 'odetransformer'], default='odetransformer')
# parser.add_argument('--tol', type=float, default=1e-3)
# parser.add_argument('--adjoint', type=eval, default=True, choices=[True, False])
# # parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
# parser.add_argument('--nepochs', type=int, default=160)
# # parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
# parser.add_argument('--lr', type=float, default=0.1)
# parser.add_argument('--batch_size', type=int, default=128)
# parser.add_argument('--test_batch_size', type=int, default=1000)

# parser.add_argument('--save', type=str, default='./experiment1')
# parser.add_argument('--debug', action='store_true')
# parser.add_argument('--gpu', type=int, default=0)
# args = parser.parse_args()



class ConcatEncoderLayer(nn.Module):
    
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(ConcatEncoderLayer, self).__init__()
        module = EncoderLayer
        self._layer = module(size, self_attn, feed_forward, dropout)
        self.size = size
        self.mask = None
        
    def update_mask(self, mask):
        self.mask = mask
    
    def forward(self, t, x):
        # x, mask = *x
        tt = torch.ones_like(x[:, :, :1]) * t
        ttx = torch.cat([tt, x], 2)
        return self._layer(ttx, self.mask)

class OdeEncoder(nn.Module):

    def __init__(self, odelayer):
        super(OdeEncoder, self).__init__()
        self.odelayer = odelayer
        self.integration_time = torch.tensor([0, 1]).float()
        self.norm = LayerNorm(odelayer.size)

    def forward(self, x, mask):
        self.integration_time = self.integration_time.type_as(x)
        #odefunc = lambda t, h: self.odelayer(t, h, mask)
        # odefunc = self.odelayer(t, h, mask)
        self.odelayer.update_mask(mask)
        out = odeint(self.odelayer, x, self.integration_time, rtol=0.01, atol=0.01)
        return self.norm(out[1])

    @property
    def nfe(self):
        return self.odelayer.nfe

    @nfe.setter
    def nfe(self, value):
        self.odelayer.nfe = value
        
class ConcatDecoderLayer(nn.Module):
    
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(ConcatDecoderLayer, self).__init__()
        module = DecoderLayer
        self._layer = module(size, self_attn, src_attn, feed_forward, dropout)
        self.size = size
        
    def update_mask(self, memory, src_mask, tgt_mask):
        self.memory = memory
        self.src_mask = src_mask
        self.tgt_mask = tgt_mask
    
    def forward(self, t, x):
        tt = torch.ones_like(x[:, :, :1]) * t
        ttx = torch.cat([tt, x], 2)
        return self._layer(ttx, self.memory, self.src_mask, self.tgt_mask)
    
class OdeDecoder(nn.Module):
    def __init__(self, odelayer):
        super(OdeDecoder, self).__init__()
        self.odelayer = odelayer
        self.integration_time = torch.tensor([0, 1]).float()
        self.norm = LayerNorm(odelayer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        self.integration_time = self.integration_time.type_as(x)
        # odefunc = lambda t, h: self.odelayer(t, h, memory, src_mask, tgt_mask)
        # odefunc = self.odelayer(t, h, memory, src_mask, tgt_mask)
        self.odelayer.update_mask(memory, src_mask, tgt_mask)
        out = odeint(self.odelayer, x, self.integration_time, rtol=0.01, atol=0.01)
        return self.norm(out[1])
        

"""Make full model with given classes & functions"""
def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=511, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model + 1)
    ff = PositionwiseFeedForward(d_model + 1, d_ff, dropout, d_out = d_model)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        OdeEncoder(ConcatEncoderLayer(d_model, c(attn), c(ff), dropout)),
        OdeDecoder(ConcatDecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout)),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


if __name__ == '__main__':
    make_model(src_vocab=10, tgt_vocab=10)

# TODO1: Change to Transformer training code from Harvard
# TODO2: Change make_model in transformer.py to allow for 
# OdeEncoder and OdeDecoder options. Default on traditional
# TODO3: Run the code in ODE mode to debug
# TODO4: Clean up the MNIST-related code above if any doesn't
# contribute to Transformer
