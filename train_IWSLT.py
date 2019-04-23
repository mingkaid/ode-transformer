
"""
Train with actual dataset
To run, add datasets de and en in the same directory
"""

import argparse
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")

from train_functions import get_std_opt, run_epoch,Batch, LabelSmoothing, NoamOpt, \
SimpleLossCompute, greedy_decode, create_checkpoint, load_checkpoint, \
MultiGPULossCompute, RunningAverageMeter

parser = argparse.ArgumentParser()
parser.add_argument('--resume-checkpoint', type=str, default=None)
parser.add_argument('--save', type=str, default='./outputs')
parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda')
parser.add_argument('--gpu-name', type=int, default=0)
parser.add_argument('--train-batch-size', type=int, default=1000)
parser.add_argument('--test-batch-size', type=int, default=1000)
parser.add_argument('--num-epochs', type=int, default=10)
parser.add_argument('--model', type=str, choices=['transformer', 'ode'], default='ode')
parser.add_argument('--warmup', type=int, default=2000)
parser.add_argument('--print-interval', type=int, default=50)
args = parser.parse_args()

is_ode = True if args.model == 'ode' else False
if is_ode == True:
    print('ODE transformer selected')
    from transformer_ode import make_model
else:
    print('Regular transformer selected')
    from transformer import make_model

"""
Data Preprocessing
"""
from torchtext import data, datasets
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
TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, 
                 eos_token = EOS_WORD, pad_token=BLANK_WORD)

MAX_LEN = 100
train, val, test = datasets.IWSLT.splits(
    exts=('.de', '.en'), fields=(SRC, TGT), 
    filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
        len(vars(x)['trg']) <= MAX_LEN)
MIN_FREQ = 2
SRC.build_vocab(train.src, min_freq=MIN_FREQ)
TGT.build_vocab(train.trg, min_freq=MIN_FREQ)


"""
Create batches
"""

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


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


"""
Train and evaluate
Everything according to Vaswani et al. (2017) except warmup steps for NoamOpt
"""
pad_idx = TGT.vocab.stoi["<blank>"]
batch_time_meter = RunningAverageMeter()
train_loss_meter = RunningAverageMeter()
if is_ode == True:
    model, enc_layer, dec_layer = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    f_nfe_meter_enc = RunningAverageMeter()
    f_nfe_meter_dec = RunningAverageMeter()
    b_nfe_meter_enc = RunningAverageMeter()
    b_nfe_meter_dec = RunningAverageMeter()
else:
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    
criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)

train_iter = MyIterator(train, batch_size=args.train_batch_size, device=args.device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=True)
valid_iter = MyIterator(val, batch_size=args.test_batch_size, device=args.device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=False)

if args.device == 'cuda':
    model.cuda()
    criterion.cuda()
    print('Training on GPU {}'.format(args.gpu_name))
    devices = [args.gpu_name]
    #num_devices = torch.cuda.device_count()
    #print('Training on {} GPUs'.format(num_devices))
    #devices = list(range(num_devices))
    
"""
If resuming from checkpoint, load weights and optimizer state now
"""
optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
if args.resume_checkpoint != None:
    if is_ode == False:
        epoch, model, model_opt = load_checkpoint(model, optimizer, args.resume_checkpoint,
                                                  train_loss_meter=train_loss_meter,
                                                  batch_time_meter=batch_time_meter)
    else:
        epoch, model, model_opt = load_checkpoint(model, optimizer, args.resume_checkpoint,
                                                  is_ode=True, 
                                                  f_nfe_meter_enc=f_nfe_meter_enc,
                                                  f_nfe_meter_dec=f_nfe_meter_dec,
                                                  b_nfe_meter_enc=b_nfe_meter_enc,
                                                  b_nfe_meter_dec=b_nfe_meter_dec,
                                                  batch_time_meter=batch_time_meter,
                                                  train_loss_meter=train_loss_meter,
                                                  resume_meters = True)
    epoch += 1
    if args.device == 'cuda':
        model_used = nn.DataParallel(model, device_ids=devices)
    else:
        model_used = model
else:
    epoch = 1
    model_size, factor, warmup = model.src_embed[0].d_model, 1, args.warmup
    model_opt = NoamOpt(model_size, factor, warmup, optimizer)
    model_used = model    
    
"""
Run for the specified number of epochs
"""
    
for i in range(epoch, epoch + args.num_epochs):
    print('Epoch {}'.format(i))
    if args.device == 'cuda':
        """ Adjust loss computing method to training device """
        train_loss_compute = MultiGPULossCompute(model.generator, criterion, 
                                                 devices=devices, opt=model_opt)
        val_loss_compute = MultiGPULossCompute(model.generator, criterion, 
                                               devices=devices, opt=None)
    else:
        train_loss_compute = SimpleLossCompute(model.generator, criterion, 
                                               opt=model_opt)
        val_loss_compute = SimpleLossCompute(model.generator, criterion, 
                                             opt=None)
    """ Train """    
    model_used.train()
    if is_ode == True:
        """ Record number of function evaluations if we are using ODE """
        run_epoch((rebatch(pad_idx, b) for b in train_iter), model_used, 
                  train_loss_compute, print_interval=args.print_interval,
                  enc_layer=enc_layer, dec_layer=dec_layer,
                  f_nfe_meter_enc=f_nfe_meter_enc, f_nfe_meter_dec=f_nfe_meter_dec,
                  b_nfe_meter_enc=b_nfe_meter_enc, b_nfe_meter_dec=b_nfe_meter_dec,
                  batch_time_meter=batch_time_meter, train_loss_meter=train_loss_meter,
                  is_ode=is_ode)
    else:
        run_epoch((rebatch(pad_idx, b) for b in train_iter), model_used, 
                  train_loss_compute, print_interval=args.print_interval,
                  batch_time_meter=batch_time_meter, train_loss_meter=train_loss_meter,
                  is_ode=is_ode)
    
    """ Validate """
    model_used.eval()
    val_loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), 
                     model_used, 
                     val_loss_compute, 
                     use_meters=False)
    print('Val Loss: {:.4f}'.format(val_loss))
    
    """ Save Progress """
    if is_ode == False:
        create_checkpoint(model, model_opt, i, val_loss, path=args.save,
                          batch_time_meter=batch_time_meter,
                          train_loss_meter=train_loss_meter)
    else:
        create_checkpoint(model, model_opt, i, val_loss, path=args.save,
                          is_ode=True, 
                          f_nfe_meter_enc=f_nfe_meter_enc,
                          f_nfe_meter_dec=f_nfe_meter_dec,
                          b_nfe_meter_enc=b_nfe_meter_enc,
                          b_nfe_meter_dec=b_nfe_meter_dec, 
                          train_loss_meter=train_loss_meter,
                          batch_time_meter=batch_time_meter)
        f_nfe_meter_enc.hist = []
        f_nfe_meter_dec.hist = []
        b_nfe_meter_enc.hist = []
        b_nfe_meter_dec.hist = []
    train_loss_meter.hist = []
    batch_time_meter.hist = []

print('Training complete')


