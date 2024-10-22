import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
from datetime import datetime
from transformer_functions import subsequent_mask
seaborn.set_context(context="talk")
# %matplotlib inline


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        #print(src)
        #print(trg)
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


def run_epoch(data_iter, model, loss_compute, print_interval=50, 
              is_ode=False, enc_layer=None, dec_layer=None,
              f_nfe_meter_enc=None, b_nfe_meter_enc=None,
              f_nfe_meter_dec=None, b_nfe_meter_dec=None,
              batch_time_meter=None, train_loss_meter=None,
              use_meters=True):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    i = 0
    for _, batch in enumerate(data_iter):
        batch_start = time.time()
        
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        if use_meters and is_ode:
            # Record number of function evaluations in the encoder and decoder layers
            nfe_forward_enc = enc_layer.nfe
            enc_layer.nfe = 0
            nfe_forward_dec = dec_layer.nfe
            dec_layer.nfe = 0
            
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        if use_meters and is_ode:
            nfe_backward_enc = enc_layer.nfe
            enc_layer.nfe = 0
            nfe_backward_dec = dec_layer.nfe
            dec_layer.nfe = 0
        
        total_loss += loss
        total_tokens += batch.ntokens.item()
        tokens += batch.ntokens.item()
        
        if use_meters == True:
            batch_time_meter.update(time.time() - batch_start)
            train_loss_meter.update(loss / batch.ntokens.item())
            if is_ode:
                f_nfe_meter_enc.update(nfe_forward_enc)
                f_nfe_meter_dec.update(nfe_forward_dec)
                b_nfe_meter_enc.update(nfe_backward_enc)
                b_nfe_meter_dec.update(nfe_backward_dec)
        
        i += 1
        #print(i)
        if (i-1) % print_interval == 0:
            elapsed = time.time() - start
            if not is_ode:
                print("Step {} Loss {:f} Tokens/Sec {:f}".format(
                      i, loss / batch.ntokens.item(), tokens / elapsed), end='\r')
            else:
                print("Step {} Loss {:.4f} Tokens/Sec {:.2f} "
                      'NFE: F-enc {:.1f} F-dec {:.1f} B-enc {:.1f} B-dec {:.1f}'.format(
                          i, loss / batch.ntokens.item(), tokens / elapsed, 
                          f_nfe_meter_enc.avg, f_nfe_meter_dec.avg, 
                          b_nfe_meter_enc.avg, b_nfe_meter_dec.avg
                      ),
                      end='\r')
            start = time.time()
            tokens = 0
    print()
    return total_loss / total_tokens


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0
        self.hist = []

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        self.hist.append(val)


class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
    
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm.float()
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm.float().item()
    
"""
Multi-GPU Loss Compute
"""
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
            out_column = [[Variable(o[:, i:i+chunk_size].data, 
                                    requires_grad=self.opt is not None)] 
                           for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss. 
            y = [(g.contiguous().view(-1, g.size(-1)), 
                  t[:, i:i+chunk_size].contiguous().view(-1)) 
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss, 
                                   target_device=self.devices[0])
            l = l.sum() / normalize
            total += l.item()

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
        return total * normalize.float().item()

"""
Regularization
"""
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
        if mask.sum() and len(mask)>0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

"""
Optimizers
"""
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
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


"""
Greedy decoding for translation
"This dataset is pretty small so the translations with greedy search are reasonably accurate."
"""
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(src.shape[0], 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(src.shape[0], 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

"""
Creates checkpoint adapted to the NoamOpt training scheme
"""
def create_checkpoint(model, model_opt, epoch, val_loss, path='./outputs/', 
                      name='checkpoint_epoch', is_ode=False, 
                      f_nfe_meter_enc=None, b_nfe_meter_enc=None,
                      f_nfe_meter_dec=None, b_nfe_meter_dec=None,
                      batch_time_meter=None, train_loss_meter=None):
    if not os.path.isdir(path):
        os.mkdir(path)
    if is_ode: name = 'ode_' + name
    model_sd = model.state_dict()
    opt_sd = model_opt.optimizer.state_dict()
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_sd,
        'optimizer_state_dict': opt_sd,
        'noam_state_dict': {
            'factor': model_opt.factor,
            'model_size': model_opt.model_size,
            'warmup': model_opt.warmup,
            '_step': model_opt._step,
            '_rate': model_opt._rate
        },
        'val_loss': val_loss,
        'batch_time_avg': batch_time_meter.avg,
        'batch_time_hist': batch_time_meter.hist,
        'train_loss_avg': train_loss_meter.avg,
        'train_loss_hist': train_loss_meter.hist
    }
    
    if is_ode == True:
        checkpoint['f_nfe_enc_avg'] = f_nfe_meter_enc.avg
        checkpoint['f_nfe_enc_hist'] = f_nfe_meter_enc.hist
        
        checkpoint['f_nfe_dec_avg'] = f_nfe_meter_dec.avg
        checkpoint['f_nfe_dec_hist'] = f_nfe_meter_dec.hist
        
        checkpoint['b_nfe_enc_avg'] = b_nfe_meter_enc.avg
        checkpoint['b_nfe_enc_hist'] = b_nfe_meter_enc.hist
        
        checkpoint['b_nfe_dec_avg'] = b_nfe_meter_dec.avg
        checkpoint['b_nfe_dec_hist'] = b_nfe_meter_dec.hist
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_name = '{}{}_{}.tar'.format(name, epoch, timestamp)
    checkpoint_path = os.path.join(path, checkpoint_name)
    torch.save(checkpoint, checkpoint_path)
    print('Epoch {}: Checkpoint saved at {}'.format(epoch, checkpoint_path))
    
def load_checkpoint(model, optimizer, checkpoint_path, is_ode=False, 
                    f_nfe_meter_enc=None, b_nfe_meter_enc=None,
                    f_nfe_meter_dec=None, b_nfe_meter_dec=None,
                    batch_time_meter=None, train_loss_meter=None,
                    load_meters=True, resume_meters=True):
    """
    Parameters:
        model: An nn.Module that corresponds to the model saved in the specified checkpoint
        optimizer: An nn.optim.Optimizer that corresponds to the underlying optimizer saved 
            in the specified checkpoint
        checkpoint_path: As the name suggests
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    noam_state_dict = checkpoint['noam_state_dict']
    model_opt = NoamOpt(noam_state_dict['model_size'], noam_state_dict['factor'], 
                       noam_state_dict['warmup'], optimizer)
    model_opt._step = noam_state_dict['_step']
    model_opt._rate = noam_state_dict['_rate']
    
    if load_meters == True:
        batch_time_meter.avg = checkpoint['batch_time_avg']
        train_loss_meter.avg = checkpoint['train_loss_avg']
        if resume_meters == True: 
            batch_time_meter.val = 0
            train_loss_meter.val = 0
            
        if is_ode == True:
            f_nfe_meter_enc.avg = checkpoint['f_nfe_enc_avg']
            f_nfe_meter_dec.avg = checkpoint['f_nfe_dec_avg']
            b_nfe_meter_enc.avg = checkpoint['b_nfe_enc_avg']
            b_nfe_meter_dec.avg = checkpoint['b_nfe_dec_avg']
            if resume_meters == True:
                f_nfe_meter_enc.val = 0
                f_nfe_meter_dec.val = 0
                b_nfe_meter_enc.val = 0
                b_nfe_meter_dec.val = 0
    
    print('Loaded checkpoint at epoch {}'.format(epoch))
    return epoch, model, model_opt


