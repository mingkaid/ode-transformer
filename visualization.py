"""
Visualize each ??? of the ODE encoder and decoder
"""

import torch
import seaborn

model, SRC, TGT = torch.load("checkpoint.pt")


model.eval()
# we can change this string
sent = "▁The ▁log ▁file ▁can ▁be ▁sent ▁secret ly ▁with ▁email ▁or ▁FTP ▁to ▁a ▁specified ▁receiver".split()
src = torch.LongTensor([[SRC.stoi[w] for w in sent]])
src = Variable(src)
src_mask = (src != SRC.stoi["<blank>"]).unsqueeze(-2)
out = greedy_decode(model, src, src_mask, 
                    max_len=60, start_symbol=TGT.stoi["<s>"])
print("Translation:", end="\t")
trans = "<s> "
for i in range(1, out.size(1)):
    sym = TGT.itos[out[0, i]]
    if sym == "</s>": break
    trans += sym + " "
print(trans)



tgt_sent = trans.split()
def draw(data, x, y, ax):
    seaborn.heatmap(data, 
                    xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, 
                    cbar=False, ax=ax)

    
# Plot heatmap for ODE Encoder
print("ODE Encoder")
fig, axs = plt.subplots(1,4, figsize=(20, 10))
for h in range(4):
    draw(model.encoder.ode_layer.self_attn.attn[0, h].data, 
        sent, sent if h ==0 else [], ax=axs[h])
plt.show()
    
# Plot heatmap for ODE Decoder
print("ODE Decoder")
fig, axs = plt.subplots(1,4, figsize=(20, 10))
for h in range(4):
    draw(model.decoder.ode_layer.self_attn.attn[0, h].data[:len(tgt_sent), :len(tgt_sent)], 
        tgt_sent, tgt_sent if h ==0 else [], ax=axs[h])
plt.show()

print("ODE Decoder with Reference to SRC")
fig, axs = plt.subplots(1,4, figsize=(20, 10))
for h in range(4):
    draw(model.decoder.ode_layer.self_attn.attn[0, h].data[:len(tgt_sent), :len(sent)], 
        sent, tgt_sent if h ==0 else [], ax=axs[h])
plt.show()






