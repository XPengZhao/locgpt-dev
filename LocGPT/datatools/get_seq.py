import pandas as pd
import numpy as np
from einops import rearrange
import torch
from gen_spt import Bartlett

# Set the random seed
np.random.seed(0)

blt = Bartlett()
df = pd.read_csv("data/transformer-test/data-s02.csv")

data = df.values
N, dim = data.shape
data = data[:N//10*10]

data_seq = rearrange(data, '(m b) n -> m b n', b=10)
np.random.shuffle(data_seq)
data_seq = torch.from_numpy(data_seq)

data_len, b, n = data_seq.shape

# [timestamp_1 + area_1 + pos_3 + spt_324]
data_all = torch.zeros((data_len, b, 1+1+3+3*9*36))
data_all[...,0] = data_seq[..., 0]
data_all[...,2:5] = data_seq[..., 1:4]

for i in range(0, len(data_seq)):
    batch_data = data_seq[i]  # (10, 4+32)
    heatmap1 = blt.get_aoa_heatmap(batch_data[:, 5:5+32:2]).detach().cpu().unsqueeze(1)  # [B, 1, 9, 36]
    heatmap2 = blt.get_aoa_heatmap(batch_data[:, 5+32:5+64:2]).detach().cpu().unsqueeze(1)
    heatmap3 = blt.get_aoa_heatmap(batch_data[:, 5+64:5+96:2]).detach().cpu().unsqueeze(1)
    heatmap = torch.concat((heatmap1, heatmap2, heatmap3), dim=1)  # [B, 3, 9, 36]
    heatmap = rearrange(heatmap, 'b c h w -> b (c h w)')  # [10, 3*9*36]
    data_all[i,:,5:] = heatmap

train_len = int(len(data_all) * 0.8)
train_data = data_all[0:train_len]
test_data = data_all[train_len:]

print("len train_data", len(train_data))
print("len train_data", len(test_data))

torch.save(train_data, "train_data-s02.t")
torch.save(test_data, "test_data-s02.t")