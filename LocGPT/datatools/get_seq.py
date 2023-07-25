import pandas as pd
import numpy as np
from einops import rearrange
import torch
from gen_spt import Bartlett
import random


# Set the random seed
np.random.seed(0)
random.seed(0)


def get_seq_index(num_seq, seq_len=10, max_step=3):
    """
    return
    ----------
    seqs: [num_seq, 10]
    """
    ind = [i for i in range(num_seq*seq_len)]
    seqs = []

    for i in range(num_seq):
        seq = [ind.pop(0)]
        old_pointer = 0
        for j in range(seq_len-1):
            step = random.randint(0,max_step-1)
            new_pointer = old_pointer + step
            if new_pointer > len(ind)-1:
                new_pointer = 0
                old_pointer = 0
            if ind[new_pointer] - seq[-1] <= max_step or new_pointer == old_pointer:
                seq.append(ind.pop(new_pointer))
            elif new_pointer > old_pointer:
                while new_pointer > old_pointer:
                    new_pointer = new_pointer - 1
                    if ind[new_pointer] - seq[-1] <= max_step:
                        seq.append(ind.pop(new_pointer))
                        break
                    elif new_pointer == old_pointer:
                        seq.append(ind.pop(new_pointer))
                        break
            old_pointer = new_pointer
            seq.sort()
        seqs.append(seq)

    return np.array(seqs)



if __name__ == "__main__":

    seq_len = 1
    scene = "02"
    data_path = f"data/mcbench/data-s{scene}.csv"
    save_train_path = f"data/mcbench/train_data-s{scene}-seq{seq_len}.t"
    save_test_path = f"data/mcbench/test_data-s{scene}-seq{seq_len}.t"


    blt = Bartlett()
    df = pd.read_csv(data_path)

    data = df.values
    N, dim = data.shape
    data = data[:N//seq_len*seq_len]
    n_seq = N//seq_len
    ind = get_seq_index(n_seq, seq_len)  # [n_seq, 10]
    data_seq = data[ind]    # [n_seq, 10, dim]

    np.random.shuffle(data_seq)
    data_seq = torch.from_numpy(data_seq)

    data_len, b, n = data_seq.shape

    # [timestamp_1 + area_1 + pos_3 + spt_324]
    data_all = torch.zeros((data_len, b, 1+1+3+3*9*36))
    data_all[...,0] = data_seq[..., 0]
    data_all[...,2:5] = data_seq[..., 1:4]

    for i in range(0, len(data_seq)):
        batch_data = data_seq[i]  # (10, 4+32*3)
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

    torch.save(train_data, save_train_path)
    torch.save(test_data, save_test_path)