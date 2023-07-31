import pandas as pd
import numpy as np
from einops import rearrange
import torch
from bartlett import Bartlett
import random
import matplotlib.image as plm



# Set the random seed
np.random.seed(0)
random.seed(0)



if __name__ == "__main__":

    seq_len = 1
    data_path = "data/ble-exp2/pq504_exp2_merge.csv"

    blt = Bartlett()
    df = pd.read_csv(data_path)

    data = df.values
    data = torch.from_numpy(data)
    N, dim = data.shape
    batch_size = 10

    for i in range(0, len(data), batch_size):
        batch_data = data[i:i+batch_size] # [B, 5+16*4]
        batch_freq = batch_data[:, 1]
        heatmap1 = blt.get_aoa_heatmap(batch_data[:, 6:6+16], batch_freq).detach().cpu()  # [B, 90, 360]
        # heatmap2 = blt.get_aoa_heatmap(batch_data[:, 6+16:6+32], batch_freq).detach().cpu()
        # heatmap3 = blt.get_aoa_heatmap(batch_data[:, 6+32:6+48], batch_freq).detach().cpu()
        # heatmap4 = blt.get_aoa_heatmap(batch_data[:, 6+48:6+64], batch_freq).detach().cpu()

        for j,heatmap in enumerate(heatmap1):
            plm.imsave(f"gateway1/t{i+j}.png", heatmap.numpy())

                # figure = np.zeros((9,36,3))
        # figure[:,:,0] = heatmap1[0,0].numpy()
        # plm.imsave(f"test{i}.png", figure)