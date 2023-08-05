# -*- coding: utf-8 -*-
"""gererate wifi dataset
"""
import math

import h5py
import matplotlib.image as plm
import numpy as np
import scipy.constants as sconst
import torch
from einops import rearrange


# Set the random seed
# np.random.seed(0)
torch.manual_seed(0)

class Bartlett():
    """Bartlett Algorithm Searching AoA space
    """

    def __init__(self, atn_loc, freq, device=None):
        """
        Params:
        ------------
        freq: tensor, shape [n_freq]
        """

        self.a_step = 1 * 36
        self.e_step = 1 * 9

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        atn_loc = torch.tensor(atn_loc)
        x, y = atn_loc[0, :], atn_loc[1, :]
        self.atn_num = atn_loc.shape[-1]            # the number of the antenna array element
        self.freq_num = freq.shape[-1]              # the number of the frequency
        self.atn_polar = torch.zeros((self.atn_num, 2))  # Polar Coordinate
        for i in range(self.atn_num):
            self.atn_polar[i, 0] = math.sqrt(x[i] * x[i] + y[i] * y[i])
            self.atn_polar[i, 1] = math.atan2(y[i], x[i])

        self.theory_phase = self.get_theory_phase(freq).to(device)    #[n_freq, n_atn, 36x9]


    def get_theory_phase(self, freq):
        """get theory phase, return [n_freq, n_atn, 360x90] tensor
        """

        spacealpha = torch.linspace(0, np.pi * 2 * (1 - 1 / self.a_step), self.a_step)  # 0-2pi
        spacebeta = torch.linspace(0, np.pi / 2 * (1 - 1 / self.e_step), self.e_step)  # 0-pi/2

        alpha = spacealpha.expand(self.e_step, -1).flatten()  # alpha[0,1,..0,1..]
        beta = spacebeta.expand(self.a_step, -1).permute(1, 0).flatten()  # beta[0,0,..1,1..]

        freq = torch.tensor(freq)
        N = freq.shape[0]  # batch size
        freq = freq.view(N, 1, 1)  # add two dimensions for antenna and angle

        theta_k = self.atn_polar[:, 1].view(1, self.atn_num, 1)  # add batch dimension
        r = self.atn_polar[:, 0].view(1, self.atn_num, 1)  # add batch dimension
        lamda = sconst.c / freq
        theta_t = -2 * math.pi / lamda * r * np.cos(alpha - theta_k) * np.cos(beta)  # (16, 360x90)

        return theta_t  # [n_freq, n_atn, 360x90]


    def get_aoa_heatmap(self, phase_m):
        """got aoa heatmap

        Params:
        ------------
        phase_m: ndarray, shape [batch_size, n_freq, n_atn]

        """
        batch_size = phase_m.shape[0]  # batch size

        #expand to [batch_size, n_freq, n_atn, 36x9]
        theory_phase = self.theory_phase.unsqueeze(0).expand(batch_size, -1, -1, -1)
        phase_m = torch.tensor(phase_m).to(self.device)
        phase_m = phase_m.view(batch_size,self.freq_num, self.atn_num, 1)  # reshape into [batch_size, n_freq, n_atn, 1]
        delta_phase = theory_phase - phase_m  # calculate delta phase

        cosd = (torch.cos(delta_phase)).sum(2)  # sum over antenna dimension [batch_size, n_freq, 36x9]
        sind = (torch.sin(delta_phase)).sum(2)
        p = torch.sqrt(cosd * cosd + sind * sind) / self.atn_num  # calculate magnitude
        p = p.sum(1) / self.freq_num   # sum over frequency dimension  [batch_size, 36x9]
        p = p.view(batch_size, self.e_step, self.a_step)  # reshape into (N, 9, 36) tensor
        return p



if __name__ == '__main__':

    scene = "July22_2_ref"
    filepath = f'data/wifi/channels_{scene}.mat'
    batch_size = 128
    atn_num = 4

    with h5py.File(filepath, 'r') as f:
        print(list(f.keys()))
        channels_wo_offset = np.array(f['channels_wo_offset'])  #[ap, atn, freq, sample]
        channels = channels_wo_offset.transpose(3, 2, 1, 0)
        data_len = channels.shape[0]
        labels = np.array(f['labels']).transpose(1, 0)
        labels = torch.tensor(labels).float()
        atn_sep = np.array(f['opt']['ant_sep']).item()
        freq = np.array(f['opt']['freq']).flatten()
        # freq = np.median(freq)

        ap = f['ap']
        ap_num = len(ap)
        gateway_pos = np.zeros((ap_num, 3))
        for i, col in enumerate(ap):
            ap_pos = f[col[0]][:]
            ap_pos = np.mean(ap_pos, axis=-1)
            gateway_pos[i, :2] = ap_pos
        print("gateway_pos: ", gateway_pos)

    # [timestamp_1 + area_1 + pos_3 + spt_324]
    data_all = torch.zeros((data_len, 1+1+3+ap_num*9*36))
    data_all[...,2:4] = labels


    ## linear antenna array
    atn_loc_x = np.linspace(0, 3 * atn_sep, atn_num)
    atn_loc_y = np.zeros(atn_num)
    atn_loc_z = np.zeros(atn_num)
    atn_loc = np.vstack((atn_loc_x, atn_loc_y, atn_loc_z))
    # print("atn_loc: ", atn_loc)
    worker = Bartlett(atn_loc, freq)
    for i in range(0, len(channels), batch_size):
        csi = channels[i:i+batch_size]  #[batch_size, freq, atn, iq]
        IQ = csi['real'] + 1j * csi['imag']
        phase = np.angle(IQ)
        heatmap1 = worker.get_aoa_heatmap(phase[...,0]).detach().cpu().unsqueeze(1)  # [B, 1, 9, 36]
        heatmap2 = worker.get_aoa_heatmap(phase[...,1]).detach().cpu().unsqueeze(1)
        heatmap3 = worker.get_aoa_heatmap(phase[...,2]).detach().cpu().unsqueeze(1)
        # heatmap4 = worker.get_aoa_heatmap(phase[...,3]).detach().cpu().unsqueeze(1)
        heatmap = torch.concat((heatmap1, heatmap2, heatmap3), dim=1)  # [B, 4, 9, 36]
        heatmap = rearrange(heatmap, 'b c h w -> b (c h w)')  # [B, 4*9*36]
        data_all[i:i+batch_size, 5:] = heatmap


        # for j,heatmap in enumerate(heatmap1):
        #     if j % 1000 == 0:
        #         plm.imsave(f"gateway1/t{i+j}.png", heatmap.cpu().numpy())


    perm = torch.randperm(data_all.size(0))
    data_all = data_all[perm]
    data_all = data_all.unsqueeze(1)  # [N, 1, 1+1+3+4*9*36] for seq = 1

    train_len = int(len(data_all) * 0.8)
    train_data = data_all[0:train_len]
    test_data = data_all[train_len:]

    print("len train_data", len(train_data))
    print("len train_data", len(test_data))


    save_train_path = f"data/wifi/train_data-s{scene}-seq1.pt"
    save_test_path = f"data/wifi/test_data-s{scene}-seq1.pt"
    torch.save(train_data, save_train_path)
    torch.save(test_data, save_test_path)