# -*- coding: utf-8 -*-
import math
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch as tr
import numpy as np
import scipy.constants as sconst
import matplotlib.image as plm
from einops import rearrange

class Bartlett():
    """Bartlett Algorithm Searching AoA space
    """

    def __init__(self, device=None):

        if device is None:
            device = 'cuda' if tr.cuda.is_available() else 'cpu'
        self.device = device

        antenna_loc = [[0, 0.16, 0.32, 0.48, 0, 0.16, 0.32, 0.48, 0, 0.16, 0.32, 0.48, 0, 0.16, 0.32, 0.48],
                       [0.48, 0.48, 0.48, 0.48, 0.32, 0.32, 0.32, 0.32, 0.16, 0.16, 0.16, 0.16, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        antenna_loc = tr.tensor(antenna_loc)
        x, y = antenna_loc[0, :], antenna_loc[1, :]
        antenna_num = 16  # the number of the antenna array element
        atn_polar = tr.zeros((antenna_num, 2))  # Polar Coordinate
        for i in range(antenna_num):
            atn_polar[i, 0] = math.sqrt(x[i] * x[i] + y[i] * y[i])
            atn_polar[i, 1] = math.atan2(y[i], x[i])

        self.theory_phase = self.get_theory_phase(atn_polar).to(device)


    def get_theory_phase(self, atn_polar):
        """get theory phase, return (360x90)x16 array
        """
        a_step = 1 * 36
        e_step = 1 * 9
        spacealpha = tr.linspace(0, np.pi * 2 * (1 - 1 / 36), a_step)  # 0-2pi
        spacebeta = tr.linspace(0, np.pi / 2 * (1 - 1 / 9), e_step)  # 0-pi/2

        alpha = spacealpha.expand(e_step, -1).flatten()  # alpha[0,1,..0,1..]
        beta = spacebeta.expand(a_step, -1).permute(1, 0).flatten()  # beta[0,0,..1,1..]

        theta_k = atn_polar[:, 1].view(16, 1)
        r = atn_polar[:, 0].view(16, 1)
        lamda = sconst.c / 920e6
        theta_t = -2 * math.pi / lamda * r * np.cos(alpha - theta_k) * np.cos(beta)  # (16, 360x90)

        return theta_t.T

    def get_aoa_heatmap(self, phase_m):
        """got aoa heatmap
        """
        delta_phase = self.theory_phase - phase_m.reshape(1, 16)  # (36x9,16) - 1x16
        cosd = (tr.cos(delta_phase)).sum(1)
        sind = (tr.sin(delta_phase)).sum(1)
        p = tr.sqrt(cosd * cosd + sind * sind) / 16
        p = p.view(9,36)
        return p


    def get_aoa_heatmap(self, phase_m):
        """got aoa heatmap"""
        phase_m = tr.tensor(phase_m).to(self.device)
        N = phase_m.shape[0]  # batch size
        delta_phase = self.theory_phase.unsqueeze(0).repeat(N, 1, 1) - phase_m.view(N, 1, 16)  # add batch dimension and repeat the theory phase for each instance in the batch
        cosd = (tr.cos(delta_phase)).sum(2)  # sum over antenna dimension
        sind = (tr.sin(delta_phase)).sum(2)  # sum over antenna dimension
        p = tr.sqrt(cosd * cosd + sind * sind) / 16  # calculate magnitude
        p = p.view(N, 9, 36)  # reshape into (N, 9, 36) tensor
        return p


if __name__ == '__main__':
    blt = Bartlett()
    df_train = pd.read_csv("data/mcbench/train_data-s02.csv")
    df_test = pd.read_csv("data/mcbench/test_data-s02.csv")

    phase1 = df_train.iloc[:, 13:44:2].values  # [n, 16]
    phase2 = df_train.iloc[:, 13+32:44+32:2].values
    phase3 = df_train.iloc[:, 13+32*2:44+32*2:2].values

    batch_size = 1024
    heatmap_train = tr.zeros(len(phase1), 3, 9, 36)
    for i in range(0, len(phase1), batch_size):
        heatmap1 = blt.get_aoa_heatmap(phase1[i:i+batch_size, :]).detach().cpu()  # [B, spt_dim]
        heatmap2 = blt.get_aoa_heatmap(phase2[i:i+batch_size, :]).detach().cpu()
        heatmap3 = blt.get_aoa_heatmap(phase3[i:i+batch_size, :]).detach().cpu()
        heatmap_train[i:i+batch_size, 0, ...] = heatmap1
        heatmap_train[i:i+batch_size, 1, ...] = heatmap2
        heatmap_train[i:i+batch_size, 2, ...] = heatmap3
    heatmap_train = rearrange(heatmap_train, 'b c h w -> b c (h w)')
    print("heatmap training shape", heatmap_train.shape)

    tr.save(heatmap_train, "train_spt-s02.t")



