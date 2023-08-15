import torch as tr
import math
import scipy.constants as sconst
import numpy as np


class Bartlett():
    """Bartlett Algorithm Searching AoA space
    """

    def __init__(self, device=None):

        self.a_step = 1 * 360
        self.e_step = 1 * 90

        if device is None:
            device = 'cuda' if tr.cuda.is_available() else 'cpu'
        self.device = device

        antenna_loc = [[0.105, 0.105, 0.105, 0.105, 0.07, 0.07, 0.07, 0.07, 0.035, 0.035, 0.035, 0.035, 0, 0, 0, 0],
                       [0.105, 0.07, 0.035, 0, 0.105, 0.07, 0.035, 0, 0.105, 0.07, 0.035, 0, 0.105, 0.07, 0.035, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        antenna_loc = tr.tensor(antenna_loc)
        x, y = antenna_loc[0, :], antenna_loc[1, :]
        antenna_num = 16  # the number of the antenna array element
        self.atn_polar = tr.zeros((antenna_num, 2))  # Polar Coordinate
        for i in range(antenna_num):
            self.atn_polar[i, 0] = math.sqrt(x[i] * x[i] + y[i] * y[i])
            self.atn_polar[i, 1] = math.atan2(y[i], x[i])


    def get_theory_phase(self, freq):
        """get theory phase, return (360x90)x16 array
        """

        spacealpha = tr.linspace(0, np.pi * 2 * (1 - 1 / self.a_step), self.a_step)  # 0-2pi
        spacebeta = tr.linspace(0, np.pi / 2 * (1 - 1 / self.e_step), self.e_step)  # 0-pi/2

        alpha = spacealpha.expand(self.e_step, -1).flatten()  # alpha[0,1,..0,1..]
        beta = spacebeta.expand(self.a_step, -1).permute(1, 0).flatten()  # beta[0,0,..1,1..]

        N = freq.shape[0]  # batch size
        freq = freq.view(N, 1, 1)  # add two dimensions for antenna and angle

        theta_k = self.atn_polar[:, 1].view(1, 16, 1)  # add batch dimension
        r = self.atn_polar[:, 0].view(1, 16, 1)  # add batch dimension
        lamda = sconst.c / freq
        theta_t = -2 * math.pi / lamda * r * np.cos(alpha - theta_k) * np.cos(beta)  # (16, 360x90)

        return theta_t

    def get_aoa_heatmap_single(self, phase_m):
        """got aoa heatmap
        """
        delta_phase = self.theory_phase - phase_m.reshape(1, 16)  # (36x9,16) - 1x16
        cosd = (tr.cos(delta_phase)).sum(1)
        sind = (tr.sin(delta_phase)).sum(1)
        p = tr.sqrt(cosd * cosd + sind * sind) / 16
        p = p.view(self.e_step, self.a_step)
        return p


    def get_aoa_heatmap(self, phase_m, freq):
        """got aoa heatmap"""
        theory_phase = self.get_theory_phase(freq).to(self.device)

        phase_m = tr.tensor(phase_m).to(self.device)
        N = phase_m.shape[0]  # batch size
        phase_m = phase_m.view(N, 16, 1)  # reshape into (N, 16, 1) tensor
        delta_phase = theory_phase - phase_m  # calculate delta phase
        cosd = (tr.cos(delta_phase)).sum(1)  # sum over antenna dimension
        sind = (tr.sin(delta_phase)).sum(1)  # sum over antenna dimension
        p = tr.sqrt(cosd * cosd + sind * sind) / 16  # calculate magnitude
        p = p.view(N, self.e_step, self.a_step)  # reshape into (N, 9, 36) tensor
        return p