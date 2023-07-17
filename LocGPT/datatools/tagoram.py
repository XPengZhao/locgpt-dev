# -*- coding: utf-8 -*-
import math
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch as tr
import numpy as np
import scipy.constants as sconst


class Bartlett():
    """Bartlett Algorithm Searching AoA space
    """

    def __init__(self):
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

        self.theory_phase = self.get_theory_phase(atn_polar)

    # def get_theory_phase(self, atn_polar):
    #     """get theory phase, return (360x90)x16 array
    #     """
    #     a_step = 1 * 360
    #     e_step = 1 * 90
    #     spacealpha = tr.linspace(0, np.pi * 2 * (1 - 1 / 360), a_step)  # 0-2pi
    #     spacebeta = tr.linspace(0, np.pi / 2 * (1 - 1 / 90), e_step)  # 0-pi/2

    #     alpha = spacealpha.expand(e_step, -1).flatten()  # alpha[0,1,..0,1..]
    #     beta = spacebeta.expand(a_step, -1).permute(1, 0).flatten()  # beta[0,0,..1,1..]

    #     theta_k = atn_polar[:, 1].view(16, 1)
    #     r = atn_polar[:, 0].view(16, 1)
    #     lamda = sconst.c / 920e6
    #     theta_t = -2 * math.pi / lamda * r * np.cos(alpha - theta_k) * np.cos(beta)  # (16, 360x90)
    #     print(theta_t.shape)

    #     return theta_t.T


    def get_theory_phase(self, atn_polar):
        """get theory phase, return (360x90)x16 array
        """
        a_step = 1 * 224
        e_step = 1 * 224
        spacealpha = tr.linspace(0, np.pi * 2 * (1 - 1 / 224), a_step)  # 0-2pi
        spacebeta = tr.linspace(0, np.pi / 2 * (1 - 1 / 224), e_step)  # 0-pi/2

        alpha = spacealpha.expand(e_step, -1).flatten()  # alpha[0,1,..0,1..]
        beta = spacebeta.expand(a_step, -1).permute(1, 0).flatten()  # beta[0,0,..1,1..]

        theta_k = atn_polar[:, 1].view(16, 1)
        r = atn_polar[:, 0].view(16, 1)
        lamda = sconst.c / 920e6
        theta_t = -2 * math.pi / lamda * r * np.cos(alpha - theta_k) * np.cos(beta)  # (16, 360x90)
        print(theta_t.shape)

        return theta_t.T

    def get_aoa_heatmap(self, phase_m):
        """got aoa heatmap
        """
        delta_phase = self.theory_phase - phase_m.reshape(1, 16)  # (360x90,16) - 1x16
        cosd = (tr.cos(delta_phase)).sum(1)
        sind = (tr.sin(delta_phase)).sum(1)
        p = tr.sqrt(cosd * cosd + sind * sind) / 16
        p = p.view(224,224)
        p = p.numpy()
        return p


if __name__ == '__main__':
    blt = Bartlett()
    for file in os.listdir(path='./data'):
        df = pd.read_csv('./data/' + file)
        df = df.sample(frac=1).reset_index(drop=True)
        l = len(df)
        dir_train = './heatmap_train_' + file[:-4]
        dir_test = './heatmap_test_' + file[:-4]
        if not os.path.exists(dir_train):
            os.makedirs(dir_train)
        if not os.path.exists(dir_test):
            os.makedirs(dir_test)
        train_label = pd.DataFrame(columns=['id', 'x', 'y', 'z', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9'])
        test_label = pd.DataFrame(columns=['id', 'x', 'y', 'z', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9'])
        print('begin: '+file)
        for i in range(l):
            phase1 = df.iloc[i, 4+9:35+9:2].values
            phase2 = df.iloc[i, 4+32+9:35+32+9:2].values
            phase3 = df.iloc[i, 4+32*2+9:35+32*2+9:2].values
            if not np.all(phase1 == 0):
                heatmap1 = blt.get_aoa_heatmap(phase1)
            else:
                continue
            if not np.all(phase2 == 0):
                heatmap2 = blt.get_aoa_heatmap(phase2)
            else:
                continue
            if not np.all(phase3 == 0):
                heatmap3 = blt.get_aoa_heatmap(phase3)
            else:
                continue
            rgb = np.stack([heatmap1, heatmap2, heatmap3], axis=2)
            if i < int(l*0.8):
                path = f'{dir_train}/{i}.png'
                new_row = {'id': path, 'x': df.iloc[i, 0+9], 'y': df.iloc[i, 1+9], 'z': df.iloc[i, 2+9], 'p1': df.iloc[i, 0], 'p2': df.iloc[i, 1], 'p3': df.iloc[i, 2], 'p4': df.iloc[i, 3], 'p5': df.iloc[i, 4], 'p6': df.iloc[i, 5], 'p7': df.iloc[i, 6], 'p8': df.iloc[i, 7], 'p9': df.iloc[i, 8]}
                train_label = train_label.append(new_row, ignore_index=True)
            else:
                path = f'{dir_test}/{i}.png'
                new_row = {'id': path, 'x': df.iloc[i, 0+9], 'y': df.iloc[i, 1+9], 'z': df.iloc[i, 2+9], 'p1': df.iloc[i, 0], 'p2': df.iloc[i, 1], 'p3': df.iloc[i, 2], 'p4': df.iloc[i, 3], 'p5': df.iloc[i, 4], 'p6': df.iloc[i, 5], 'p7': df.iloc[i, 6], 'p8': df.iloc[i, 7], 'p9': df.iloc[i, 8]}
                test_label = test_label.append(new_row, ignore_index=True)
            plt.imsave(path, rgb)
        train_label.to_csv('train_'+file, index=False, header=False)
        test_label.to_csv('test_'+file, index=False, header=False)
        df1 = pd.read_csv('train_'+file)
        df2 = pd.read_csv('test_'+file)
        df1.insert(1, 's', [0. for i in range(len(df1))])
        df2.insert(1, 's', [0. for i in range(len(df2))])
        df1.to_csv('train_'+file, index=False, header=False)
        df2.to_csv('test_'+file, index=False, header=False)
        print(l, len(df1), len(df2))
        print('done: '+file)



