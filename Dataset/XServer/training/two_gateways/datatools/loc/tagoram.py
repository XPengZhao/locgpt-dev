# -*- coding: utf-8 -*-
import math
import os
import sys

import numpy as np
import scipy.constants as sconst
from scipy.stats import norm

sys.path.append(os.path.dirname(__file__) + os.sep + '../')

from loc.coor_transform import rigid_transform_3D


class Hologram():
    """Hologram Algorithm Searching Area
    """

    def __init__(self, atn_coor, xRange, yRange, zRange, resolution):
        self.x = np.arange(xRange[0], xRange[1], resolution)
        self.y = np.array(yRange[0])
        self.z = np.arange(zRange[0], zRange[1], resolution)
        self.xlen, self.ylen, self.zlen = self.x.size, self.y.size, self.z.size

        self.phase_theory = self.__area_phase_theory(atn_coor)


    @staticmethod
    def single_fineind(row_ind, col_ind):
        row_fineind = np.arange(row_ind*10, (row_ind+1)*10, dtype=int)
        col_fineind = np.arange(col_ind*10, (col_ind+1)*10, dtype=int)

        col_fineind, row_fineind = np.meshgrid(col_fineind, row_fineind)
        col_fineind = col_fineind.flatten()
        row_fineind = row_fineind.flatten()

        return row_fineind, col_fineind


    @classmethod
    def get_indexfine(cls, rowind_coarse, colind_coarse):
        """get fine search area index
        """
        row_fineinds = np.array([], int)
        col_fineinds = np.array([], int)

        for i in range(rowind_coarse.size):
            row_fineind, col_fineind =  cls.single_fineind(rowind_coarse[i], colind_coarse[i])
            row_fineinds = np.hstack((row_fineinds, row_fineind))
            col_fineinds = np.hstack((col_fineinds, col_fineind))

        return row_fineinds, col_fineinds


    def __area_phase_theory(self, atn_coor):
        """theory phase of searching area
        """
        X,Y,Z = np.meshgrid(self.x, self.y, self.z)

        phase_theory = np.zeros((self.xlen * self.zlen, 16))     # 1000000x16
        X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
        area_abs_coor =  np.vstack((X,Y,Z))# 3xn
        B = np.array([[0,0,0.468], [0,0.156,0], [0,0,0]])
        R,t = rigid_transform_3D(atn_coor, B)
        area_rel_coor = R@area_abs_coor + t                      #相对与阵列坐标系的坐标 3x(100x40)

        filepath =  "loc/near_atn_loc32.csv"
        antenna_loc = np.loadtxt(filepath, delimiter=",")    # 3x16

        lamda = sconst.c / 915e6
        for i in range(area_rel_coor.shape[1]):
            dis = np.linalg.norm(antenna_loc-area_rel_coor[:,i].reshape(3,1), axis=0)
            phase_gt = (dis / lamda * 2*np.pi) % (2*np.pi)   # 1x16 1D
            phase_theory[i] = phase_gt

        return phase_theory


    def hologram_pyramid(self, phase_m):
        """Hologram pyramid algorithm
        """
        pass


    def hologram_coarse(self, phase_m):
        """hologram算法 粗粒度（10cm）
        """
        # phase_t = self.phase_theory        # 1000000 x 16
        # k = phase_t.shape[1]               # 16

        # delta = (phase_t - phase_m.reshape(1,16)) % (2*np.pi)  # 1000000x16  - 1x16

        # cosd = (np.cos(delta)).sum(1)
        # sind = (np.sin(delta)).sum(1)
        # p = np.sqrt(cosd * cosd + sind * sind) / k

        # p = p.reshape(self.xlen, self.zlen)

        # ind = np.unravel_index(np.argmax(p, axis=None), p.shape)
        # x = self.x[ind[0]]
        # z = self.z[ind[1]]
        # y = 1
        # return x,y,z, p

        phase_t = self.phase_theory        # 1000000 x 16
        delta = (phase_t - phase_m.reshape(1,16)) % (2*np.pi)  # 1000000x16  - 1x16
        cosd = (np.cos(delta)).sum(1)
        sind = (np.sin(delta)).sum(1)
        return cosd, sind


    def hologram_fine(self, phase_m, rowind_fine, colind_fine):
        """hologram算法 细粒度（1cm）
        """
        # steer_ind = rowind_fine*self.zlen + colind_fine
        # phase_t = self.phase_theory[steer_ind,:]        # n x 16
        # k = phase_t.shape[1]                          # 16
        # delta = (phase_t - phase_m.reshape(1,16)) % (2*np.pi)  # nx16  - 1x16
        # cosd = (np.cos(delta)).sum(1)
        # sind = (np.sin(delta)).sum(1)
        # p = np.sqrt(cosd * cosd + sind * sind) / k             #nx1
        # p = p.flatten()
        # return p
        steer_ind = rowind_fine*self.zlen + colind_fine
        phase_t = self.phase_theory[steer_ind,:]               # nx16
        delta = (phase_t - phase_m.reshape(1,16)) % (2*np.pi)  # nx16  - 1x16
        cosd = (np.cos(delta)).sum(1)
        sind = (np.sin(delta)).sum(1)
        return cosd, sind



class Tagoram():

    def __init__(self, atn_coor):
        self.x = np.arange(-5, 5, 0.01)
        self.y = np.array([1])
        self.z = np.arange(-5, 5, 0.01)
        self.xlen, self.ylen, self.zlen = self.x.size, self.y.size, self.z.size

        self.phase_theory = self.__area_phase_theory(atn_coor)



    def __area_phase_theory(self, atn_coor):
        """搜索区域理论相位
        """
        X,Y,Z = np.meshgrid(self.x, self.y, self.z)

        phase_theory = np.zeros((self.xlen * self.zlen, 16))     # 1000000x16
        X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
        area_abs_coor =  np.vstack((X,Y,Z))# 3xn
        B = np.array([[0,0,0.48], [0,0.16,0], [0,0,0]])
        R,t = rigid_transform_3D(atn_coor, B)

        area_rel_coor = R@area_abs_coor + t    #相对与阵列坐标系的坐标 3x(100x40)

        filepath =  "loc/near_atn_loc32.csv"
        antenna_loc = np.loadtxt(filepath, delimiter=",")    # 3x16

        lamda = sconst.c / 915e6
        for i in range(area_rel_coor.shape[1]):
            dis = np.linalg.norm(antenna_loc-area_rel_coor[:,i].reshape(3,1), axis=0)
            phase_gt = (dis / lamda * 2*np.pi) % (2*np.pi)   # 1x16 1D
            phase_theory[i] = phase_gt

        return phase_theory


    def tagoram(self, phase_m):
        phase_t = self.phase_theory        # 1000000 x 16
        k = phase_t.shape[1]               # 16

        delta = (phase_t - phase_m.reshape(1,16)) % (2*np.pi)  # 1000000x16  - 1x16

        alpha = 2*(1-norm.cdf(np.abs(delta), 0, 2))

        cosd = (np.cos(delta)).sum(1)
        sind = (np.sin(delta)).sum(1)
        p = np.sqrt(cosd * cosd + sind * sind) / k

        p = p.reshape(self.xlen, self.zlen)

        ind = np.unravel_index(np.argmax(p, axis=None), p.shape)
        x = self.x[ind[0]]
        z = self.z[ind[1]]
        y = 1
        return x,y,z, p


class HologramAoA():
    """Hologram Algorithm Searching AoA space
    """

    def __init__(self):
        filepath =  "loc/near_atn_loc32.csv"
        antenna_loc = np.loadtxt(filepath, delimiter=",")    # 3x16
        x,y = antenna_loc[0,:],antenna_loc[1,:]
        antenna_num = 16                                     # the number of the antenna array element
        atn_polar   = np.zeros((antenna_num,2))              # Polar Coordinate
        for i in range(antenna_num):
            atn_polar[i,0] = math.sqrt(x[i] * x[i] + y[i] * y[i])
            atn_polar[i,1] = math.atan2(y[i], x[i])

        self.theory_phase = self.get_theory_phase(atn_polar)


    def get_theory_phase(self, atn_polar):
        """get theory phase, return (360x90)x16 array
        """
        a_step = 1 * 360
        e_step = 1 * 90
        spacealpha = np.linspace(0, np.pi * 2, a_step)  # 0-2pi
        spacebeta = np.linspace(0, np.pi / 2, e_step)   # 0-pi/2

        alpha,beta = np.meshgrid(spacealpha, spacebeta)
        alpha, beta = alpha.flatten(), beta.flatten()   #alpha[0,1,..0,1..], beta [0,0,..1,1..]
        theta_k = atn_polar[:,1].reshape(16,1)          #alpha 1x(360x90)
        r = atn_polar[:,0].reshape(16,1)
        lamda = sconst.c / 920e6
        theta_t = -2 * math.pi / lamda * r * np.cos(alpha - theta_k) * np.cos(beta)
        print(theta_t.shape)
        # np.savetxt('theta.txt', ((alpha - theta_k)*180/np.pi).T[:361,:], fmt='%.04f')
        # np.savetxt('theta_t.txt', theta_t.T[0:360,:], fmt='%.04f')

        return theta_t.T

    def get_theory_phase_nonuniform(self, atn_polar):
        """get theory phase by uniform sampling cos beta instead of beta
           return (360x90)x16 array
        """
        a_step = 2 * 360
        spacealpha = np.linspace(0, np.pi * 2, a_step)  # 0-2pi

        # cosbeta = np.linspace(0,1,180)[::-1]
        # spacebeta = np.arccos(cosbeta)                     # 0-pi/2
        cosbeta = np.linspace(0,np.sqrt(2)/2,45)[::-1]
        spacebeta = np.arccos(cosbeta)                     # 0-pi/2

        alpha,beta = np.meshgrid(spacealpha, spacebeta)
        alpha, beta = alpha.flatten(), beta.flatten()   #alpha[0,1,..0,1..], beta [0,0,..1,1..]
        theta_k = atn_polar[:,1].reshape(16,1)          #alpha 1x(360x90)
        r = atn_polar[:,0].reshape(16,1)
        lamda = sconst.c / 915e6
        theta_t = -2 * math.pi / lamda * r * np.cos(alpha - theta_k) * np.cos(beta)

        return theta_t.T


    def get_aoa_heatmap(self, phase_m):
        """got aoa heatmap
        """
        delta_phase = self.theory_phase - phase_m.reshape(1,16)     #(720x180)x16 - 1x16
        cosd = (np.cos(delta_phase)).sum(1)
        sind = (np.sin(delta_phase)).sum(1)
        p = np.sqrt(cosd * cosd + sind * sind) / 16
        p = p.reshape(90,360)
        return p





if __name__ == '__main__':
    HologramAoA()
