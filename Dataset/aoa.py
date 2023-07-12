# -*- coding: utf-8 -*-
"""MUSIC Algorithm
"""
import numpy as np
import matplotlib.pyplot as plt

import config


class Music():

    """music implementation, """
    def __init__(self, radius=0.1, array_num=16, freq=2.45e9):
        self.radius = radius
        self.array_num = array_num
        self.freq = freq
        self.wavelength = 3e8 / self.freq     # wavelength
        self.arrayCoor = self.getArrayCoor()  # 阵元坐标



    def getArrayCoor(self):
        """get the coordinate of element [[x11,y11], [x12, y12], ...., [x44, y44]]
        """
        array_coor = np.zeros((self.array_num, 2))
        if config.polar_mode:
            for ind in range(16):
                array_coor[ind, 0] = -0.054 + (ind % 4) * 0.036
                array_coor[ind, 1] = 0.054 - (ind // 4) * 0.036
        else:
            angles = np.arange(0, np.pi * 2, np.pi * 2 / self.array_num)
            for ind, angle in enumerate(angles):
                array_coor[ind, 0] = self.radius * np.cos(angle)
                array_coor[ind, 1] = self.radius * np.sin(angle)


        return array_coor

    def channel2frequecy(self,channel):
        logical_to_physical_channel = (
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 0, 12, 39)
        return 2402000000 + 2000000 * logical_to_physical_channel[channel]

    def arrayVector(self, theta, phi):
        """计算阵列流形矩阵
        """
        disdiff = self.arrayCoor[:,0]*np.cos(theta)*np.cos(phi) + self.arrayCoor[:, 1]*np.sin(theta)*np.cos(phi)
        arrayVector = np.exp(-2j * np.pi * disdiff / self.wavelength)

        return arrayVector



    def music(self, phase,channel):
        covMat = phase @ phase.conj().T    # calculate cov matrix
        spaceAzimuth, spaceElevation = np.linspace(0, 2*np.pi, 360), np.linspace(0, np.pi/2, 90)

        U, sigma, _ = np.linalg.svd(covMat)
        k = next(i for i, s in enumerate(sigma[1:], 1) if sigma[0] / s > 10)
        Qn = U[:, k:]

        theta, phi = np.repeat(np.linspace(0, 2*np.pi, 360), 90), np.tile(np.linspace(0, np.pi/2, 90), 360)
        x, y  = np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi)
        angs = np.dstack((x, y)).reshape(360*90, 2, 1)


        arrVec = np.exp(-2j * np.pi * self.arrayCoor @ angs / 3e8 * self.channel2frequecy(channel))
        temp = arrVec.conj().reshape(360*90, 1, 16) @ Qn
        pspectrum = (1 / abs(temp @ temp.conj().reshape(360*90, -1, 1))).reshape(360, 90)

        ind = np.unravel_index(np.argmax(pspectrum), pspectrum.shape)
        return pspectrum, spaceAzimuth[ind[0]], spaceElevation[ind[1]],pspectrum[ind[0],ind[1]]



    def pseudoSingal(self, K, SNR, azimuth, elevation, M):
        """
        生成模拟的到达信号X(k)
        ----------

        Parameters
        ----------
        N 阵元数量; M 信源数量; K 快拍数; thetas 模拟到达角(DoA);
        array 阵列位置; wavelength 波长

        Returns
        -----------
        """
        S = np.random.randn(M, K) + 1j * np.random.randn(M, K)  # 入射信号矩阵S（M*K维）
        arrayMatrix = np.zeros((16, M)) + 1j * np.zeros((16, M))  # 阵列流形矩阵A（N*M维）

        for i in range(M):
            a = self.arrayVector(azimuth[i], elevation[i])
            arrayMatrix[:, i] = a

        X = arrayMatrix @ S

        X += np.sqrt(0.5 / SNR) * (np.random.randn(16, K) + np.random.randn(16, K) * 1j)

        return X




if __name__ == "__main__":


    music_worker = Music()
    az = [np.pi / 4]
    el = [np.pi / 3]
    signal = music_worker.pseudoSingal(1, 15, az, el, 1)
    phase = np.angle(signal)
    phase = np.array([3.59733638553222,3.48824457806239,3.51088373754496,3.18216728133891,5.64354192871613,4.68587789005535,5.47666714818062,5.87618535648459,1.49728354729333,4.24510852784118,5.51293104727126,1.36334216434960,0.513851570753218,1.13597008423401,0.829230157395744,5.38355265175846])
    phase = phase.reshape((16, 1))
    signal = np.exp(1j * phase)
    psp, azimuth, elevation = music_worker.music(signal,2)
    print(azimuth *180/ np.pi, elevation*180/np.pi)
    plt.figure()
    plt.imshow(psp)
    plt.show()