# -*- coding: utf-8 -*-
"""MUSIC Algorithm
"""
import numpy as np
import matplotlib.pyplot as plt

class AOA_1D():

    """music implementation, """
    def __init__(self, radius=0.10, array_num=16, freq=2.4e9):
        self.radius = radius
        self.array_num = array_num
        self.freq = freq
        self.wavelength = 3e8 / self.freq     # wavelength
        self.arrayCoor = self.getArrayCoor()  # 阵元坐标
        self.phase_t = self.phase_theory()



    def getArrayCoor(self):
        """get the coordinate of element [[x11,y11], [x12, y12], ...., [x44, y44]]
        """
        array_coor = np.zeros((self.array_num, 2))
        angles = np.arange(0, np.pi *2, np.pi*2/self.array_num)
        for ind, angle in enumerate(angles):
            array_coor[ind, 0] = self.radius * np.cos(angle)
            array_coor[ind, 1] = self.radius * np.sin(angle)
        return array_coor



    def arrayVector(self, theta):
        """计算阵列流形矩阵
        """
        disdiff = self.arrayCoor[:,0]*np.cos(theta) + self.arrayCoor[:, 1]*np.sin(theta)
        arrayVector = np.exp(-2j * np.pi * disdiff / self.wavelength)

        return arrayVector

    def phase_theory(self):  # 1XN
        phase_t = np.zeros([self.array_num,360])
        theta = np.linspace(0, 2 * np.pi - np.pi / 360, 360)
        angles = np.arange(0, np.pi * 2, np.pi * 2 / self.array_num)
        for i in range(self.array_num):
            for j in range(360):
                phase_t[i,j] = -2*np.pi/self.wavelength*self.radius*np.cos(theta[j]-angles[i])
        return phase_t

    def p0(self,phase):
        p = np.zeros(360)
        for i in range(360):
            delta = phase - np.reshape(self.phase_t[:,i],[16,1])
            p[i] = np.abs(np.sum(np.exp(1j * delta)))
        ind = np.unravel_index(np.argmax(p), p.shape)
        return p,ind[0]/180*np.pi, p[ind[0]]
    def music(self, phase,channel):
        covMat = phase @ phase.conj().T    # calculate cov matrix

        U, sigma, _ = np.linalg.svd(covMat)
        k = next(i for i, s in enumerate(sigma[1:], 1) if sigma[0] / s > 10)
        Qn = U[:, k:]

        theta = np.linspace(0, 2*np.pi-np.pi/360, 360)
        x, y  = np.cos(theta), np.sin(theta)
        angs = np.dstack((x, y)).reshape(360, 2, 1)

        arrVec = np.exp(-2j * np.pi * self.arrayCoor @ angs /  3e8 * (2.404+channel*0.002)*1e9)
        temp = arrVec.conj().reshape(360, 1, self.array_num) @ Qn
        pspectrum = (1 / abs(temp @ temp.conj().reshape(360, -1, 1))).reshape(360)

        ind = np.unravel_index(np.argmax(pspectrum), pspectrum.shape)
        return pspectrum, theta[ind[0]], pspectrum[ind[0]]



    def pseudoSingal(self, K, SNR, theta, M, N):
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
        arrayMatrix = np.zeros((N, M)) + 1j * np.zeros((N, M))  # 阵列流形矩阵A（N*M维）

        for i in range(M):
            a = self.arrayVector(theta[i])
            arrayMatrix[:, i] = a

        X = arrayMatrix @ S

        X += np.sqrt(0.5 / SNR) * (np.random.randn(N, K) + np.random.randn(N, K) * 1j)

        return X




if __name__ == "__main__":

    music_worker = AOA_1D()
    theta = [np.pi*1.5]
    signal = music_worker.pseudoSingal(1, 15, theta, 1, 16)
    phase = np.angle(signal)

    signal = np.exp(1j * phase)
    #psp, azimuth= music_worker.music(signal)
    psp, azimuth = music_worker.p0(phase)
    print(azimuth)
    plt.figure()
    plt.plot(psp)
    plt.show()