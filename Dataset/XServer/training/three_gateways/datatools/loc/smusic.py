# -*- coding: utf-8 -*-
"""music算法与slide music
"""
import os
import sys
import time

import numpy as np
import scipy.constants as sconst
from skspatial.objects import Line, Plane

sys.path.append(os.path.dirname(__file__) + os.sep + '../')

from loc.coor_transform import rigid_transform_3D
from utils.param_loader import ParamLoader as PM

class Music():

    """music算法实现，平面阵列"""
    def __init__(self, arrayRow, arrayCol, distance, freq):
        self.arrayRow = arrayRow
        self.arrayCol = arrayCol
        self.freq = freq
        self.wavelength = 3e8 / self.freq     # wavelength
        self.distance = distance              # 阵元间距
        self.arrayCoor = self.getArrayCoor()  # 阵元坐标


    def getArrayCoor(self):
        """计算均匀平面阵列坐标，列向量化[e1,e2,e3,e4,e5...,e16], 其中ei为坐标[x,y]

        return
        -----------
        arrayCorr : 2-D array, shape(16,2)
            阵元位置 [[x11,y11], [x12, y12], ...., [x44, y44]], x23表示第二行第三列
        """
        arrayCorr = np.zeros((self.arrayRow * self.arrayCol, 2))
        for i in range(self.arrayRow * self.arrayCol):
            arrayCorr[i, 0] = i % self.arrayRow * self.distance   # x坐标
            arrayCorr[i, 1] = i // self.arrayCol * self.distance  # y坐标

        return arrayCorr

    def arrayVector(self, theta, phi):
        """计算阵列流形矩阵

        Parameters
        ----------
        theta : float
            Azimuth 方位角
        phi : float
            Elevation 俯仰角

        Returns
        -------
        arrayVector : 1-D array
            阵列流形向量
        """
        disdiff = self.arrayCoor[:,0]*np.cos(theta)*np.cos(phi) + self.arrayCoor[:, 1]*np.sin(theta)*np.cos(phi)
        arrayVector = np.exp(-2j * np.pi * disdiff / self.wavelength)

        return arrayVector

    @staticmethod
    def getCov(sigArray):
        covMat = np.cov(sigArray)  # 16 x 16维协方差矩阵
        return covMat

    @staticmethod
    def getCovSingle(sigArray):
        # num = sigArray.shape[0]
        temp = sigArray
        # temp = np.delete(sigArray,num-1,axis=0)          #去掉末尾
        # temp = np.insert(temp,0,sigArray[0],axis=0)      #添加开头
        # temp = sigArray / temp
        covMat = temp @ temp.conj().transpose()
        return covMat


    def getMusicResult(self, covMat):
        """计算music结果 for循环实现
        """
        spaceAzimuth = np.linspace(0, np.pi * 2, 360)  # 空间谱搜索角度 0-2pi
        spaceElevation = np.linspace(0, np.pi / 2, 90)  # 0-pi/2

        s1 = time.time()

        U, sigma, V = np.linalg.svd(covMat)
        s0 = sigma[0]
        k = 0
        for i in range(1, len(sigma)):
            if s0 / sigma[i] > 10:
                k = i
                break
        Qn = U[:, k:]
        # print(k)
        # sigma, v = np.linalg.eig(covMat)
        # sigma = abs(sigma)
        # Qn = v[:, k:]

        s2 = time.time()
        print("svd time consuming: %0.5f"%(s2-s1))

        numAzimuth = spaceAzimuth.size
        numElevation = spaceElevation.size
        pspectrum = np.zeros((numAzimuth, numElevation))  # 360 x 90 的空间谱数据

        for i in range(numAzimuth):
            for j in range(numElevation):
                arrayvector = self.arrayVector(spaceAzimuth[i], spaceElevation[j])
                temp = arrayvector.conj().transpose() @ Qn
                pspectrum[i, j] = 1 / abs(temp @ temp.conj().transpose())

        ind = np.unravel_index(np.argmax(pspectrum, axis=None), pspectrum.shape)
        Azimuth = spaceAzimuth[ind[0]]
        Elevation = spaceElevation[ind[1]]

        s3 = time.time()
        print("search time consuming: %0.2f"%(s3-s2))

        return pspectrum, Azimuth, Elevation


    def getMusicResultFast(self, covMat):

        a_step = 1 * 360
        e_step = 1 * 90
        spaceAzimuth = np.linspace(0, np.pi * 2, a_step)  # 空间谱搜索角度 0-2pi
        spaceElevation = np.linspace(0, np.pi / 2, e_step)  # 0-pi/2

        # s1 = time.time()
        U, sigma, V = np.linalg.svd(covMat)
        s0 = sigma[0]
        k = 0
        for i in range(1, len(sigma)):
            if s0 / sigma[i] > 10:
                k = i
                break
        Qn = U[:, k:]
        # print(k)
        # sigma, v = np.linalg.eig(covMat)
        # sigma = abs(sigma)
        # Qn = v[:, k:]
        # s2 = time.time()
        # print("svd time consuming: %0.5f"%(s2-s1))

        theta = np.linspace(0, 2*np.pi, a_step)
        theta = np.repeat(theta,e_step)
        phi = np.linspace(0, np.pi/2, e_step)
        phi = np.tile(phi,a_step)

        x = np.cos(theta) * np.cos(phi)
        y = np.sin(theta) * np.cos(phi)

        angs = np.dstack((x,y))
        angs =  angs.reshape(a_step*e_step, 2, 1)

        arrVec = self.arrayCoor @ angs
        arrVec = np.exp(-2j * np.pi * arrVec / self.wavelength)  # 32400x9x1

        arrVecT = arrVec.conj().reshape(a_step*e_step, 1, self.arrayRow*self.arrayCol)    #32400x1x9
        temp = arrVecT @ Qn                             #32400x1xn
        tempT = temp.conj().reshape(a_step*e_step,-1, 1)
        pspectrum = temp @ tempT                        #32400x1x1
        pspectrum = 1/abs(pspectrum)
        pspectrum = pspectrum.reshape(a_step,e_step)

        ind = np.unravel_index(np.argmax(pspectrum, axis=None), pspectrum.shape)
        Azimuth = spaceAzimuth[ind[0]]
        Elevation = spaceElevation[ind[1]]

        # s3 = time.time()
        # print("search time consuming: %0.2f"%(s3-s2))

        return pspectrum, Azimuth, Elevation


    def getMusicResultFast_2(self, covMat):

        spaceAzimuth = np.linspace(0, np.pi * 2, 360)  # 空间谱搜索角度 0-2pi
        spaceElevation = np.linspace(0, np.pi / 2, 90)  # 0-pi/2

        # s1 = time.time()
        U, sigma, V = np.linalg.svd(covMat)
        s0 = sigma[0]
        k = 0
        for i in range(1, len(sigma)):
            if s0 / sigma[i] > 10:
                k = i
                break
        Qn = U[:, k:]

        theta = np.linspace(0, np.pi * 2, 360)
        theta = np.repeat(theta,90)
        phi = np.linspace(0, np.pi/2, 90)
        phi = np.tile(phi,360)

        x = np.cos(theta) * np.cos(phi)
        y = np.sin(theta) * np.cos(phi)

        angs = np.dstack((x,y))
        angs =  angs.reshape(360*90, 2, 1)

        arrVec = self.arrayCoor @ angs
        arrVec = np.exp(-2j * np.pi * arrVec / self.wavelength)  # 32400x9x1

        arrVecT = arrVec.conj().reshape(360*90, 1, 16)    #32400x1x9
        temp = arrVecT @ Qn                             #32400x1xn
        tempT = temp.conj().reshape(360*90,-1, 1)
        pspectrum = temp @ tempT                        #32400x1x1
        pspectrum = 1/abs(pspectrum)
        pspectrum = pspectrum.reshape(360,90)

        ind = np.unravel_index(np.argmax(pspectrum, axis=None), pspectrum.shape)
        Azimuth = spaceAzimuth[ind[0]]
        Elevation = spaceElevation[ind[1]]

        return pspectrum, Azimuth, Elevation


    def algMusic(self, sigArray):
        """music算法实现

        Parameters
        ----------
        sigArray : 2-D array, shape(16,K), K为快拍数
            16个相位数据
        """
        Cov = self.getCovSingle(sigArray)
        pspectrum, Azimuth, Elevation = self.getMusicResultFast_2(Cov)
        return pspectrum, Azimuth, Elevation


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



class SmoothMusic(object):
    """music算法实现，平面阵列, smoothing version"""

    def __init__(self,distance,arrLength, subarrLength, freq):
        self.arrayRow = arrLength
        self.arrayCol = arrLength
        self.arrayRow_sub = subarrLength
        self.arrayCol_sub = subarrLength
        self.music = Music(self.arrayRow_sub, self.arrayCol_sub, distance, freq)

    def _getSubCovMean(self, phaseData):
        """计算子阵协方差矩阵平均值

        Parameters
        ----------
        phaseData : 2-D array (shape 16xn)
            16个阵元的相位数据,n为快拍数（16行 n列）
        """
        rowStep = self.arrayRow - self.arrayRow_sub + 1
        colStep = self.arrayCol - self.arrayCol_sub + 1
        allStep = rowStep * colStep
        covSum=np.zeros((self.arrayRow_sub*self.arrayCol_sub,self.arrayRow_sub*self.arrayCol_sub))

        for i in range(0,rowStep):
            for j in range(0,colStep):
                topLeftind  = i*self.arrayCol + j     #子阵左上角索引
                templete    = np.arange(topLeftind,topLeftind+self.arrayCol_sub)
                subArrInd   = np.array([],int)            #子阵阵元索引
                for k in range(0, self.arrayCol_sub):
                    subArrInd = np.hstack((subArrInd, templete+k*self.arrayCol))
                subArrPhase = phaseData[subArrInd]
                covSum = covSum + self.music.getCovSingle(subArrPhase)

        covMean=covSum/allStep
        return covMean


    def getSMusicResult(self,data):
        """计算slide Music结果

        Parameters
        ----------
        data : 2-D array
            [[e1t1,e2t2,..], [e2t1,e2t2...], ..., [e16t1,e16t2,..]]
        """
        covMean = self._getSubCovMean(data)
        pspectrum, Azimuth, Elevation=self.music.getMusicResultFast(covMean)
        return pspectrum, Azimuth, Elevation


    def getpseudoSingal(self):
        K = 10
        M = 1
        SNR = 20
        azimuth = [3.14]
        elevation = [1.0]

        elementNum=self.arrayRow_sub*self.arrayCol_sub

        S = np.random.randn(M, K) + 1j * np.random.randn(M, K)  # 入射信号矩阵S（M*K维）
        arrayMatrix = np.zeros((elementNum, M)) + 1j * np.zeros((elementNum, M))  # 阵列流形矩阵A（N*M维）

        for i in range(M):
            a = self.music.arrayVector(azimuth[i], elevation[i])
            arrayMatrix[:, i] = a

        X = arrayMatrix @ S
        X += np.sqrt(0.5 / SNR) * (np.random.randn(elementNum, K) + np.random.randn(elementNum, K) * 1j)
        return X

    @staticmethod
    def position(azimuth, elevation):
        x = np.cos(azimuth) * np.cos(elevation) / np.sin(elevation)
        y = np.sin(azimuth) * np.cos(elevation) / np.sin(elevation)
        z = 1
        try:
            line = Line(point=[0,0,0], direction=[x,y,z])
            plane = Plane(point=[0,0,0.9], normal=[0,0,1])
            pin = plane.intersect_line(line)
            pin[0] = pin[0] + 0.24
            pin[1] = -(pin[1] - 0.24)
            x = pin[0]*100
            y = pin[1]*100
        except:
            print('get position error')


if __name__ =="__main__":
    pass