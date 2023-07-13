# -*- coding: utf8 -*-
"""卡尔曼滤波
"""
from cv2 import KalmanFilter
import numpy as np

class Kalman2D(KalmanFilter):

    def __init__(self):
        super().__init__(2,2)
        self.measurementMatrix = np.array([[1,0],[0,1]],np.float32)
        self.transitionMatrix = np.array([[1,0],[0,1]], np.float32)
        self.processNoiseCov = np.array([[1,0],[0,1]], np.float32) * 0.01
        self.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 0.05

        self.count = 0

    def ch_param(self, process_noise, measure_noise):
        self.processNoiseCov = np.array([[1,0],[0,1]], np.float32) * process_noise
        self.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * measure_noise

    def kalmanPredict(self, mes):

        mes = np.reshape(mes,(2,1))
        if self.count == 0:
            self.statePost = np.array(mes,np.float32)
        else:
            self.predict()                          # 预测
            self.correct(mes)                       # 用测量值纠正
        self.count += 1

        return self.statePost


class Kalman1D(KalmanFilter):

    def __init__(self):
        super().__init__(1,1)
        self.measurementMatrix = np.array([[1]],np.float32)
        self.transitionMatrix = np.array([[1]], np.float32)
        self.processNoiseCov = np.array([[1]], np.float32) * 0.01
        self.measurementNoiseCov = np.array([[1]], np.float32) * 0.08

        self.count = 0

    def kalmanPredict(self, mes):

        mes = np.reshape(mes,(1,1))
        if self.count == 0:
            self.statePost = np.array(mes, np.float32)
        else:
            self.predict()                          # 预测
            self.correct(mes)                       # 用测量值纠正
        self.count += 1

        return self.statePost
