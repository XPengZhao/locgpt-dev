# -*- coding: utf-8 -*-
import numpy as np

class Track_filter(object):

    def __init__(self):
        self.tr_buff = np.zeros((0,3))
        self.med_len = 5

    @staticmethod
    def uf_filter():
        return np.random.uniform(0.05, 0.2, 3)

    def med_filter(self, loc):
        """median filter

        Parameters
        ----------
        loc : 1D ndarray
            [x,y,z]

        Returns
        -------
        loc_med : 1D ndarray
            [x,y,z]
        """
        self.tr_buff = np.vstack((self.tr_buff, loc))
        if self.tr_buff.shape[0] <= self.med_len:
            return np.array([])

        self.tr_buff = np.delete(self.tr_buff, 0, axis=0)
        return np.median(self.tr_buff, axis=0)

# class Kalman2D(KalmanFilter):

#     def __init__(self):
#         super().__init__(2,2)
#         self.measurementMatrix = np.array([[1,0],[0,1]],np.float32)
#         self.transitionMatrix = np.array([[1,0],[0,1]], np.float32)
#         self.processNoiseCov = np.array([[1,0],[0,1]], np.float32) * 0.01
#         self.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 0.08

#         self.count = 0

#     def kalmanPredict(self, mes):

#         mes = np.reshape(mes,(2,1))
#         if self.count == 0:
#             self.statePost = np.array(mes,np.float32)
#         else:
#             self.predict()                          # 预测
#             self.correct(mes)                       # 用测量值纠正
#         self.count += 1

#         return self.statePost
