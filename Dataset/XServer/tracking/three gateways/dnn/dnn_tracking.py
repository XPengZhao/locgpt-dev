# -*- coding: utf-8 -*-
"""DNN module for real time tracking
"""
import os
import sys
import time

import matplotlib.image as plm
import numpy as np
import scipy.io as scio
import torch
from torch.autograd import Variable

sys.path.append(os.path.dirname(__file__) + os.sep + '../')

from dnn import resnet


class DNN_tracking(object):

    def __init__(self, model_name) -> None:
        store_path =  'models/'
        if not os.path.isdir(store_path):
            raise ValueError('No folder named \"models/\"')

        if not os.path.exists(store_path + model_name):
            raise ValueError('No model: %s found, please check it' % model_name)

        self.model = resnet.resnet50(pretrained=False)
        checkpoint = torch.load(store_path + '/%s' % model_name)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def get_location(self, img):
        """get the localtion

        Parameters
        ----------
        img : 2D ndarray
            [R,G,B]

        Returns
        ----------
        location : ndarray
            [x,y,z]
        """
        batch = np.zeros((1,3,90,360))
        batch[0,0,:,:] = img[:,:,0]
        batch[0,1,:,:] = img[:,:,1]
        batch[0,2,:,:] = img[:,:,2]
        # img = img.transpose((2,0,1))
        # batch = img[np.newaxis, :,:]
        # print(batch.shape)
        batch = Variable(torch.FloatTensor(batch), requires_grad=False)
        if torch.cuda.is_available():
            batch = batch.cuda()
        location = self.model(batch)
        location = location.cpu().detach().numpy()[0]
        return location

    @staticmethod
    def error_cal(loc, locgt):
        return np.linalg.norm((np.array(loc)-np.array(locgt)))


if __name__ == "__main__":
    track_worker1, track_worker2, track_worker3 = DNN_tracking(), DNN_tracking(), DNN_tracking()
    folder_path = '../dataset3_kl/train/dataset/heatmap/'
    txt_path =  '../dataset3_kl/train/dataset/poslabel/'
    labels = np.zeros((0,3))
    for txt in sorted(os.listdir(txt_path)):
        labels = np.vstack((labels, np.loadtxt(txt_path+txt, delimiter=',')))

    for ind,img in enumerate(sorted(os.listdir(folder_path))):
        start_time = time.time()
        imgpath = folder_path + img
        fig = plm.imread(imgpath) / 255
        loc1 = track_worker1.get_location(fig)
        loc2 = track_worker1.get_location(fig)
        loc3 = track_worker1.get_location(fig)
        loc = (loc1 + loc2 + loc3) / 3
        end_time = time.time()

        print('time:',end_time - start_time)

        print(DNN_tracking.error_cal(loc1, labels[ind]))



