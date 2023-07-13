# -*- coding: utf-8 -*-
"""training code for Resnet-50
"""
import datetime
import math
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable

import BatchLoader_90_360_1 as BatchLoader_90_360

import resnet

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def generate_txt(base_path):
    """generate train.txt and test.txt
    """
    img_path = base_path + '/train/dataset/heatmap/'
    IDList = [x.split('.')[0] for x in sorted(os.listdir(img_path))]
    random.shuffle(IDList)
    train_number = int(len(IDList) * 0.8)
    train_list = IDList[:train_number]
    test_list = IDList[train_number:]
    train_list = np.array(train_list)
    test_list = np.array(test_list)

    train_txt = base_path + 'train.txt'
    test_txt = base_path + 'test.txt'
    np.savetxt(train_txt, train_list, fmt='%s')
    np.savetxt(test_txt, test_list, fmt='%s')


if __name__ == '__main__':
    store_path = os.path.abspath(os.path.dirname(__file__)) + '/models'
    if not os.path.isdir(store_path):
        os.mkdir(store_path)

    model_lst = [x for x in sorted(os.listdir(store_path)) if x.endswith('.pkl')]
    epochs = 1200
    batchsize = 16

    base_path = './dataset12/'

    # generate train.txt and test.txt
    file = os.listdir(base_path)
    if 'train.txt' not in file:
        print('creating train.txt and test.txt')
        generate_txt(base_path)

    img_path = base_path + '/train/dataset/heatmap/'
    label_path = base_path + '/train/dataset/poslabel/'
    all_image = os.listdir(img_path)
    test_img = cv2.imread(img_path + all_image[0])
    img_width = test_img.shape[1]
    img_height = test_img.shape[0]
    if img_height == 90 and img_width == 360:
        data = BatchLoader_90_360.FileDataset(img_path, label_path, base_path)
        data = BatchLoader_90_360.BatchDataset(data, batchsize)
        print(data.num_of_patch)
        print("Image Size is 90*360")
    else:
        print('There is error in image size! Please check image in base_path')
        sys.exit()
    print('training path', img_path)

    writer_loss = SummaryWriter('log')

    if len(model_lst) == 0:
        print('No previous model found, start training')
        model = resnet.resnet50(pretrained=True)
    else:
        print('Find previous model %s'%model_lst[-1])
        model = resnet.resnet50(pretrained=False)
        checkpoint = torch.load(store_path + '/%s'%model_lst[-1])
        model.load_state_dict(checkpoint)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    location_LossFunc = nn.MSELoss()

    if torch.cuda.is_available():
        model.cuda()
        location_LossFunc = location_LossFunc.cuda()

    iters_per_epoch = int(data.num_of_patch / batchsize)
    for epoch in range(1, 1200):  # loop over the dataset multiple times
        loss_total = 0
        for i in range(int(iters_per_epoch)):
            # get the inputs
            batch, locationGT = data.Next()
            batch = Variable(torch.FloatTensor(batch), requires_grad=False)
            locationGT = Variable(torch.FloatTensor(locationGT), requires_grad=False)

            if torch.cuda.is_available():
                batch = batch.cuda()
                locationGT = locationGT.cuda()

            location = model(batch)
            loss = location_LossFunc(location, locationGT)

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss.item()

        writer_loss.add_scalar('loss_surr', loss_total, epoch)
        loss_ave =   float(loss_total)/ iters_per_epoch
        print('Training Epoch', epoch + 1)
        print('Total Loss', loss_ave)

        now = datetime.datetime.now()
        now_s = now.strftime('%Y-%m-%d-%H-%M-%S')
        name = store_path + '/model_%s_epoch%d.pkl' % (now_s, epoch)
        model_lst = [x for x in sorted(os.listdir(store_path)) if x.endswith('.pkl')]
        if len(model_lst) > 5:
            os.remove(store_path + '/%s' % model_lst[0])
        torch.save(model.state_dict(), name)

    print('Finished Training')
