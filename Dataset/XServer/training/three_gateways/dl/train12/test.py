# -*- coding: utf-8 -*-
"""test
"""
import math
import os
import random
import sys

import cv2
import numpy as np
import scipy.io as scio
import torch
from torch.autograd import Variable

import BatchLoader_90_360
import resnet


def error_cal(loc, locgt):
    err = 0
    for i in range(len(loc)):
        err += np.square((loc[i] - locgt[i]))
    return math.sqrt(err)

def generate_txt(base_path):
    img_path = base_path + '/train/dataset/heatmap/'
    IDList = [x.split('.')[0] for x in sorted(os.listdir(img_path))]
    random.shuffle(IDList)

    test_path = base_path + '/test/dataset/heatmap/'
    testList = [x.split('.')[0] for x in sorted(os.listdir(test_path))]
    random.shuffle(testList)

    train_txt = base_path + 'train.txt'
    test_txt = base_path + 'test.txt'
    np.savetxt(train_txt, IDList, fmt='%s')
    np.savetxt(test_txt, testList, fmt='%s')

if __name__ == '__main__':
    store_path = os.path.abspath(os.path.dirname(__file__)) + '/models'
    if not os.path.isdir(store_path):
        print('No folder named \"models/\"')


    model_lst = [x for x in sorted(os.listdir(store_path)) if x.endswith('.pkl')]
    epochs = 1000
    batchsize = 1

    base_path = './dataset12/'
    # generate train.txt and test.txt

    file = os.listdir(base_path)
    if 'train.txt' in file:
        pass
    else:
        print('creating train.txt and test.txt')
        generate_txt(base_path)


    # path_error_dic_base = '/home/guest/yxy/ANNTNN_test/'
    if not os.path.exists('./result/error'):
        os.makedirs('./result/error')

    path_error_dic_base = './result/'
    error_path = './result/error/'

    # img_path = base_path + '/test/dataset/heatmap/'
    # label_path = base_path + '/test/dataset/poslabel/'
    img_path = base_path + '/heatmap/'
    label_path = base_path + '/poslabel/'
    all_image = os.listdir(img_path)
    test_img = cv2.imread(img_path + all_image[0])
    img_width = test_img.shape[1]
    img_height = test_img.shape[0]
    if img_height == 90 and img_width == 360:
        data = BatchLoader_90_360.FileDataset(img_path, label_path, base_path, 'test')
        data = BatchLoader_90_360.BatchDataset(data, batchsize, 'test')
        print("Image Size is 90*360")
    else:
        print('There is error in image size! Please check image in base_path')
        sys.exit()



    if len(model_lst) == 0:
        print('No previous model found, please check it')
        exit()
    else:
        print('Find previous model %s' % model_lst[-1])
        model = resnet.resnet50(pretrained=False)
        checkpoint = torch.load(store_path + '/%s' % model_lst[-1])
        model.load_state_dict(checkpoint)
        model.eval()

    error_dic = []
    error_dic_x = []
    error_dic_y = []
    error_dic_z = []
    error_dic_xy = []
    error_dic_yz = []
    error_dic_xz = []
    error = []
    errof_dic = {}

    location_list = np.zeros((0,3))
    gt_list = np.zeros((0,3))

    # data.num_of_patch
    for i in range(data.num_of_patch):
        batch, locationGT, imgID = data.EvalBatch()
        # locationGT = info['Location']
        batch = Variable(torch.FloatTensor(batch), requires_grad=False)
        if torch.cuda.is_available():
            model = model.cuda()
            batch = batch.cuda()
        location = model(batch)


        location = location.cpu().detach().numpy()[0]
        locationGT = locationGT[0]
        xz = []
        xz.append(location[0])
        xz.append(location[2])

        xzgt = []
        xzgt.append(locationGT[0])
        xzgt.append(locationGT[2])
        location_error = error_cal(location, locationGT)
        txt_path = error_path + imgID + '.txt'
        f = open(txt_path, 'w')
        f.write(str(location_error))
        f.write('\n')
        f.close()

        location_list = np.vstack((location_list, location))
        gt_list = np.vstack((gt_list, locationGT))

        location_error_xy = error_cal(location[:2], locationGT[:2])
        location_error_yz = error_cal(location[1:], locationGT[1:])
        location_error_xz = error_cal(xz, xzgt)

        # error_all = sum(abs(np.array(locationGT) - location))
        error_x = abs((locationGT[0]) - location[0])
        error_y = abs((locationGT[1]) - location[1])
        error_z = abs((locationGT[2]) - location[2])

        error.append(location_error)
        aa = str(i)
        # error_dic_all[str[i]] = error_all

        error_dic.append(location_error)
        error_dic_x.append(error_x)
        error_dic_xy.append(location_error_xy)
        error_dic_yz.append(location_error_yz)
        error_dic_xz.append(location_error_xz)
        error_dic_y.append(error_y)
        error_dic_z.append(error_z)
        print(i, location_error, location_error_xy, location_error_yz, location_error_xz,  error_x, error_y, error_z)
    # path_error_dic_all = 'E:/yangxueyuan/error_dic_all.mat'
    # scio.savemat(path_error_dic_all, error_dic_all)
    errof_dic = {'Y1':error_dic, 'Y2':error_dic_xy, 'Y3':error_dic_xz, 'Y4':error_dic_yz, 'Y5':error_dic_x, 'Y6':error_dic_y, 'Y7':error_dic_z}

    path_error_dic = path_error_dic_base + 'error_dic_big.mat'
    scio.savemat(path_error_dic, errof_dic)

    path_loc_dic = path_error_dic_base + 'location.mat'
    scio.savemat(path_loc_dic, {'loc':location_list, 'gt':gt_list})

        # if i % 60 == 0:
        #    print (theta, Ry)
        #    print (dim.tolist(), dimGT)
        # if i % 100 == 0:
        #     now = datetime.datetime.now()
        #     now_s = now.strftime('%Y-%m-%d-%H-%M-%S')
        #     print('------- %s %.5d -------' % (now_s, i))
        #
        #     print('Location error: %lf' % (np.mean(error)))

    print('Location error: %lf' % (np.mean(error)))
