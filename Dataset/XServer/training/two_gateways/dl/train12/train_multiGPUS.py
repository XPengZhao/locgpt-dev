import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import model_comb as ResnetModel
import resnet
import os
# import BatchLoader_saveindex

# import BatchLoader_90_360
import BatchLoader_90_360_2antenas as BatchLoader_90_360
import datetime
from tensorboardX import SummaryWriter
import cv2
import random
import sys

def generate_txt1(base_path):
    img_path = base_path + '/train/dataset/heatmap/'
    IDList = [x.split('.')[0] for x in sorted(os.listdir(img_path))]
    random.shuffle(IDList)

    test_path = base_path + '/test/dataset/heatmap/'
    testList = [x.split('.')[0] for x in sorted(os.listdir(test_path))]
    random.shuffle(testList)

    # np.savetxt('E:/yangxueyuan/dataset/AoAMap/dataset2/train.txt', train_list, fmt = '%s')
    # np.savetxt('E:/yangxueyuan/dataset/AoAMap/dataset2/test.txt', test_list, fmt = '%s')
    train_txt = base_path + 'train.txt'
    test_txt = base_path + 'test.txt'
    np.savetxt(train_txt, IDList, fmt='%s')
    np.savetxt(test_txt, testList, fmt='%s')
    # img_path = base_path + '/train/dataset/heatmap/'
    # IDList = [x.split('.')[0] for x in sorted(os.listdir(img_path))]
    # random.shuffle(IDList)
    # train_number = int(len(IDList) * 0.8)
    # train_list = IDList[:train_number]
    # test_list = IDList[train_number:]
    # train_list = np.array(train_list)
    # test_list = np.array(test_list)
    #
    #
    # train_txt = base_path + 'train.txt'
    # test_txt = base_path + 'test.txt'
    # np.savetxt(train_txt, train_list, fmt='%s')
    # np.savetxt(test_txt, test_list, fmt='%s')
def generate_txt(base_path):


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
    epochs = 500
    batchsize = 150

    # base_path = '/home/guest/yxy/13m/'
    base_path = '/dataset2_1/'

    file = os.listdir(base_path)
    if 'train.txt' in file:
        pass
    else:
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
        print("Image Size is 90*360")
    else:
        print('There is error in image size! Please check image in base_path')
        sys.exit()

    writer_loss = SummaryWriter('log')


    if len(model_lst) == 0:
        print('No previous model found, start training')
        model = resnet.resnet50(pretrained=True)
    else:
        print('Find previous model %s'%model_lst[-1])
        # model = ResnetModel.LocationResNet50(pretrained=False)
        model = resnet.resnet50(pretrained=False)

        # checkpoint = torch.load(store_path + '/%s'%model_lst[-1])
        # model.load_state_dict(checkpoint)
        model_dict = torch.load(store_path + '/%s'%model_lst[-1]).module.state_dict()
        model.module.load_state_dict(model_dict)
        print('success!')

    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    location_LossFunc = nn.MSELoss()

    iters_per_epoch = int(data.num_of_patch / batchsize)

    USE_CUDA = 1
    if torch.cuda.is_available():
        device = torch.device("cuda:0" if USE_CUDA else "cpu")
        model = nn.DataParallel(model, device_ids=[0, 1]).cuda()
        location_LossFunc = location_LossFunc.cuda()

    for epoch in range(epochs):  # loop over the dataset multiple times
        loss_total = 0
        for i in range(int(iters_per_epoch)):
            # get the inputs
            batch, locationGT = data.Next()
            # test_tmp = batch[:, 2, :, :]
            # if epoch == 0:
            #     if np.all(test_tmp == 0):
            #         print('Iuput is Okay')
            #     else:
            #         print('dimension2 is not always zero')
            #         sys.exit()


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
            # print(loss.item())

            # print(i)
        writer_loss.add_scalar('loss_surr', loss_total, epoch)
        loss_ave =   float(loss_total)/ iters_per_epoch
        print('Training Epoch', epoch + 1)
        print('Total Loss', loss_ave)

        now = datetime.datetime.now()
        now_s = now.strftime('%Y-%m-%d-%H-%M-%S')
        name = store_path + '/model_%s.pkl' % now_s
        model_lst = [x for x in sorted(os.listdir(store_path)) if x.endswith('.pkl')]
        if len(model_lst) > 100:
            os.remove(store_path + '/%s' % model_lst[0])
        torch.save(model.module.state_dict(), name)

    print('Finished Training')
