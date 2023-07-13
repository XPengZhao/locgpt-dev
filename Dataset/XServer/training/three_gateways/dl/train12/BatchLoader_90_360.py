import numpy as np
import torch
import torch.utils.data
import cv2
import sys
import os

class FileDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, label_path, base_path, mode='train'):
        self.img_path = img_path
        self.label_path = label_path

        self.labels = np.loadtxt(os.path.join(self.label_path, 'labels.txt'),delimiter=',')

        train_txt = base_path + 'train.txt'
        test_txt = base_path + 'test.txt'

        self.train_index = list(np.loadtxt(train_txt))
        self.test_index = list(np.loadtxt(test_txt))
        self.all_index = self.train_index + self.test_index
        self.IDList = [str(int(x)).zfill(5) for x in self.all_index]



    def __getitem__(self, index):
        tmp = {}

        tmp['ID'] = self.IDList[index]
        ind = int(self.IDList[index])
        tmp['Label'] = [{'Location':self.labels[ind-1].tolist()}]

        return tmp


    def GetImage(self, idx):
        name = '%s/%s.jpg'%(self.img_path, self.IDList[idx])
        # print(name)
        img = cv2.imread(name).astype(np.float) / 255
        return img

    def __len__(self):
        return len(self.IDList)



class BatchDataset:
    def __init__(self, imgDataset, batchSize=1, mode='train'):
        self.imgDataset = imgDataset
        self.batchSize = batchSize

        self.mode = mode
        self.imgID = None

        self.info = self.getBatchInfo()
        self.Total = len(self.imgDataset.all_index)
        self.train_number = len(self.imgDataset.train_index)
        self.test_number = len(self.imgDataset.test_index)
        print('total dataset:%d, trainset:%d, testset:%d' % (self.Total, self.train_number, self.test_number))

        if mode == 'train':
            self.idx = 0
            self.num_of_patch = self.train_number
        else:
            self.idx = self.train_number
            self.num_of_patch = self.test_number


    def getBatchInfo(self):
        """read all  labels in the dataset

        Returns
        -------
        list
            [{'ID':num in the txt, 'Index':ind, 'Location':pos}]
        """
        data = []
        total = len(self.imgDataset)
        for idx, one in enumerate(self.imgDataset):
            ID = one['ID']
            # img = one['Image']
            allLabel = one['Label']
            for label in allLabel:
                data.append({'ID': ID,
                             'Index': idx,
                    'Location': label['Location']
                })
        return data


    def Next(self):
        batch = np.zeros([self.batchSize, 3, 90, 360], np.float)
        location = np.zeros([self.batchSize, 3], np.float)
        record = None
        for one in range(self.batchSize):
            data = self.info[self.idx]

            imgID = data['Index']
            if imgID != record:
                img = self.imgDataset.GetImage(imgID)

            location[one, :] = data['Location']
            batch[one, 0, :, :] = img[:, :, 2]
            batch[one, 1, :, :] = img[:, :, 1]
            batch[one, 2, :, :] = img[:, :, 0]
            if self.mode == 'train':
                if self.idx + 1 < self.num_of_patch:
                    self.idx += 1
                else:
                    self.idx = 0
            elif self.mode == 'test':
                if self.idx + 1 < self.Total:
                    self.idx += 1
                else:
                    self.idx = self.train_number
        return batch, location


    def EvalBatch(self):
        batch = np.zeros([1, 3, 90, 360], np.float)
        location = np.zeros([1, 3], np.float)
        info = self.info[self.idx]
        imgID = info['Index']
        if imgID != self.imgID:
            self.img = self.imgDataset.GetImage(imgID)
            self.imgID = imgID
        print(imgID)
        location[0, :] = info['Location']
        batch[0, 0, :, :] = self.img[:, :, 2]
        batch[0, 1, :, :] = self.img[:, :, 1]
        batch[0, 2, :, :] = self.img[:, :, 0]

        if self.mode == 'train':
            if self.idx + 1 < self.num_of_patch:
                self.idx += 1
            else:
                self.idx = 0
        elif self.mode == 'test':
            if self.idx + 1 < self.Total:
                self.idx += 1
            else:
                self.idx = self.train_number
        return batch,  location, self.imgDataset.IDList[imgID]
