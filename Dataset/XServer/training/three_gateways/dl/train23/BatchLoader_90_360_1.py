import numpy as np
import torch
import torch.utils.data
import cv2

class FileDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, label_path, base_path):
        self.img_path = img_path
        self.label_path = label_path

        train_txt = base_path + 'train.txt'
        test_txt = base_path + 'test.txt'

        train_index = list(np.loadtxt(train_txt))
        test_index = list(np.loadtxt(test_txt))

        train_index.extend(test_index)
        self.IDList = [str(int(x)).zfill(5) for x in train_index]


    def __getitem__(self, index):
        tmp = {}
        with open(self.label_path + '/%s.txt' % self.IDList[index], 'r') as f:
            buf = []
            for line in f:
                line = line[:-1].split(',')
                for i in range(1, len(line)):
                    line[i] = float(line[i])
                Location = [line[0], line[1], line[2]]
                buf.append({'Location': Location})
        tmp['ID'] = self.IDList[index]

        tmp['Label'] = buf
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
        self.Total = len(self.info)
        self.train_number = int(self.Total * 0.8)
        if mode == 'train':
            self.idx = 0
            self.num_of_patch = int(self.Total * 0.8)
        else:
            self.idx = self.train_number
            self.num_of_patch = int(self.Total * 0.2)


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
            else:
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
        else:
            if self.idx + 1 < self.Total:
                self.idx += 1
            else:
                self.idx = self.train_number
        return batch,  location, self.imgDataset.IDList[imgID]
