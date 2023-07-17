import argparse
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from shutil import copyfile

import numpy as np
import pandas as pd
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm

from model import *

dis2mse = lambda x, y: torch.mean((x - y) ** 2)
dis2me = lambda x, y: np.linalg.norm(x - y, axis=-1)


class MyDataset(Dataset):
    def __init__(self, data_dir):

        df = pd.read_csv(data_dir+'.csv')
        self.spt = torch.load(data_dir+'.t')  #[datalen, 3, spt_dim]
        self.enc_token = torch.arange(len(self.spt)).unsqueeze(1) #[datalen, 1]
        self.dec_token = torch.arange(len(self.spt)).unsqueeze(1) #[datalen, 1]

        area = np.zeros((len(df), 1))
        tag = df.iloc[:, 9:12].values
        gateway = df.iloc[:, 0:9].values
        self.labels = np.concatenate((area, tag, gateway), axis=-1)  # (datalen, 4+9)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def loaddata(self):
        return self.enc_token, self.spt, self.dec_token, self.labels




class LocGPT_Runner():
    def __init__(self, mode, **kwargs) -> None:

        kwargs_path = kwargs['path']
        kwargs_network = kwargs['networks']
        kwargs_train = kwargs['training']

        # Path
        self.expname = kwargs_path['expname']
        self.datadir = kwargs_path['datadir']
        self.logdir = kwargs_path['logdir']
        self.load_ckpt = kwargs_path['load_ckpt']
        self.train_file = kwargs_path['train_file']
        self.test_file = kwargs_path['test_file']

        self.devices = torch.device('cuda')
        self.phase_encoder, _ = get_embedder(multires=10, input_ch=1)  # 1 -> 1x2x10

        ## Network
        self.locgpt = LocGPT().to(self.devices)
        if kwargs_network['init_weight'] and mode=='train':
            self.locgpt.apply(self.init_weights)

        params = list(self.locgpt.parameters())
        self.optimizer = optim.Adam(params, lr=float(kwargs_train['lr']),
                                    weight_decay=float(kwargs_train['weight_decay']))
        self.cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,T_max=20,eta_min=1e-5)


        ## Train settings
        self.epoch_start = 1
        if kwargs_path['load_ckpt'] or mode=='test':
            self.load_checkpoints()
        self.current_epoch = self.epoch_start
        self.batch_size = kwargs_train['batch_size']
        self.total_epoches = kwargs_train['total_epoches']
        self.beta = kwargs_train['beta']
        self.i_save = kwargs_train['i_save']


        ## Dataset
        train_data_dir = os.path.join(self.datadir, self.train_file)
        test_data_dir = os.path.join(self.datadir, self.test_file)
        train_set = MyDataset(train_data_dir)
        test_set = MyDataset(test_data_dir)
        train_enc_token, self.train_spt, train_dec_token, self.train_label = train_set.loaddata()
        test_enc_token, self.test_spt, test_dec_token, self.test_label = test_set.loaddata()
        train_dataset = TensorDataset(train_enc_token, train_dec_token)
        test_dataset = TensorDataset(test_enc_token, test_dec_token)

        self.transform_iter = torch.utils.data.DataLoader(train_dataset, self.batch_size,
                                                          shuffle=False, drop_last=False, num_workers=0)
        self.train_iter = torch.utils.data.DataLoader(train_dataset, self.batch_size,
                                                      shuffle=True, drop_last=True, num_workers=0)
        self.test_iter = torch.utils.data.DataLoader(test_dataset, self.batch_size,
                                                     shuffle=False, drop_last=False, num_workers=0)

        self.logger = SummaryWriter(os.path.join(self.logdir, self.expname, 'tensorboard'))


    def init_weights(self, m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)


    def load_checkpoints(self):
        """load checkpoints and epoch
        """
        epoch_start = 1
        ckptsdir = os.path.join(self.logdir, self.expname, 'ckpts')
        if not os.path.exists(ckptsdir):
            os.makedirs(ckptsdir)
        ckpts = [os.path.join(ckptsdir, f) for f in sorted(os.listdir(ckptsdir)) if 'tar' in f]

        print('Found ckpts', ckpts)
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print('Reload from', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.devices)

            self.locgpt.load_state_dict(ckpt['locgpt_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,T_max=20,eta_min=1e-5)
            self.cosine_schedule.load_state_dict(ckpt['schedule_state_dict'])
            self.epoch_start = ckpt['epoch_start']




    def saveckpts(self):
        """save checkpoints and epoch
        """
        ckptsdir = os.path.join(self.logdir, self.expname, 'ckpts')
        model_lst = [x for x in sorted(os.listdir(ckptsdir)) if x.endswith('.tar')]
        if len(model_lst) > 2:
            os.remove(ckptsdir + '/%s' % model_lst[0])

        ckptname = os.path.join(ckptsdir, '{:06d}.tar'.format(self.current_epoch))
        torch.save({
            'epoch_start': self.current_epoch,
            'locgpt_state_dict': self.locgpt.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'schedule_state_dict': self.cosine_schedule.state_dict(),
        }, ckptname)
        print('Saved checkpoints at', ckptname)


    def criterion(self, x, y):
        def loss_l2(preds, labels):
            l_ = preds.shape[0]
            w = torch.tensor([(i, j) for i in range(l_ - 1) for j in range(i + 1, l_)]).to(preds.device)
            diff_preds = preds[w[:, 0], 1:] - preds[w[:, 1], 1:]
            diff_preds = torch.norm(diff_preds, dim=1)
            diff_labels = labels[w[:, 0], 1:] - labels[w[:, 1], 1:]
            diff_labels = torch.norm(diff_labels, dim=1)

            return dis2mse(diff_preds, diff_labels), torch.unique(w, return_counts=True)[1].reshape(-1, 1)

        mse = nn.MSELoss()
        l1 = mse(x[:, 0], y[:, 0])
        l2, _ = loss_l2(x, y)
        l3 = self.beta * l1 + (1-self.beta) * l2

        return l1, l2, l3


    def train_network(self):

        self.locgpt.train()
        total_num = len(self.train_iter.dataset)
        num_batches = len(self.train_iter)
        log_step_interval = 1
        print(total_num, num_batches)
        for epoch in range(self.epoch_start, self.total_epoches):
            with tqdm(total=num_batches, desc=f"Epoch {epoch}/{self.total_epoches}") as pbar:
                for step, (enc_token, dec_token) in enumerate(self.train_iter):

                    spt = self.train_spt[enc_token].to(self.devices)
                    label = self.train_label[dec_token].to(self.devices)
                    area_tagpos, gateway_pos = label[..., :4], label[..., 4:]
                    enc_token = enc_token.to(self.devices, dtype=torch.int32)
                    dec_token = dec_token.to(self.devices, dtype=torch.int32)    #[B, n_seq]
                    dec_input = torch.ones((len(dec_token), 1, 3), dtype=torch.float32).to(self.devices)

                    self.optimizer.zero_grad()
                    # output = self.locgpt(enc_token, spt, dec_token, area_tagpos[...,1:4], gateway_pos)
                    output = self.locgpt(enc_token, spt, dec_token, dec_input, gateway_pos)
                    l1, l2, l3 = self.criterion(output.squeeze(), area_tagpos.squeeze())
                    loss = l3
                    loss.backward()

                    self.optimizer.step()
                    global_iter_num = epoch * num_batches + step + 1
                    pbar.update(1)
                    pbar.set_postfix_str(f"l1 loss: {l1.item():.6f}, l2 loss: {l2.item():.6f}, lr: {self.optimizer.param_groups[0]['lr']:.9f}")
                    if global_iter_num % log_step_interval == 0:
                        self.logger.add_scalar("l1 loss", l1.item(), global_step=global_iter_num)
                        self.logger.add_scalar("l2 loss", l2.item(), global_step=global_iter_num)
            self.cosine_schedule.step()
            self.current_epoch = epoch

            if self.current_epoch % self.i_save == 0:
                self.saveckpts()


    def pred(self, dataset):
        """
        Returns
        -----------
        pred_all: [B, 4]. predict results (s, x, y, z)
        gt_all: [B, 4]. ground truth results (s, x, y, z)
        """

        self.locgpt.eval()
        dataset_len = len(dataset.dataset)
        gt_all = np.zeros((dataset_len, 4))
        pred_all = np.zeros((dataset_len, 4))

        for i, (enc_token, dec_token) in enumerate(dataset):

            spt = self.train_spt[enc_token].to(self.devices)
            label = self.train_label[dec_token]
            area_tagpos, gateway_pos = label[..., :4], label[..., 4:]
            test_labels = area_tagpos.squeeze().numpy()
            enc_token = enc_token.to(self.devices, dtype=torch.int32)
            dec_token = dec_token.to(self.devices, dtype=torch.int32)    #[B, n_seq]
            dec_input = torch.ones((len(dec_token), 1, 3), dtype=torch.float32).to(self.devices)


            with torch.no_grad():
                preds = self.locgpt(enc_token, spt, dec_token, dec_input, gateway_pos).squeeze()
                preds = preds.cpu().detach().numpy()  # [B, 4]
                pred_all[i*self.batch_size:(i+1)*self.batch_size] = preds
                gt_all[i*self.batch_size:(i+1)*self.batch_size] = test_labels



        return pred_all, gt_all


    def eval_network(self):

        pred_all, gt_all = self.pred(self.test_iter)

        ## Calculate distance diff errors
        points_preds, points_labels = pred_all[:, 1:], gt_all[:, 1:]
        if points_preds.shape[0] % 2 != 0:
            points_preds = np.delete(points_preds, -1, axis=0)
            points_labels = np.delete(points_labels, -1, axis=0)
        diff_features = points_labels[0::2, :] - points_labels[1::2, :]
        diff_labels = points_preds[0::2, :] - points_preds[1::2, :]
        diff_features_dist = np.linalg.norm(diff_features, axis=1)
        diff_labels_dist = np.linalg.norm(diff_labels, axis=1)
        diff_error = abs(diff_features_dist - diff_labels_dist)
        print("Distance diff median error:", np.median(diff_error))

        ## pos error
        R, t = self.get_transform()

        points_preds = points_preds @ R.T + t
        pos_error = dis2me(points_preds, points_labels)
        scio.savemat(os.path.join(self.logdir, self.expname, "pos_error.mat"),
                     {"pos_error":pos_error})

        print('Location Median Error', np.median(pos_error))



    def get_transform(self):

        # # R = 3x3 rotation matrix
        # # t = 3 column vector
        # # B = A@R.T + t
        pred_all, gt_all = self.pred(self.transform_iter)
        pred_pos, gt_pos = pred_all[:, 1:], gt_all[:, 1:]
        pos_error = np.linalg.norm(gt_pos-pred_pos, axis=1)

        print('train data Median error before transform:', np.median(pos_error))

        R, t = self.learn_Rt(gt_pos, pred_pos)
        R, t = R.detach().numpy(), t.detach().numpy()
        # np.savetxt('gt_pre.txt', np.concatenate((gt_pos,pred_pos), axis=-1), fmt='%.3f')

        pred_pos_A = pred_pos @ R.T + t
        pos_error = np.linalg.norm(gt_pos-pred_pos_A, axis=1)
        print('train data Median error after transform', np.median(pos_error))
        return R, t


    def learn_Rt(self, coords_A, coords_B):
        """
        coords_A: [N, 3]
        coords_B: [N, 3]
        """
        # Load data
        coords_A = torch.tensor(coords_A, dtype=torch.float32)
        coords_B = torch.tensor(coords_B, dtype=torch.float32)

        # Initialize parameters
        R = nn.Parameter(torch.eye(3))  # Initialize rotation matrix as identity
        t = nn.Parameter(torch.zeros(3))  # Initialize translation vector as zero

        # Set up the optimizer
        optimizer = optim.SGD([R, t], lr=0.05)

        # Training loop
        for i in range(10000):  # 1000 iterations
            optimizer.zero_grad()  # Clear previous gradients

            # Apply the transformation
            coords_A_pred = torch.mm(coords_B, R.t()) + t
            loss = nn.functional.mse_loss(coords_A_pred, coords_A)
            loss.backward()
            optimizer.step()

            if i % 500 == 0:  # Print loss every 100 iterations
                print(f"Iteration {i}: Loss = {loss.item()}")

        return R, t





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='conf/s02-enc-dec.yaml', help='config file path')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--mode', type=str, default='test')
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    with open(args.config) as f:
        kwargs = yaml.safe_load(f)
        f.close()
    ## backup config file
    if args.mode == 'train':
        logdir = os.path.join(kwargs['path']['logdir'], kwargs['path']['expname'])
        os.makedirs(logdir, exist_ok=True)
        copyfile(args.config, os.path.join(logdir,'config.yaml'))

    worker = LocGPT_Runner(**kwargs, mode=args.mode)
    if args.mode == 'train':
        worker.train_network()
    if args.mode == 'test':
        worker.eval_network()