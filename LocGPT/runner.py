# -*- coding: utf-8 -*-
"""LocGPT runner
"""
import argparse
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from shutil import copyfile

import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import yaml
from einops import rearrange
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm

from logger import logger_config
from model import LocGPT

dis2mse = lambda x, y: torch.mean((x - y) ** 2)
dis2me = lambda x, y: np.linalg.norm(x - y, axis=-1)


class MyDataset(Dataset):
    def __init__(self, data_dir):

        self.data = torch.load(data_dir)  #[datalen, n_seq, timestamp+area+pos+spt]
        self.enc_token = torch.arange(len(self.data)).unsqueeze(1) #[datalen, 1]
        self.dec_token = torch.arange(len(self.data)).unsqueeze(1) #[datalen, 1]

        self.labels = self.data[..., :5]  # [datalen, n_seq, timestamp+area+pos]
        self.spt = self.data[..., 5:]


    def loaddata(self):
        return self.enc_token, self.spt, self.dec_token, self.labels



class LocGPT_Runner():
    def __init__(self, mode, **kwargs) -> None:

        kwargs_path = kwargs['path']
        kwargs_dataset = kwargs['dataset']
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

        # Dataset
        self.gateways_pos = kwargs_dataset['gateways_pos']
        self.gateways_pos = torch.tensor(self.gateways_pos, dtype=torch.float32).to(self.devices)
        self.n_seq = kwargs_dataset['n_seq']

        # Logger
        log_filename = "logger.log"
        log_savepath = os.path.join(self.logdir, self.expname, log_filename)
        self.logger = logger_config(log_savepath=log_savepath, logging_name='locgpt')
        self.logger.info("expname:%s, datadir:%s, logdir:%s, train_file:%s, test_file:%s",
                          self.expname, self.datadir, self.logdir, self.train_file, self.test_file)
        self.logger_tb = SummaryWriter(os.path.join(self.logdir, self.expname, 'tensorboard'))


        ## Network
        self.locgpt = LocGPT(self.gateways_pos, **kwargs_network["transformer"]).to(self.devices)
        if kwargs_network['init_weight'] and mode=='train':
            self.locgpt.apply(self.init_weights)

        params = list(self.locgpt.parameters())
        self.optimizer = optim.Adam(params, lr=float(kwargs_train['lr']),
                                    weight_decay=float(kwargs_train['weight_decay']))
        self.cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                    T_max=kwargs_train['T_max'],eta_min=1e-6,
                                                                    last_epoch=-1)

        ## Train settings
        self.epoch_start = 1
        if kwargs_path['load_ckpt'] or mode=='test':
            self.load_checkpoints()
        self.current_epoch = self.epoch_start
        self.batch_size = kwargs_train['batch_size']
        self.total_epoches = kwargs_train['total_epoches']
        self.beta = kwargs_train['beta']
        self.i_save = kwargs_train['i_save']
        self.mask_id = 0

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
                                                      shuffle=True, drop_last=False, num_workers=0)
        self.test_iter = torch.utils.data.DataLoader(test_dataset, self.batch_size,
                                                     shuffle=False, drop_last=False, num_workers=0)
        self.logger.debug("transform_iter length:%s, train_iter length:%s, test_iter length:%s",
                           len(self.transform_iter.dataset), len(self.train_iter.dataset), len(self.test_iter.dataset))




    def init_weights(self, m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)


    def load_checkpoints(self):
        """load checkpoints and epoch
        """
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
            self.epoch_start = ckpt['epoch_start'] + 1


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


    def criterion(self, x, y, mask):
        """
        Parameters
        --------------
        x: [B, n_seq, 4]. area+pos
        mask: [B, n_seq]
        """
        mask = torch.logical_not(mask)

        l1 = x[..., 0][mask]
        l1 = torch.mean(l1**2)

        l2 = torch.linalg.norm(x[..., 1:] - y[..., 1:], dim=-1)
        l2 = torch.mean(l2[mask] ** 2)
        l3 = self.beta * l1 + (1-self.beta) * l2

        return l1, l2, l3


    def get_random_mask(self, B, n_seq):
        """randomly sample in one sequence
        Return
        -------------
        mask: [B, n_seq]
        """

        ## mask 1
        mask = torch.zeros((B, n_seq))

        ## mask 2
        # mask = torch.ones((B, n_seq)) # Initialize the mask with all ones
        # mask_start = torch.randint(0, n_seq, (B,))  # [B,]
        # mask_len = torch.randint(1, n_seq+1, (B,))
        # mask_end = torch.min(mask_start + mask_len, torch.tensor(n_seq))
        # for i in range(B):
        #     mask[i, mask_start[i]:mask_end[i]] = 0

        ## mask 3
        mask = torch.ones((B, n_seq)) # Initialize the mask with all ones
        mask[:, 0:self.mask_id+1] = 0
        self.mask_id = (self.mask_id + 1) % n_seq


        mask = mask.eq(1).to(self.devices)   # B, seq
        return mask



    def train_network(self):
        """auto regressing mode
        """

        self.locgpt.train()
        total_num = len(self.train_iter.dataset)
        num_batches = len(self.train_iter)
        log_step_interval = 1
        print(total_num, num_batches)
        for epoch in range(self.epoch_start, self.total_epoches+1):
            with tqdm(total=num_batches, desc=f"Epoch {epoch}/{self.total_epoches}") as pbar:
                for step, (enc_token, dec_token) in enumerate(self.train_iter):
                    spt = self.train_spt[enc_token.view(-1)].to(self.devices)   # [B, n_seq, 3*9*36]
                    label = self.train_label[dec_token.view(-1)].to(self.devices)
                    timestamp = label[..., 0:1]    # [B, n_seq, 1]
                    timestamp = timestamp - timestamp[:, 0:1, :]  # relative timestamp to seq start
                    area_tagpos = label[..., 1:5]    # [B, n_seq, 4]

                    ind = torch.arange(self.n_seq)
                    enc_token = enc_token*self.n_seq + ind
                    dec_token = dec_token*self.n_seq + ind
                    enc_token = enc_token.to(self.devices, dtype=torch.int32)
                    dec_token = dec_token.to(self.devices, dtype=torch.int32)    #[B, n_seq]
                    B, n_seq = enc_token.shape

                    start_token = torch.zeros((B, 1, 3), dtype=torch.float32).to(self.devices) # [B, 1, 3]
                    dec_input_chunk = start_token

                    mask = self.get_random_mask(B, n_seq)
                    enc_token.masked_fill_(mask, -1)
                    dec_token.masked_fill_(mask, -1)

                    # Initialize a tensor to store the outputs
                    outputs = torch.zeros_like(area_tagpos)
                    self.optimizer.zero_grad()
                    for j in range(1, n_seq+1):
                        enc_token_chunk, dec_token_chunk = enc_token[:, :j], dec_token[:, :j]
                        timestamp_chunk, spt_chunk = timestamp[:, 0:j, :], spt[:, 0:j, :]
                        output = self.locgpt(timestamp_chunk, enc_token_chunk, spt_chunk,
                                             dec_token_chunk, dec_input_chunk)  # [B, n_seq, 4]
                        input_mask =  torch.logical_not(mask[:, j-1:j]).unsqueeze(-1)
                        dec_input_chunk = torch.concat((dec_input_chunk, input_mask*output[:,-1:,1:]), dim=1)
                        outputs[:, j-1:j, :] = output[:,-1:,:]

                    l1, l2, l3 = self.criterion(output, area_tagpos, mask)
                    loss = l3
                    loss.backward()
                    self.optimizer.step()
                    global_iter_num = epoch * num_batches + step + 1
                    pbar.update(1)
                    pbar.set_postfix_str(f"l1 loss: {l1.item():.6f}, l2 loss: {l2.item():.6f}, lr: {self.optimizer.param_groups[0]['lr']:.9f}")
                    if global_iter_num % log_step_interval == 0:
                        self.logger_tb.add_scalar("l1 loss", l1.item(), global_step=global_iter_num)
                        self.logger_tb.add_scalar("l2 loss", l2.item(), global_step=global_iter_num)
            self.cosine_schedule.step()
            self.current_epoch = epoch

            if self.current_epoch % self.i_save == 0:
                self.saveckpts()




    def pred(self, dataset, spt_set, label_set):
        """
        Returns
        -----------
        pred_all: [B, n_seq, 4]. predict results (s, x, y, z)
        gt_all: [B, n_seq, 4]. ground truth results (s, x, y, z)
        """
        self.locgpt.eval()
        dataset_len = len(dataset.dataset)
        gt_all = np.zeros((dataset_len, self.n_seq, 4))
        pred_all = np.zeros((dataset_len, self.n_seq, 4))

        for i, (enc_token, dec_token) in enumerate(dataset):

            spt = spt_set[enc_token.view(-1)].to(self.devices)   # [B, n_seq, 3*9*36]
            label = label_set[dec_token.view(-1)]
            timestamp = label[..., 0:1].to(self.devices)    # [B, n_seq, 1]
            timestamp = timestamp - timestamp[:, 0:1, :]
            area_tagpos = label[..., 1:5]    # [B, n_seq, 4]
            ind = torch.arange(self.n_seq)
            enc_token = enc_token*self.n_seq + ind
            dec_token = dec_token*self.n_seq + ind
            enc_token = enc_token.to(self.devices, dtype=torch.int32)
            dec_token = dec_token.to(self.devices, dtype=torch.int32)    #[B, n_seq]

            B, n_seq = enc_token.shape
            start_token = torch.zeros((B, 1, 3), dtype=torch.float32).to(self.devices) # [B, 1, 3]
            dec_input_chunk = start_token

            preds = np.zeros((B, n_seq, 4))
            with torch.no_grad():
                for j in range(1, n_seq+1):
                    enc_token_chunk, dec_token_chunk = enc_token[:, :j], dec_token[:, :j]
                    timestamp_chunk, spt_chunk = timestamp[:, 0:j, :], spt[:, 0:j, :]
                    output = self.locgpt(timestamp_chunk, enc_token_chunk, spt_chunk, dec_token_chunk, dec_input_chunk)  # [B, n_seq, 4]
                    pos_current = output[:,j-1,:].cpu().detach()
                    pos_label = area_tagpos[:, j-1,:]
                    preds[:,j-1,:] = pos_current
                    dec_input_chunk = torch.concat((dec_input_chunk, output[:,-1:,1:]), dim=1)
                pred_all[i*self.batch_size:(i+1)*self.batch_size] = preds
                gt_all[i*self.batch_size:(i+1)*self.batch_size] = area_tagpos

        # pred_all = rearrange(pred_all, 'b n d -> (b n) d')
        # gt_all = rearrange(gt_all, 'b n d -> (b n) d')

        return pred_all, gt_all


    def eval_network(self):

        self.logger.info("Start eval network----------------------------------")
        R,t = self.get_transform()

        self.logger.info("before transform---------------------------")
        pred_all_test, gt_all_test = self.pred(self.test_iter, self.test_spt, self.test_label)
        points_preds_test, points_labels_test = pred_all_test[..., 1:], gt_all_test[..., 1:]
        pos_error_test = np.linalg.norm(points_labels_test-points_preds_test, axis=-1)
        self.logger.info('Location error on testing each trace set:\n mean:%s\n std:%s',
                         np.median(pos_error_test, axis=0), np.std(pos_error_test, axis=0))
        self.logger.info('Location error on testing globally set:\n mean:%s\n std:%s',
                         np.median(pos_error_test), np.std(pos_error_test))


        # After Transform
        self.logger.info("after transform---------------------------")
        points_preds_test_A = points_preds_test @ R.T + t
        pos_error_test = np.linalg.norm(points_labels_test-points_preds_test_A, axis=-1)
        self.logger.info('After transform, Location error on testing each trace set:\n mean:%s\n std:%s',
                         np.median(pos_error_test, axis=0), np.std(pos_error_test, axis=0))
        self.logger.info('After transform, Location error on testing globally set:\n mean:%s\n std:%s',
                         np.median(pos_error_test), np.std(pos_error_test))


        scio.savemat(os.path.join(self.logdir, self.expname, "test_pos_pred.mat"),
                     {"points_preds":points_preds_test_A,
                      "points_labels":points_labels_test,
                      "pos_error":pos_error_test})





    def get_transform(self):

        # # R = 3x3 rotation matrix
        # # t = 3 column vector
        # # B = A@R.T + t
        pred_all_train, gt_all_train = self.pred(self.transform_iter, self.train_spt, self.train_label)
        points_preds_train, points_labels_train = pred_all_train[..., 1:], gt_all_train[..., 1:]
        pos_error_train = np.linalg.norm(points_labels_train-points_preds_train, axis=-1)
        self.logger.info('Location error on training set each trace median:%s, std:%s',
                         np.median(pos_error_train, axis=0), np.std(pos_error_train, axis=0))
        self.logger.info('Location error on training set globally median:%s, std:%s',
                         np.median(pos_error_train), np.std(pos_error_train))

        points_preds_train = rearrange(points_preds_train, 'b n d -> (b n) d')
        points_labels_train = rearrange(points_labels_train, 'b n d -> (b n) d')

        # SVD method
        R, t = self.learn_Rt(points_labels_train, points_preds_train)
        R, t = R.detach().numpy(), t.detach().numpy()

        points_preds_train_A = points_preds_train @ R.T + t
        points_preds_train_A = rearrange(points_preds_train_A, '(b n) d -> b n d', n=self.n_seq)
        points_labels_train = rearrange(points_labels_train, '(b n) d -> b n d', n=self.n_seq)
        pos_error_train = np.linalg.norm(points_preds_train_A-points_labels_train, axis=-1)
        self.logger.info('After transform, Location error on training set each trace median:%s, std:%s',
                         np.median(pos_error_train, axis=0), np.std(pos_error_train, axis=0))
        self.logger.info('After transform, Location Error on training set globally median:%s, std:%s',
                         np.median(pos_error_train), np.std(pos_error_train))

        ## save results and loggers
        scio.savemat(os.path.join(self.logdir, self.expname, "train_pos_pred.mat"),
                     {"points_preds":points_preds_train,
                      "points_labels":points_labels_train,
                      "pos_error":pos_error_train})

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
        for i in range(10001):  # 1000 iterations
            optimizer.zero_grad()  # Clear previous gradients

            # Apply the transformation
            coords_A_pred = torch.mm(coords_B, R.t()) + t
            loss = nn.functional.mse_loss(coords_A_pred, coords_A)
            loss.backward()
            optimizer.step()

            if i % 5000 == 0:  # Print loss every 100 iterations
                print(f"Iteration {i}: Loss = {loss.item()}")

        return R, t



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='conf/mcbench/mcbench-s02.yaml', help='config file path')
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--mode', type=str, default='train')
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