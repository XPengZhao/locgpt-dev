import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from d2l import torch as d2l
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from tqdm import tqdm
from tensorboardX import SummaryWriter
logger = SummaryWriter('log')


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
l = 4

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2*l*16, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(0.15)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.15)
        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dp3 = nn.Dropout(0.10)
        self.fc4 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dp4 = nn.Dropout(0.10)
        self.fc5 = nn.Linear(256, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.dp5 = nn.Dropout(0.05)
        self.fc6 = nn.Linear(256, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.dp6 = nn.Dropout(0.05)
        self.fc7 = nn.Linear(256, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp7 = nn.Dropout(0.05)
        self.fc8_1 = nn.Linear(256, 1)
        self.fc8_2 = nn.Linear(256, 1)
        self.p = PositionalEncoding()

    def forward(self, x):
        x = self.p(x, l)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        # x = self.bn1(x)
        x = torch.relu(x)
        x = self.dp1(x)
        residual = x
        x = self.fc2(x)
        # x = self.bn2(x)
        x = torch.relu(x)
        x = self.dp2(x)
        x = self.fc3(x)
        # x = self.bn3(x)
        x = torch.relu(x)
        x = self.dp3(x)
        x = self.fc4(x)
        # x = self.bn4(x)
        x = torch.relu(x)
        x = self.dp4(x)
        x = self.fc5(x)
        # x = self.bn5(x)
        x = torch.relu(x)
        x = self.dp5(x)
        x = self.fc6(x)
        # x = self.bn6(x)
        x = torch.relu(x)
        x = self.dp6(x)
        x = self.fc7(x)
        # x = self.bn7(x)
        x = x + residual
        x = torch.relu(x)
        x = self.dp7(x)
        feature_map = x
        alpha = self.fc8_1(x)
        beta = self.fc8_2(x)
        return torch.cat((alpha, beta, feature_map), dim=1)


def spherical2pointandline(x):
    # print(x.shape)
    alpha, beta = x[:, 0].reshape((-1, 1)), x[:, 1].reshape((-1, 1))
    return torch.cat((torch.sin(beta) * torch.cos(alpha), torch.sin(beta) * torch.sin(alpha), torch.cos(beta)), dim=1)


def intersect_or_distance(p1, v1, p2, v2):
    p1.expand(v1.shape[0], 3)
    p2.expand(v2.shape[0], 3)
    if torch.all(torch.cross(v1, v2) == 0):
        dist = distance_between_lines(p1, v1, p2, v2)
        return 1, dist
    s = compute_intersection(p1, v1, p2, v2)
    if isinstance(s, int):
        return 3, distance_between_lines(p1, v1, p2, v2)
    return 2, s


# def compute_intersection(p1, v1, p2, v2):
#     v1_ = v1.unsqueeze(-1).reshape(-1, 3, 1)
#     v2_ = v2.unsqueeze(-1).reshape(-1, 3, 1)
#     A = torch.cat((v1_, v2_, torch.ones(v1_.shape).to(device)), dim=2)
#     b = (p2 - p1).squeeze()
#     x = torch.linalg.solve(A, b)
#     zero = x[:, 2].unsqueeze(-1)
#     if not torch.allclose(zero, torch.zeros_like(zero)):
#         return 1
#     p = p1 + v1 * x[:, 0].unsqueeze(-1)
#     return p
def compute_intersection(p1, v1, p2, v2):
    v1_ = v1.unsqueeze(-1).reshape(-1, 3, 1)
    v2_ = v2.unsqueeze(-1).reshape(-1, 3, 1)
    one = torch.ones(v1_.shape)
    A = torch.cat((v1_, v2_, one.to(v1.device)), dim=2)
    A_inv = torch.pinverse(A)
    b = (p2 - p1).squeeze()
    # x = torch.linalg.solve(A, b)
    x = torch.matmul(A_inv, b)
    zero = x[:, 2].unsqueeze(-1)
    if not torch.allclose(zero, torch.zeros_like(zero)):
        return 1
    p = p1 + v1 * x[:, 0].unsqueeze(-1)
    return p


def distance_between_lines(p1, v1, p2, v2):
    # Calculate the direction vector perpendicular to both lines
    cross_product = torch.cross(v1, v2, dim=1)
    # cross_product = torch.clamp(cross_product, min=1e-15)
    # Calculate the distance between the two lines
    distance = torch.abs(torch.einsum('ij, ij->i', (p2 - p1, cross_product))) / torch.norm(cross_product, dim=1)

    return distance.unsqueeze(1)


def area(x1, x2, x3):
    v1 = spherical2pointandline(x1)
    v2 = spherical2pointandline(x2)
    v3 = spherical2pointandline(x3)
    res = [intersect_or_distance(p1.to(x1.device), v1, p2.to(x2.device), v2), intersect_or_distance(p2.to(x2.device), v2, p3.to(x3.device), v3),
           intersect_or_distance(p3.to(x3.device), v3, p1.to(x1.device), v1)]
    zero = torch.zeros((x1.shape[0], 1))
    s = zero.to(v1.device)
    if res[0][0] + res[1][0] + res[2][0] == 6:
        s += tri(res[0][1], res[1][1], res[2][1])
    else:
        for flag, r in res:
            if flag != 2:
                s += r
    # print(s)
    return s


class MLP_(nn.Module):
    def __init__(self):
        super(MLP_, self).__init__()
        self.fc1 = nn.Linear(256*3, 256)
        # self.bn1 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(0.15)
        self.fc2 = nn.Linear(256, 256)
        # self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.15)
        self.fc3 = nn.Linear(256, 256)
        self.dp3 = nn.Dropout(0.10)
        self.fc4 = nn.Linear(256, 256)
        self.dp4 = nn.Dropout(0.10)
        self.fc5 = nn.Linear(256, 256)
        self.dp5 = nn.Dropout(0.05)
        self.fc6 = nn.Linear(256, 256)
        self.dp6 = nn.Dropout(0.05)
        self.fc7 = nn.Linear(256, 256)
        self.dp7 = nn.Dropout(0.05)
        self.fc8 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = torch.relu(x)
        x = self.dp1(x)
        residual = x
        x = self.fc2(x)
        # x = self.bn2(x)
        x = torch.relu(x)
        x = self.dp2(x)
        x = torch.relu(self.fc3(x))
        x = self.dp3(x)
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.dp4(x)
        x = self.fc5(x)
        x = torch.relu(x)
        x = self.dp5(x)
        x = torch.relu(self.fc6(x))
        x = self.dp6(x)
        x += residual
        x = torch.relu(self.fc7(x))
        x = self.dp7(x)
        x = self.fc8(x)
        return x


class network(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = MLP()
        self.mlp2 = MLP()
        self.mlp3 = MLP()
        self.mlp4 = MLP_()

    def forward(self, x):
        xx = x[:, 9:]
        x1 = self.mlp1(xx[:, :16])
        x2 = self.mlp2(xx[:, 16:32])
        x3 = self.mlp3(xx[:, 32:])
        s = area(x1[:, :-256], x2[:, :-256], x3[:, :-256], x[:, :3], x[:, 3:6], x[:, 6:9])
        p = self.mlp4(torch.cat((x1[:, 2:], x2[:, 2:], x3[:, 2:]), dim=1).reshape(-1, 256 * 3))
        return torch.cat((s, p), dim=1)


class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def forward(self, x, L):
        x = x.unsqueeze(-1).repeat(1, 1, L * 2)
        # print(x[:, :, 0::2])
        x[:, :, 0::2] = torch.pi * torch.pow(2, torch.arange(0, L, dtype=torch.float32)).to(x.device) * x[:, :, 0::2].clone()
        # print(x[:, :, 0::2])
        x[:, :, 1::2] = torch.pi * torch.pow(2, torch.arange(0, L, dtype=torch.float32)).to(x.device) * x[:, :, 1::2].clone()
        # print(x[:, :, 0::2])
        x[:, :, 0::2] = torch.sin(x[:, :, 0::2].clone())
        # print(x[:, :, 0::2])
        x[:, :, 1::2] = torch.cos(x[:, :, 1::2].clone())
        return x


def tri(a, b, d):
    def angle2point(alpha, beta, d):
        return torch.cat(
            (d * torch.sin(beta) * torch.cos(alpha), d * torch.sin(beta) * torch.sin(alpha), d * torch.cos(beta)),
            dim=1)

    def triangle_area(p_1, p_2, p_3):
        a_ = torch.norm(p_2 - p_3, dim=1)
        b_ = torch.norm(p_1 - p_3, dim=1)
        c = torch.norm(p_1 - p_2, dim=1)
        s_ = (a_ + b_ + c) / 2
        area_ = torch.sqrt(s_ * (s_ - a_) * (s_ - b_) * (s_ - c))
        return area_

    p1_ = angle2point(a[:, 0].reshape(-1, 1), b[:, 0].reshape(-1, 1), d[:, 0].reshape(-1, 1))
    p2_ = angle2point(a[:, 1].reshape(-1, 1), b[:, 1].reshape(-1, 1), d[:, 1].reshape(-1, 1))
    p3_ = angle2point(a[:, 2].reshape(-1, 1), b[:, 2].reshape(-1, 1), d[:, 2].reshape(-1, 1))
    s = triangle_area(p1_, p2_, p3_).reshape((-1, 1))
    print(s)
    return s


dis2me1 = lambda x, y: torch.norm(x - y)
dis2me = lambda x, y: np.linalg.norm(x - y)
dis2mse = lambda x, y: torch.mean((x - y) ** 2)


def minus2(y):
    #     y --> torch.Size(64,4)
    minus = []
    l = int(y.shape[0])
    for i in range(0, l, 2):
        minus.append(dis2me1(y[i, -3:], y[i + 1, -3:]))
    return torch.tensor(minus, requires_grad=True).reshape(-1, 1)


modelpath = '/home/wgs/loc/Antenna/Antenna/model_co'


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size, num_gpus):
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    train_ls, test_ls = [], []
    train_iter = load_array((train_features, train_labels), batch_size)
    valid_iter = load_array((test_features, test_labels), batch_size)
    devices = [try_gpu(i) for i in range(num_gpus)]
    net = nn.DataParallel(net, device_ids=devices)
    net = net.to(devices[0])
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00001)
    num_batches = len(train_iter)
    log_step_interval = 1000
    for epoch in range(num_epochs):
        net.train()
        total_loss_this_epoch = 0
        with tqdm(total=num_batches, desc=f"Train Epoch {epoch + 1}/{num_epochs}") as pbar:
            for step, (X, y) in enumerate(train_iter):
                # with autograd.detect_anomaly():
                optimizer.zero_grad()
                X, y = X.to(devices[0]), y.to(devices[0])
                l = loss(net(X), y)
                assert torch.isnan(l).sum() == 0, print(l)
                l.backward()
                # nn.utils.clip_grad_norm(net.parameters, 1, norm_type=2)
                optimizer.step()
                total_loss_this_epoch += l.item()
                global_iter_num = epoch * num_batches + step + 1
                pbar.update(1)
                pbar.set_postfix_str(f"train loss: {l.item():.6f}, lr: {optimizer.param_groups[0]['lr']:.6f}")
                if global_iter_num % log_step_interval == 0:
                    logger.add_scalar("train loss", l.item(), global_step=global_iter_num)
        scheduler.step()
        net.eval()
        total_valid_loss = 0
        num = len(valid_iter)
    # with tqdm(total=num, desc=f"Valid Epoch {epoch + 1}/{num_epochs}") as pbar:
        for step, (X, y) in enumerate(valid_iter):
            p = net(X)
            l = loss(p, y)
            total_valid_loss += l.item()
            global_step = epoch * num + step + 1
            # pbar.update(1)
            # pbar.set_postfix_str(f"valid loss: {l.item():.6f}, lr: {optimizer.param_groups[0]['lr']:.6f}")
            if global_step % 10 == 0:
                    logger.add_scalar("valid loss", l.item(), global_step=global_step)
        if (epoch+1) % 5 == 0:
            model_lst = [x for x in sorted(os.listdir(modelpath)) if x.endswith('.tar')]
            if len(model_lst) > 2:
                os.remove(modelpath + '/%s' % model_lst[0])
            path = os.path.join(modelpath, '{:06d}.tar'.format(epoch))
            torch.save({
                'epoch': epoch,
                'model': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)
        train_ls.append(total_loss_this_epoch / train_features.shape[0])
        if test_labels is not None:
            test_ls.append(total_valid_loss / test_features.shape[0])
    return train_ls, test_ls


def load_array(data_arrays, batch_size, is_train=True):
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    if is_train:
        data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True, num_workers=0)
    else:
        data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False, drop_last=True, num_workers=0)

    return data_iter


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx, :]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = network()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size, 1)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i >= 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
            plt.show()
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


def loss(x, y):
    def loss_l2(predict, label):
        # label:[batch-size, 4]
        if predict.shape[0] % 2 != 0:
            predict = predict[:-1]
            label = label[:-1]

        label_diff = label[0::2, 1:] - label[1::2, 1:]
        label_dis = torch.norm(label_diff, dim=1)

        predict_diff = predict[0::2, 1:] - predict[1::2, 1:]
        predict_dis = torch.norm(predict_diff, dim=1)

        length = int(label.shape[0] / 2)

        label_diff1 = label[:length, 1:] - label[length:, 1:]
        label_dis1 = torch.norm(label_diff1, dim=1)

        predict_diff1 = predict[:length, 1:] - predict[length:, 1:]
        predict_dis1 = torch.norm(predict_diff1, dim=1)

        label_diff2 = label[:length, 1:] - torch.flip(label, [0])[:length, 1:]
        label_dis2 = torch.norm(label_diff2, dim=1)

        predict_diff2 = predict[:length, 1:] - torch.flip(predict, [0])[:length, 1:]
        predict_dis2 = torch.norm(predict_diff2, dim=1)

        predict_dis = torch.cat((predict_dis, predict_dis1, predict_dis2), dim=0)
        label_dis = torch.cat((label_dis, label_dis1, label_dis2), dim=0)

        # indices = torch.nonzero(label_dis <= 2., as_tuple=True)[0]

        # predict_dis = torch.index_select(predict_dis, dim=0, index=indices)

        # label_dis = torch.index_select(label_dis, dim=0, index=indices)
        # print(label_dis.shape[0])
        return dis2mse(predict_dis, label_dis)

    mse = nn.MSELoss()

    l1 = mse(x[:, 0], y[:, 0])
    l2 = loss_l2(x, y)
    # idx = torch.randperm(x.shape[0])
    # x1 = x[idx]
    # y1 = y[idx]
    # l3 = loss_l2(x1, y1)
    # l2 += l3

    return 3 * l1 + 0*l2


def loss_dist(x, y):
    mse = nn.MSELoss()
    return mse(x[:, 1:], y[:, 1:])


if torch.cuda.is_available():
    device = torch.device('cuda:1')
    print('CUDA is available')
else:
    device = torch.device('cpu')
    print('CUDA is not available')

p1 = torch.tensor([[3.0017452239990234,3.983450412750244,0.34878233075141907]])
p2 = torch.tensor([[-2.931766986846924,4.035231113433838,0.34878233075141907]])
p3 = torch.tensor([[0.17407386004924774,-4.984780788421631,0.34878233075141907]])

error_l = []


def train_and_pred(train_features, test_features, train_labels, test_labels,
                   num_epochs, lr, weight_decay, batch_size, modelpath, num_gpus):
    ckpts = [os.path.join(modelpath, f) for f in sorted(os.listdir(modelpath)) if 'tar' in f]
    if len(ckpts) == 0:
        raise Exception('No previous model found, please check it')
    else:
        print('Found ckpts', ckpts)
        ckpt_path = ckpts[-1]
        print('Reload from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        net = network()
        # net.load_state_dict(ckpt['model'])
        # net.cuda()
        devices = [try_gpu(i) for i in range(num_gpus)]
        net = nn.DataParallel(net, device_ids=devices).to(devices[0])
        net.load_state_dict(ckpt['model'])
        net.eval()
        preds = net(test_features).cpu().detach().numpy()
        for i in range(test_labels.shape[0]):
            error = dis2me(preds[i, 1:], test_labels[i, 1:])
            error_l.append(error)
            # print(f'error {i}:', error)
        print('Median error', np.median(error_l))
        print('Average error', np.average(error_l))
        df_ = pd.DataFrame(preds.reshape((-1, 4)), columns=['s', 'x', 'y', 'z'])
        df_['Location error'] = error_l
        df_.to_csv('submission_ss.csv', index=False)


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
