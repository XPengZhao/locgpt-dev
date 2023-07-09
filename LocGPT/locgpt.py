import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset, random_split
import timm
import pandas as pd
from tqdm import tqdm
import numpy as np
from torchvision import models
import torchvision.datasets as datasets
from transformers import ViTImageProcessor, ViTModel
from torchtoolbox.tools import mixup_data, mixup_criterion
from torchtoolbox.transform import Cutout
import os
from torch.autograd import Variable
from PIL import Image
from loc_comdel import area
from tensorboardX import SummaryWriter
abs_path = os.path.abspath(os.path.dirname(__file__))
abs_path += '/'
logger = SummaryWriter(abs_path + 'log_vit_s02')
modelpath = abs_path + 'model_vit_s02'
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
transform_ini = transforms.Compose([
    transforms.ToTensor(),
])
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    Cutout(),
    transforms.ToTensor(),
    # transforms.Normalize([0.24592678, 0.24547586, 0.24202907], [0.2228598, 0.22451608, 0.24079134])
    # transforms.Normalize([0.24787958, 0.24733353, 0.24400552], [0.22422023, 0.22590108, 0.24233654])
    transforms.Normalize([0.2501609, 0.2415813, 0.25076947], [0.22090364, 0.19479173, 0.24216957])
])
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize([0.24592678, 0.24547586, 0.24202907], [0.2228598, 0.22451608, 0.24079134])
    transforms.Normalize([0.2476261, 0.24735165, 0.24419768], [0.22461787, 0.22624663, 0.24236584])
])

model_single = timm.create_model('vit_base_patch16_224', pretrained=False)
model_single.blocks = model_single.blocks[:3]
num_ftrs = model_single.head.in_features
model_single.head = nn.Linear(num_ftrs, 3)

def vit_worker():
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.patch_embed.proj = nn.Conv2d(1, 256, kernel_size=(16, 16), stride=(16, 16))
    model.blocks = model.blocks[:2]
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, 2)

    return model

model_ch1 = vit_worker()
model_ch2 = vit_worker()
model_ch3 = vit_worker()

class TNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=256, output_ch=3, skips=[4]):
        """input + 8 hidden layer + output

        Parameters
        ----------
        D : hidden layer number
        W : Dimension per hidden layer
        input_ch : int, input channel, by default 256
        output_ch : int, out channel, by default 3
        skip : list, residual layer, optional
        """
        super().__init__()
        self.D, self.W = D, W
        self.input_ch = input_ch
        self.output_ch = output_ch

        self.skips = skips
        self.num_layers = D + 2    # hidden + input + output

        self.linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)
             for i in range(D - 1)]
        )
        # self.pos_bn = nn.ModuleList([nn.BatchNorm1d(W) for i in range(D)])
        self.output_layer = nn.Linear(W, output_ch)

        geometric_init = False
        weight_norm = False
        if geometric_init:
            for i, layer in enumerate(self.linears):
                # final hidden layer -> output layer
                if i == self.num_layers - 2:
                    torch.nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(layer.in_features), std=0.0001)
                    torch.nn.init.constant_(layer.bias, -0.5)

                # input layer -> first hidden layer
                elif i == 0:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(layer.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(layer.out_features))
                # skip layer -> next hidden layer
                elif i in self.skips:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(layer.out_features))
                    torch.nn.init.constant_(layer.weight[:, -(self.input_ch):], 0.0)
                else:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(layer.out_features))

                if weight_norm:
                    self.linears[i] = nn.utils.weight_norm(layer)

        self.activation = nn.Softplus(beta=100)



    def forward(self, inputs):
        """forward function of the model

        Parameters
        ----------
        x : [batchsize ,input]

        Returns
        ----------
        outputs: [batchsize, 3].   position
        """
        inputs = inputs.to(torch.float32)
        h = inputs

        for i, layer in enumerate(self.linears):
            h = layer(h)
            # h = F.relu(h)
            h = self.activation(h)
            if i in self.skips:
                h = torch.cat([inputs, h], -1)

        pos = self.output_layer(h)    # (batch_size, 3)
        return pos


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val=0.1):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]).to(x.device) * self.variance




class MyVit(nn.Module):
    def __init__(self):
        super(MyVit, self).__init__()
        self.vit1 = model_ch1
        self.vit2 = model_ch2
        self.vit3 = model_ch3
        self.vit4 = model_single
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.mlp = TNetwork(input_ch=768)
        self.pe1 = nn.Linear(3, 256)
        self.pe2 = nn.Linear(3, 256)
        self.pe3 = nn.Linear(3, 256)
        # self.alpha1 = nn.Linear(256, 1)
        # self.beta1 = nn.Linear(256, 1)
        # self.alpha2 = nn.Linear(256, 1)
        # self.beta2 = nn.Linear(256, 1)
        # self.alpha3 = nn.Linear(256, 1)
        # self.beta3 = nn.Linear(256, 1)


    def forward(self, x, y):
        B, C, H, W = x.shape
        # x1 = torch.cat((x[:, 0, :, :].unsqueeze(1), x[:, 0, :, :].unsqueeze(1), x[:, 0, :, :].unsqueeze(1)), dim=1)
        # x2 = torch.cat((x[:, 1, :, :].unsqueeze(1), x[:, 1, :, :].unsqueeze(1), x[:, 1, :, :].unsqueeze(1)), dim=1)
        # x3 = torch.cat((x[:, 2, :, :].unsqueeze(1), x[:, 2, :, :].unsqueeze(1), x[:, 2, :, :].unsqueeze(1)), dim=1)
        # print(x1.shape)
        x1 = x[:, 0, :, :].unsqueeze(1)
        x2 = x[:, 1, :, :].unsqueeze(1)
        x3 = x[:, 2, :, :].unsqueeze(1)
        # x1 = x[:, 0, :, :].repeat(1, 3, 1, 1)
        # x2 = x[:, 1, :, :].repeat(1, 3, 1, 1)
        # x3 = x[:, 2, :, :].repeat(1, 3, 1, 1)
        x1, l1 = self.vit1(x1)
        x2, l2 = self.vit2(x2)
        x3, l3 = self.vit3(x3)
        l1 = l1[:, 0]
        l2 = l2[:, 0]
        l3 = l3[:, 0]
        # l1 = torch.mean(l1[:, 1:], dim=1)
        # l2 = torch.mean(l2[:, 1:], dim=1)
        # l3 = torch.mean(l3[:, 1:], dim=1)
        # x1_1 = self.alpha1(x1)
        # x1_2 = self.beta1(x1)
        # x2_1 = self.alpha2(x2)
        # x2_2 = self.beta2(x2)
        # x3_1 = self.alpha3(x3)
        # x3_2 = self.beta3(x3)
        # x_angle_1 = torch.cat((x1_1, x1_2), dim=1)
        # x_angle_2 = torch.cat((x2_1, x2_2), dim=1)
        # x_angle_3 = torch.cat((x3_1, x3_2), dim=1)
        x_angle_1 = x1
        x_angle_2 = x2
        x_angle_3 = x3
        s = area(x_angle_1, x_angle_2, x_angle_3)
        p1 = self.pe1(y[:, :3])
        p2 = self.pe2(y[:, 3:6])
        p3 = self.pe3(y[:, 6:])
        # p1 = y[:, :3].reshape(B, 1, 3, 1)
        # p2 = y[:, 3:6].reshape(B, 1, 3, 1)
        # p3 = y[:, 6:].reshape(B, 1, 3, 1)
        # p1 = F.interpolate(p1, size=(224, 224), mode='bilinear', align_corners=False)
        # p2 = F.interpolate(p2, size=(224, 224), mode='bilinear', align_corners=False)
        # p3 = F.interpolate(p3, size=(224, 224), mode='bilinear', align_corners=False)
        # p = torch.cat((p1, p2, p3), dim=1)
        # f1 = self.fc1(l1).reshape(B, 16, 16, 1).permute(0, 3, 1, 2)
        # f2 = self.fc2(l2).reshape(B, 16, 16, 1).permute(0, 3, 1, 2)
        # f3 = self.fc3(l3).reshape(B, 16, 16, 1).permute(0, 3, 1, 2)
        # f = torch.cat((f1, f2, f3), dim=1)
        # f = F.interpolate(f, size=(224, 224), mode='bilinear', align_corners=False)
        # f = p + f
        # f1 = self.fc1(l1)
        # f2 = self.fc2(l2)
        # f3 = self.fc3(l3)
        f = torch.cat((l1, l2, l3), dim=1)
        # print(f)
        p = self.mlp(f)
        # p, _ = self.vit4(f)
        return torch.cat((s, p), dim=1)


class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.fc1 = nn.Linear(256*3, 256)
        self.dp1 = nn.Dropout(0.15)
        self.fc2 = nn.Linear(256, 256)
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
        x = torch.relu(x)
        x = self.dp1(x)
        residual = x
        x = self.fc2(x)
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
        x += residual
        x = torch.relu(self.fc6(x))
        x = self.dp6(x)
        x = torch.relu(self.fc7(x))
        x = self.dp7(x)
        x = self.fc8(x)
        return x


# class MyVit(nn.Module):
#     def __init__(self):
#         super(MyVit, self).__init__()
#         self.vit1 = model
#         self.vit2 = model
#         self.vit3 = model
#         self.fc1 = nn.Linear(256, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.alpha1 = nn.Linear(256, 1)
#         self.beta1 = nn.Linear(256, 1)
#         self.alpha2 = nn.Linear(256, 1)
#         self.beta2 = nn.Linear(256, 1)
#         self.alpha3 = nn.Linear(256, 1)
#         self.beta3 = nn.Linear(256, 1)
#         self.mlp = network()
#         # self.mlp = TNetwork()
#         self.pe1 = nn.Linear(3, 256)
#         self.pe2 = nn.Linear(3, 256)
#         self.pe3 = nn.Linear(3, 256)


#     def forward(self, x, y):
#         B, C, H, W = x.shape
#         # x1 = torch.cat((x[:, 0, :, :].unsqueeze(1), x[:, 0, :, :].unsqueeze(1), x[:, 0, :, :].unsqueeze(1)), dim=1)
#         # x2 = torch.cat((x[:, 1, :, :].unsqueeze(1), x[:, 1, :, :].unsqueeze(1), x[:, 1, :, :].unsqueeze(1)), dim=1)
#         # x3 = torch.cat((x[:, 2, :, :].unsqueeze(1), x[:, 2, :, :].unsqueeze(1), x[:, 2, :, :].unsqueeze(1)), dim=1)
#         x1 = x[:, 0, :, :].unsqueeze(1)
#         x2 = x[:, 1, :, :].unsqueeze(1)
#         x3 = x[:, 2, :, :].unsqueeze(1)
#         x1 = self.vit1(x1)
#         x2 = self.vit2(x2)
#         x3 = self.vit3(x3)
#         x1_1 = self.alpha1(x1)
#         x1_2 = self.beta1(x1)
#         x2_1 = self.alpha2(x2)
#         x2_2 = self.beta2(x2)
#         x3_1 = self.alpha3(x3)
#         x3_2 = self.beta3(x3)
#         x_angle_1 = torch.cat((x1_1, x1_2), dim=1)
#         x_angle_2 = torch.cat((x2_1, x2_2), dim=1)
#         x_angle_3 = torch.cat((x3_1, x3_2), dim=1)
#         s = area(x_angle_1, x_angle_2, x_angle_3)
#         # p1 = y[:, :3].reshape(B, 1, 3, 1)
#         # p2 = y[:, 3:6].reshape(B, 1, 3, 1)
#         # p3 = y[:, 6:].reshape(B, 1, 3, 1)
#         # p1 = F.interpolate(p1, size=(256, 1), mode='bilinear', align_corners=False).reshape(B, 256)
#         # p2 = F.interpolate(p2, size=(256, 1), mode='bilinear', align_corners=False).reshape(B, 256)
#         # p3 = F.interpolate(p3, size=(256, 1), mode='bilinear', align_corners=False).reshape(B, 256)
#         p1 = self.pe1(y[:, :3])
#         p2 = self.pe2(y[:, 3:6])
#         p3 = self.pe3(y[:, 6:])
#         x1, x2, x3 = p1+x1, p2+x2, p3+x3
#         # x_all = x1 + x2 + x3
#         x_all = torch.cat((x1, x2, x3), dim=1)
#         p = self.mlp(x_all)
#         return torch.cat((s, p), dim=1)


def criterion(x, y, beta=0.5):
    def loss_l2(preds, labels):
        l_ = preds.shape[0]
        w = torch.tensor([(i, j) for i in range(l_ - 1) for j in range(i + 1, l_)]).to(preds.device)
        diff_preds = preds[w[:, 0], 1:] - preds[w[:, 1], 1:]
        diff_preds = torch.norm(diff_preds, dim=1)
        diff_labels = labels[w[:, 0], 1:] - labels[w[:, 1], 1:]
        diff_labels = torch.norm(diff_labels, dim=1)
        # ind_ = torch.nonzero((diff_labels <= 1.), as_tuple=True)[0]
        # w = torch.index_select(w, dim=0, index=ind_)
        # print(diff_preds.shape)
        # diff_preds = torch.index_select(diff_preds, dim=0, index=ind_)
        # print(diff_preds.shape)
        # total.append(diff_labels.shape[0])
        # diff_labels = torch.index_select(diff_labels, dim=0, index=ind_)
        # util.append(diff_labels.shape[0])
        return dis2mse(diff_preds, diff_labels), torch.unique(w, return_counts=True)[1].reshape(-1, 1)


    mse = nn.MSELoss()
    l1 = mse(x[:, 0], y[:, 0])
    # l2 = mse(x[:, 1:], y[:, 1:])
    # l3 = l1 + l2
    l2, _ = loss_l2(x, y)
    # l3 = 32*31*l1 + l2
    l3 = beta * l1 + (1-beta) * l2
    # l3 = l2

    return l1, l2, l3


class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = []
        with open(data_dir, "r") as f:
            for line in f.readlines():
                items = line.strip().split(",")
                img_path = items[0]
                label = torch.tensor(list(map(float, items[1:])))
                self.data.append((img_path, label))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        with open(img_path, "rb") as f:
            img = Image.open(f).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label


lr = 1e-4
weight_decay = 1e-4
BATCH_SIZE = 32
EPOCHS = 5000
ckpts = [os.path.join(modelpath, f) for f in sorted(os.listdir(modelpath)) if 'tar' in f]
flag = False
net = MyVit()
beta = SingleVarianceNetwork()
# if len(ckpts) == 0:
#     pass
    # raise Exception('No previous model found, please check it')
# else:
#     flag = True
#     print('Found ckpts', ckpts)
#     ckpt_path = ckpts[-1]
#     print('Reload from', ckpt_path)
#     ckpt = torch.load(ckpt_path)
    # net.load_state_dict(ckpt['model'])
# net.to(device)
data_dir = abs_path + 'train_data-s02.csv'
train_data_dir = abs_path + 'train_data-s02.csv'
test_data_dir = abs_path + 'test_data-s02.csv'
dataset = MyDataset(data_dir, transform=transform_ini)
train_dataset = MyDataset(train_data_dir, transform=transform)
test_dataset = MyDataset(test_data_dir, transform=transform_test)
train_iter = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
test_iter = torch.utils.data.DataLoader(test_dataset, 32, shuffle=False, drop_last=False, num_workers=0)
params = list(net.parameters())
optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)

optimizer = torch.optim.Adam([{"params": params}, {"params": beta.parameters(), "lr": lr * 10}], lr=lr,
                             weight_decay=weight_decay)

cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=20,eta_min=1e-5)
dis2mse = lambda x, y: torch.mean((x - y) ** 2)
dis2me = lambda x, y: np.linalg.norm(x - y)


def train(model, beta_net, device, train_loader, optimizer, num_epochs, modelpath, num_gpus=1):
    devices = [try_gpu(i) for i in range(num_gpus)]
    model = nn.DataParallel(model, device_ids=devices).to(devices[0])
    beta_net = nn.DataParallel(beta_net, device_ids=devices).to(devices[0])

    if not flag:
        model.apply(init_weights)
    model.train()
    total_num = len(train_loader.dataset)
    num_batches = len(train_loader)
    log_step_interval = 1
    print(total_num, num_batches)
    for epoch in range(num_epochs):
        with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for step, (data, target) in enumerate(train_loader):
                # with torch.autograd.set_detect_anomaly(True):
                data, target = data.to(devices[0], non_blocking=True), target.to(devices[0], non_blocking=True)
                label = target[:, :4]
                pos = target[:, 4:]
                optimizer.zero_grad()
                output = model(data, pos)
                beta = beta_net(torch.zeros([1, 3]))[:, :1].clip(1e-2, 1)
                l1, l2, l3 = criterion(output, label, beta)
                loss = l3
                loss.backward()

                # if epoch > 20:
                #     for name, param in model.named_parameters():
                #         if param.grad is not None:
                #             if 'module.mlp' in name:
                #                 print(name, param.grad)

                optimizer.step()
                global_iter_num = epoch * num_batches + step + 1
                pbar.update(1)
                pbar.set_postfix_str(f"l1 loss: {l1.item():.6f}, l2 loss: {l2.item():.6f}, lr: {optimizer.param_groups[0]['lr']:.9f}, beta: {beta.item():.6f}")
                if global_iter_num % log_step_interval == 0:
                    logger.add_scalar("l1 loss", l1.item(), global_step=global_iter_num)
                    logger.add_scalar("l2 loss", l2.item(), global_step=global_iter_num)
        cosine_schedule.step()
        if (epoch+1) % 1 == 0:
            model_lst = [x for x in sorted(os.listdir(modelpath)) if x.endswith('.tar')]
            if len(model_lst) > 2:
                os.remove(modelpath + '/%s' % model_lst[0])
            path = os.path.join(modelpath, '{:06d}.tar'.format(epoch))
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'beta_net': beta_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)


ACC=0
def val(model, device, test_loader, epoch):
    global ACC
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output.data, 1)
            correct += torch.sum(pred == target)
            print_loss = loss.data.item()
            test_loss += print_loss
        correct = correct.data.item()
        acc = correct / total_num
        avgloss = test_loss / len(test_loader)
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avgloss, correct, len(test_loader.dataset), 100 * acc))
        if acc > ACC:
            torch.save(model, 'model_' + str(epoch) + '_' + str(round(acc, 3)) + '.pth')
            ACC = acc


# val(model, device, test_loader)


def get_mean_and_std(train_data):
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


# print(get_mean_and_std(dataset))
error_l = []


def pred(test_iter, modelpath, num_gpus=1):
    ckpts = [os.path.join(modelpath, f) for f in sorted(os.listdir(modelpath)) if 'tar' in f]
    if len(ckpts) == 0:
        raise Exception('No previous model found, please check it')
    else:
        print('Found ckpts', ckpts)
        ckpt_path = ckpts[-1]
        print('Reload from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        model = MyVit()
        devices = [try_gpu(i) for i in range(num_gpus)]
        model = nn.DataParallel(model, device_ids=devices).to(devices[0])
        model.load_state_dict(ckpt['model'])
        # model.cuda(1)
        model.eval()
        out = []
        for x, y in test_iter:
            test_data = x
            test_labels = y[:, :4].numpy()
            test_pos = y[:, 4:]
            preds = model(test_data.cuda(1), test_pos.cuda(1)).cpu().detach().numpy()
            for i in range(test_labels.shape[0]):
                error = dis2me(preds[i, 1:], test_labels[i, 1:])
                error_l.append(error)
            out = np.append(out, preds)
        print('Median error', np.median(error_l))
        print('Average error', np.average(error_l))
        df_ = pd.DataFrame(out.reshape((-1, 4)), columns=['s', 'x', 'y', 'z'])
        df_['Location error'] = error_l
        df_.to_csv('submission_vit.csv', index=False)


# # Input: expects 3xN matrix of points
# # Returns R,t
# # R = 3x3 rotation matrix
# # t = 3x1 column vector
def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def p():
    pred(test_iter, modelpath)
    preds = pd.read_csv(abs_path + 'submission_vit.csv')
    labels = pd.read_csv(abs_path + 'test_data-s02.csv', names=['id', 's', '# tag_x', ' tag_y', ' tag_z', 'p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'])
    x, y, z = preds['x'].tolist(), preds['y'].tolist(), preds['z'].tolist()
    x1, y1, z1 = labels['# tag_x'].tolist(), labels[' tag_y'].tolist(), labels[' tag_z'].tolist()
    points_preds = np.array([x, y, z]).T
    points_labels = np.array([x1, y1, z1]).T
    if points_preds.shape[0] % 2 != 0:
        points_preds = np.delete(points_preds, -1, axis=0)
        points_labels = np.delete(points_labels, -1, axis=0)
    diff_features = points_labels[0::2, :] - points_labels[1::2, :]
    diff_labels = points_preds[0::2, :] - points_preds[1::2, :]
    diff_features_dist = np.linalg.norm(diff_features, axis=1)
    diff_labels_dist = np.linalg.norm(diff_labels, axis=1)
    l_abs = np.linalg.norm(points_labels - points_preds, axis=1)
    l = abs(diff_features_dist - diff_labels_dist)
    li = l.tolist()
    # print(li)
    # print(l_abs.tolist())
    print(np.median(li))
    # print(np.median(l_abs))
    error_list = []
    R, t = rigid_transform_3D(points_preds.T, points_labels.T)
    points_preds = (R @ points_preds.T) + t
    points_preds = points_preds.T
    for i in range(points_preds.shape[0]):
        error = dis2me(points_preds[i], points_labels[i])
        error_list.append(error)
    print('Median Location Error', np.median(error_list))
    # print(error_list)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


train(net, beta, device, train_iter, optimizer, EPOCHS, modelpath)
# p()





