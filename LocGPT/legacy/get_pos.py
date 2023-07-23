import scipy.io as scio
import torch
from einops import rearrange
train = torch.load("data/ble-exp2/train_data-pq504.t")
test = torch.load("data/ble-exp2/test_data-pq504.t")

print(train.shape)
pos_train = train[...,2:5]
pos_test = test[...,2:5]

pos_train = rearrange(pos_train, "b n p -> (b n) p")
pos_test = rearrange(pos_test, "b n p -> (b n) p")
print(pos_train.shape)

scio.savemat("pos.mat", {"pos_train":pos_train.numpy(), "pos_test":pos_test.numpy()})
