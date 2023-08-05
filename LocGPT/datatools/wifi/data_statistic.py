# -*- coding: utf-8 -*-
"""
"""
import math

import h5py
import matplotlib.image as plm
import numpy as np
import torch



if __name__ == '__main__':

    scene = "July18"
    filepath = f'data/wifi/channels_{scene}.mat'

    with h5py.File(filepath, 'r') as f:
        print(list(f.keys()))
        channels_wo_offset = np.array(f['channels_wo_offset'])  #[ap, atn, freq, sample]
        channels = channels_wo_offset.transpose(3, 2, 1, 0)
        data_len = channels.shape[0]
        labels = np.array(f['labels']).transpose(1, 0)
        labels = torch.tensor(labels).float()
        atn_sep = np.array(f['opt']['ant_sep']).item()
        freq = np.array(f['opt']['freq']).flatten()
        # freq = np.median(freq)

        ap = f['ap']
        ap_num = len(ap)
        gateway_pos = np.zeros((ap_num, 2))
        for i, col in enumerate(ap):
            ap_pos = f[col[0]][:]
            ap_pos = np.mean(ap_pos, axis=-1)
            gateway_pos[i, :2] = ap_pos
        # print("gateway_pos: ", gateway_pos)


    labels_min_x = labels[:, 0].min()
    labels_max_x = labels[:, 0].max()
    labels_min_y = labels[:, 1].min()
    labels_max_y = labels[:, 1].max()
    labels_area = (labels_max_x - labels_min_x) * (labels_max_y - labels_min_y)
    labels_density = data_len / labels_area
    print(f"data len: {data_len}, labels_area: {labels_area:.1f}, labels_density: {labels_density:.0f}")

    labels_mean = labels.mean(axis=0)
    gateway_dis_mean = np.linalg.norm(labels_mean - gateway_pos, axis=-1).mean()
    print(f"gateway_dis_mean: {gateway_dis_mean}")

    # print(np.max(abs(channels['imag'])))
    IQ = channels['real'] + 1j * channels['imag']
    rss = np.median(np.abs(IQ), axis=(0,1,2,3))
    print(20*np.log10(rss) - 100)