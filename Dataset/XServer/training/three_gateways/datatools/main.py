# -*-coding: utf-8 -*-
"""dataset tools confidence=2
"""

import os

import matplotlib.image as plm
import numpy as np

from loc.kalfilter import Kalman1D
from loc.tagoram import HologramAoA
from mq.mq_mergefast import merge_data
from utils.param_loader import ParamLoader as PM

if __name__ == '__main__':

    if not os.path.exists('./dataset2_1/heatmap'):
        os.makedirs('./dataset2_1/heatmap')
        os.makedirs('./dataset2_1/poslabel')
        os.makedirs('./dataset2_2/heatmap')
        os.makedirs('./dataset2_2/poslabel')

    kalman_workers = {'gateway1':[Kalman1D() for _ in range(16)], 'gateway2':[Kalman1D() for _ in range(16)]}


    gateway_names = PM.gateway_names
    atnoffsets = {'gateway1':PM.ant1offsets, 'gateway2':PM.ant2offsets}

    merge_datas = merge_data(PM.ip, PM.port)


    # initialize HologramAoA
    hologramAoA_workers = {gateway_name:HologramAoA() for gateway_name in gateway_names}

    num_all, num_2= 0, 1
    print('Running in merge_data generator...')
    for datas, confidence in merge_datas:
        _,_ = datas.pop('logTime'),datas.pop('phyTime')
        num_all += 1
        print(num_all)

        # if confidence != 1:
            # continue
        gateways,psp_dict = {},{}

        # get pos label
        opti_data = datas.pop('optitrack')
        for data_id, pos_data in opti_data.items():
            if data_id == 'target':
                target_pos = np.array(pos_data).reshape(1,3)

        if confidence == 2:
            np.savetxt("dataset2_1/poslabel/%05d.txt"%num_2, target_pos, delimiter=',',fmt='%.04f')
            np.savetxt("dataset2_2/poslabel/%05d.txt"%num_2, target_pos, delimiter=',',fmt='%.04f')

            # get heatmap
            for gateway_id, data in datas.items():
                data.pop('tagid')
                atn_phase = np.array(data['phase'])
                atn_phase = -atn_phase - atnoffsets[gateway_id]
                kalman_worker = kalman_workers[gateway_id]       # list
                atn_phase = [kalman_worker[i].kalmanPredict(np.array(atn_phase[i], np.float32))[0,0] for i in range(16)]
                atn_phase = np.array(atn_phase, np.float64)

                dataoffset = atn_phase.reshape(1, -1)
                dataoffset = dataoffset[:, PM.atnseq].transpose()
                psp = hologramAoA_workers[gateway_id].get_aoa_heatmap(dataoffset)
                psp_dict.update({gateway_id:psp})


            psp_dict['gateway3'] = np.zeros((90, 360))

            psp_all = np.concatenate((psp_dict['gateway1'].reshape(90,360,1),psp_dict['gateway2'].reshape(90,360,1),psp_dict['gateway3'].reshape(90,360,1)),axis=2)

            figure1, figure2 = psp_all.copy(), psp_all.copy()
            figure1[:,:,1], figure2[:,:,0] = 0, 0
            plm.imsave("dataset2_1/heatmap/%05d.jpg"%num_2, figure1)
            plm.imsave("dataset2_2/heatmap/%05d.jpg"%num_2, figure2)
            num_2 += 1
