# -*- coding: utf-8 -*-
"""generate dataset12
"""
import os
import sys

from tqdm import tqdm

sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import matplotlib.image as plm
import matplotlib.pyplot as plt
import numpy as np
from pymongo import MongoClient

from loc.kalfilter import Kalman2D
# from loc.smusic import SmoothMusic
from loc.tagoram import HologramAoA
from utils.param_loader import ParamLoader as PM


def get_db():
    """Connect to MongoDB"""
    try:
        client = MongoClient('mongodb://tagsys:tagsys@158.132.255.118:27017/')
        # client = MongoClient('mongodb://tagsys:tagsys@158.132.255.205:27017/')
        db = client.LRT  # tabledatas
    except:
        raise Exception("can't connect to mongodb")
    return db


if __name__ == '__main__':

    db = get_db()
    collist = db.list_collection_names()
    collist.sort()
    for i, element in enumerate(collist):
        print('[%d]:'%i, element)
    data_list = [36]

    if not os.path.exists('./dataset12/train/dataset/heatmap'):
        # os.makedirs('./dataset2_1/heatmap')
        # os.makedirs('./dataset2_1/poslabel')
        # os.makedirs('./dataset2_2/heatmap')
        # os.makedirs('./dataset2_2/poslabel')
        os.makedirs('./dataset12/train/dataset/heatmap')
        os.makedirs('./dataset12/train/dataset/poslabel')

    kalman_workers = {'gateway1':[Kalman2D() for _ in range(16)], 'gateway2':[Kalman2D() for _ in range(16)]}
    gateway_names = PM.gateway_names
    atnoffsets = {'gateway1':PM.ant1offsets, 'gateway2':PM.ant2offsets}
    # musicAoA_workers = {gateway_name:SmoothMusic(0.16,4,3,920e6) for gateway_name in gateway_names}
    hologramAoA_workers = {gateway_name:HologramAoA() for gateway_name in gateway_names}

    phase1_before = np.array([])
    phase1_after = np.array([])

    num_all, num_2= 0, 1
    last_pos = [10,10,10]
    for ii in data_list:
        col = db[collist[ii]]                  # tabledatas
        data = col.find({}, {'_id': 0}).sort([("phyTime",1)]).allow_disk_use(True)
        datalen = data.count()
        for each in tqdm(data, total=datalen):
            num_all += 1

            gateways = each['xServer'][0]['gateways']
            target_pos = each['truth']
            dis = np.linalg.norm((np.array(target_pos)-np.array(last_pos)))
            # if dis < 0.005:
            #     continue
            last_pos = target_pos
            target_pos = np.array(target_pos).reshape(1,3)
            psp_dict = {}
            if len(gateways)==2 and 'gateway3' not in gateways:

                # np.savetxt("dataset2_1/poslabel/%05d.txt"%num_2, target_pos, delimiter=',',fmt='%.04f')
                # np.savetxt("dataset2_2/poslabel/%05d.txt"%num_2, target_pos, delimiter=',',fmt='%.04f')
                np.savetxt("./dataset12/train/dataset/poslabel/%05d.txt"%num_2, target_pos, delimiter=',',fmt='%.04f')

                for name in gateways:
                    gateway = gateways[name]
                    atn_phase = gateway['phase']

                    # if name == 'gateway1':
                    #     phase1_before = np.append(phase1_before, atn_phase[0])

                    kalman_worker = kalman_workers[name]       # list
                    atn_iq = np.exp(1j * np.array(atn_phase))
                    atn_phase = [kalman_worker[i].kalmanPredict(np.array([atn_iq[i].real, atn_iq[i].imag], np.float32))for i in range(16)]
                    atn_phase = np.array(atn_phase, np.float64)
                    atn_real = atn_phase[:,0,0]
                    atn_imag = atn_phase[:,1,0]
                    atn_phase = np.angle(atn_real + 1j*atn_imag)

                    # if name == 'gateway1':
                    #     phase1_after = np.append(phase1_after, atn_phase[0])

                    dataoffset = atn_phase.reshape(1, -1)
                    dataoffset = dataoffset[:, PM.atnseq_ours].transpose()
                    # data_signal= np.exp(1j * dataoffset)
                    # psp = musicAoA_workers[name].getSMusicResult(data_signal)
                    psp = hologramAoA_workers[name].get_aoa_heatmap(dataoffset)
                    # psp = ((psp - np.min(psp)) / (np.max(psp) - np.min(psp))).transpose()

                    psp_dict.update({name:psp})
                psp_dict['gateway3'] = np.zeros((90, 360))

                psp_all = np.concatenate((psp_dict['gateway1'].reshape(90,360,1),psp_dict['gateway2'].reshape(90,360,1),psp_dict['gateway3'].reshape(90,360,1)),axis=2)

                figure1, figure2 = psp_all.copy(), psp_all.copy()
                figure1[:,:,1], figure2[:,:,0] = 0, 0
                # plm.imsave("dataset2_1/heatmap/%05d.jpg"%num_2, figure1)
                # plm.imsave("dataset2_2/heatmap/%05d.jpg"%num_2, figure2)
                plm.imsave("./dataset12/train/dataset/heatmap/%05d.jpg"%num_2, psp_all)
                num_2 += 1

                # if num_2 % 300 == 0:
                #     plt.plot(phase1_after)
                #     plt.plot(phase1_before)
                #     plt.show()