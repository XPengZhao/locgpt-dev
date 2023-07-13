# -*-coding: utf-8 -*-
"""code for real time tracking
"""

import datetime
import json
import os
import sys

import matplotlib.image as plm
import numpy as np

from dnn.dnn_tracking import DNN_tracking
from loc.kalfilter import Kalman2D
from loc.tagoram import HologramAoA
from mq.mq_mergefast import ClearableStack, merge_data
from mq.mq_publish import MQPublish, NpEncoder
from utils.ft import Track_filter
from utils.param_loader import ParamLoader as PM
import threading

if __name__ == '__main__':

    mc = MQPublish(PM.ip, PM.port)
    try:
        mc.connect()
    except Exception:
        raise Exception("can't connect to rabbit")

    dnn_worker2, dnn_worker12 = DNN_tracking('model2.pkl'), DNN_tracking('model12.pkl')
    kalman_workers = kalman_workers = {'gateway1':[Kalman2D() for _ in range(16)], 'gateway2':[Kalman2D() for _ in range(16)]}
    trft = Track_filter()
    KL = True
    loc_kl = Kalman2D()
    loc_kl.ch_param(0.01, 0.02)

    gateway_names = PM.gateway_names
    atnoffsets = {'gateway1':PM.ant1offsets, 'gateway2':PM.ant2offsets}
    lifoqueue1 = ClearableStack(-1)
    sub_threading = threading.Thread(target=merge_data, args=(PM.ip, PM.port,lifoqueue1,))
    sub_threading.start()


    # initialize HologramAoA
    hologramAoA_workers = {gateway_name:HologramAoA() for gateway_name in gateway_names}

    print('Running in merge_data generator...')
    # for datas, confidence in merge_datas:
    while True:
        if not lifoqueue1.empty():
            lastest_data = lifoqueue1.get()
            lifoqueue1.clear()
            (confidence, datas), = lastest_data.items()
            spectrum = {'algorithm': PM.algorithm, 'confidence':confidence, 'xRange':PM.xRange,
                        'yRange':PM.yRange, 'zRange':PM.zRange}
            xServer = {'frequency':PM.freq}
            data2gui = {'logTime':datas.pop('logTime'), 'phyTime':datas.pop('phyTime')}
            gateway_coordict, gateways, psp_dict, singal_dict = {}, {}, {}, {}

            # get pos label
            opti_data = datas.pop('optitrack')
            for data_id, pos_data in opti_data.items():
                if data_id in gateway_names:
                    pos_data = np.array(pos_data).T
                    gateway_coordict.update({data_id:pos_data})
                if data_id == 'target':
                    data2gui.update({"truth":pos_data})
                    truth = pos_data
                    target_pos = np.array(pos_data)
                    pos_gt = np.array(pos_data).flatten()

            # get heatmap
            for gateway_id, data in datas.items():
                data2gui.update({"tagId":data.pop('tagid')})
                atn_phase = np.array(data['phase'])
                atn_phase = -atn_phase - atnoffsets[gateway_id]
                data.update({'phase':np.around(atn_phase,4), "phaseOffset":np.around(atnoffsets[gateway_id],4),
                            'position':[[0,0,0],[0,0,0],[0,0,0],[0,0,0]], "aoa":{"azimuth":0,"elevation":0}})
                gateways.update({gateway_id:data})

                kalman_worker = kalman_workers[gateway_id]       # list
                atn_iq = np.exp(1j * np.array(atn_phase))
                atn_phase = [kalman_worker[i].kalmanPredict(np.array([atn_iq[i].real, atn_iq[i].imag], np.float32))for i in range(16)]
                atn_phase = np.array(atn_phase, np.float64)
                atn_real = atn_phase[:,0,0]
                atn_imag = atn_phase[:,1,0]
                atn_phase = np.angle(atn_real + 1j*atn_imag)

                dataoffset = atn_phase.reshape(1, -1)
                dataoffset = dataoffset[:, PM.atnseq].transpose()
                psp = hologramAoA_workers[gateway_id].get_aoa_heatmap(dataoffset)
                psp_dict.update({gateway_id:psp})

            xServer.update({'gateways':gateways})

            if confidence == 2:
                print('confidence = 2')
                psp_dict['gateway3'] = np.zeros((90, 360))

                psp_all = np.concatenate((psp_dict['gateway1'].reshape(90,360,1),psp_dict['gateway2'].reshape(90,360,1),psp_dict['gateway3'].reshape(90,360,1)),axis=2)

                figure1, figure2 = psp_all.copy(), psp_all.copy()
                figure1[:,:,1], figure2[:,:,0] = 0, 0
                loc12 = dnn_worker12.get_location(psp_all)
                loc2 = dnn_worker2.get_location(figure2)

                dis12_2 = DNN_tracking.error_cal(loc12, loc2)
                d_2points = dis12_2
                print('distance gap of two gateways:', d_2points)
                if d_2points < 0.3:
                    loc = (loc12 + loc2) / 2
                else:
                    print('the distance between two point is too large')
                    continue

            elif confidence == 1:
                print('confidence = 1')
                continue
                g_names = ['gateway1', 'gateway2', 'gateway3']
                for gateway in g_names:
                    if gateway not in psp_dict:
                        psp_dict[gateway] = np.zeros((90, 360))
                psp_all = np.concatenate((psp_dict['gateway1'].reshape(90,360,1),psp_dict['gateway2'].reshape(90,360,1),psp_dict['gateway3'].reshape(90,360,1)),axis=2)

                if gateway_id == 'gateway1':
                    loc = dnn_worker1.get_location(psp_all)
                elif gateway_id == 'gateway2':
                    loc = dnn_worker2.get_location(psp_all)

            position = np.around(loc, 2)
            position = trft.med_filter(position)

            if position.shape[0] != 0:
                xy = [position[0], position[2]]
                xykl = loc_kl.kalmanPredict(np.array(xy, np.float32))
                if KL:
                    position[0] = xykl[0,0]
                    position[2] = xykl[1,0]
                print("location of tag:", loc, "\t error:", DNN_tracking.error_cal(position, pos_gt))
                data2gui.update({"position":position, "xServer":xServer})
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')
                spectrum.update({"data":[[0]], "createdTime":timestamp})
                data2gui.update({"spectrum":spectrum})
                data_upload = json.dumps(data2gui, cls=NpEncoder)
                mc.sendData(data_upload)
