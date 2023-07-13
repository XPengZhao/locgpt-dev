# -*-coding: utf-8 -*-
"""xServer 合并gateway，optitrack数据流，采集数据代码
"""
import datetime
import json

import numpy as np

from mq.mq_mergefast import merge_data
from mq.mq_publish import MQPublish, NpEncoder
from utils.param_loader import ParamLoader as PM
from pymongo import MongoClient
import time

if __name__ == '__main__':

    time_now = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
    tablename = 't' + time_now
    # tablename = "test_data4"
    describe = "test_lab2"


    client = MongoClient('mongodb://tagsys:tagsys@158.132.255.118:27017/')
    db = client.LRT  # tabledatas
    tabledata = db['tabledatas']
    metadata = {'Describe': describe, 'img': "undefined", 'tableName': tablename}
    tabledata.insert(metadata)

    collection = db[tablename]

    mc = MQPublish(PM.ip, PM.port)
    try:
        mc.connect()
    except Exception:
        raise Exception("can't connect to rabbit")

    gateway_names = PM.gateway_names
    atnoffsets = {'gateway1':PM.ant1offsets, 'gateway2':PM.ant2offsets, 'gateway3':PM.ant3offsets}

    merge_datas = merge_data(PM.ip, PM.port)

    # get optitrack coordinate
    R = 4
    gateway_coors = {'gateway1':np.array([0,-R]), 'gateway2':np.array( [-R*np.cos(np.pi/6), R*np.sin(np.pi/6)] ),
                     'gateway3':np.array([R*np.cos(np.pi/6), R*np.sin(np.pi/6)])}

    # gateway_coors ={'gateway2': np.array([[-4.44, -4.438, -4.244], [0.975, 1.13, 0.972], [-0.99, -0.99, -1.415]]), 'gateway3': np.array([[4.301, 4.294, 4.311], [1.113, 1.27, 1.112], [1.401, 1.402, 1.867]]), 'gateway1': np.array([[3.337, 3.332, 3.531], [1.02, 1.179, 1.023], [-1.529, -1.528, -1.109]])}

    num = 0
    print('Running in merge_data generator...')
    for datas, confidence in merge_datas:

        timestart = time.time()
        spectrum = {'algorithm': PM.algorithm, 'confidence':confidence, 'xRange':PM.xRange,
                    'yRange':PM.yRange, 'zRange':PM.zRange}
        xServer = {'frequency':PM.freq}
        data2gui = {'logTime':datas.pop('logTime'), 'phyTime':datas.pop('phyTime')}
        gateway_coordict = {}
        gateways = {}
        psp_dict = {}
        singal_dict = {}

        # 获取gateway坐标信息
        opti_data = datas.pop('optitrack')
        for data_id, pos_data in opti_data.items():
            if data_id in gateway_names:
                pos_data = np.array(pos_data).T
                gateway_coordict.update({data_id:pos_data})
            if data_id == 'target':
                data2gui.update({"truth":pos_data})
                truth = pos_data
                target_pos = np.array(pos_data)

        for gateway_id, data in datas.items():

            data2gui.update({"tagId":data.pop('tagid')})
            atn_phase = np.array(data['phase'])
            atn_phase = -atn_phase - atnoffsets[gateway_id]
            data.update({'phase':np.around(atn_phase,4), "phaseOffset":np.around(atnoffsets[gateway_id],4),
                         'position':gateway_coors[gateway_id], "aoa":{"azimuth":0,"elevation":0}})
            gateways.update({gateway_id:data})

        # no meaning, just for GUI web not crash
        # if num%100 == 0:
        #     if 'gateway3' not in gateway_names:
        #         gateways.update({'gateway3':data})

        xServer.update({'gateways':gateways})

        position = np.around((np.array([truth[0], truth[1], truth[2]])), 2)
        print("truth",truth)


        data2gui.update({"position":position, "xServer":xServer})


        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')

        spectrum.update({"createdTime":timestamp})
        data2gui.update({"spectrum":spectrum})

        num += 1
        print(num)

        data_upload = json.dumps(data2gui, cls=NpEncoder)
        mc.sendData(data_upload)

        data_dict = json.loads(data_upload)
        timestamp2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')
        data_dict.update({"savedTime":timestamp2})
        data_dict["spectrum"] = [data_dict["spectrum"]]
        data_dict["xServer"] = [data_dict["xServer"]]
        collection.insert(data_dict)

        timeend = time.time()

        print('time consuming: %f' % (timeend - timestart))