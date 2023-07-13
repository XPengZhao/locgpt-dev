# -*-coding: utf-8 -*-
"""xServer 合并gateway，optitrack数据流，采集数据代码
"""
import datetime
import json
import time

import numpy as np

from mq.mq_optitrack import merge_data
from mq.mq_publish import MQPublish, NpEncoder
from utils.param_loader import ParamLoader as PM

if __name__ == '__main__':

    mc = MQPublish(PM.ip, PM.port)
    try:
        mc.connect()
    except Exception:
        raise Exception("can't connect to rabbit")

    gateway_names = PM.gateway_names
    atnoffsets = {'gateway1':PM.ant1offsets, 'gateway2':PM.ant2offsets, 'gateway3':PM.ant3offsets}

    merge_datas = merge_data(PM.ip, PM.port)

    # get optitrack coordinate
    gateway_coors ={'gateway2': np.array([[-4.44, -4.438, -4.244], [0.975, 1.13, 0.972], [-0.99, -0.99, -1.415]]), 'gateway3': np.array([[4.301, 4.294, 4.311], [1.113, 1.27, 1.112], [1.401, 1.402, 1.867]]), 'gateway1': np.array([[3.337, 3.332, 3.531], [1.02, 1.179, 1.023], [-1.529, -1.528, -1.109]])}

    num = 0
    print('Running in merge_data generator...')
    for datas, confidence in merge_datas:
        time.sleep(0.05)