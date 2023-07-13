# -*- coding: utf-8 -*-
import json
import os
import sys
import time

sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import numpy as np
from pymongo import MongoClient
from mq.mq_publish import MQPublish_gateway
from utils.param_loader import ParamLoader as PM

last_createdtime1 = ""
time_num1 = 0
last_createdtime2 = ""
time_num2 = 0
last_createdtime3 = ""
time_num3 = 0

def get_db():
    """Connect to MongoDB"""
    try:
        client = MongoClient('mongodb://tagsys:tagsys@127.0.0.1:27017/')
        # client = MongoClient('mongodb://tagsys:tagsys@158.132.255.205:27017/')
        db = client.LRT  # tabledatas
    except:
        raise Exception("can't connect to mongodb")
    return db


def inite_mc(name):
    """Initial Rabbitmq connection"""
    mc = MQPublish_gateway(PM.ip, PM.port, name)
    try:
        mc.connect()
    except:
        raise Exception("can't connect to localhost")
    return mc


def get_time_today(timestamp):
    """Get Timestamp (The number of seconds counted from zero of the day) """
    today_time = timestamp.split(' ')[1]
    h, m, s, us = today_time.split(':')
    today_s = float(h) * 3600 + float(m) * 60 + float(s) + float(us) / 1000000
    return round(today_s, 1)


def find_gateway(data):
    """Find all gateway positions (Two gateways)"""
    gateway_position = {}
    for each in data:
        gateways = each['xServer'][0]['gateways']
        for name in gateways:
            if name not in gateway_position:
                gateway = gateways[name]
                gateway_position[name] = np.transpose(gateway['position']).tolist()
            if 'gateway1' in gateway_position and 'gateway2' in gateway_position:
                print(gateway_position)
                return gateway_position


def convert_data(origin_data, gateway_position):
    gateways = origin_data['xServer'][0]['gateways']
    data_upload = {}
    global time_num1,time_num2,time_num3
    global  last_createdtime1,last_createdtime2,last_createdtime3

    print('-------------------------------------')
    opti_time = 99998
    # gateways
    tagid = origin_data['tagId']
    flag = 0

    if len(gateways) == 2 and 'gateway3' not in gateways:
        t1 = gateways['gateway1']['createdTime']
        t2 = gateways['gateway2']['createdTime']
        if t1<t2:
            flag = 2
            print('t1%f<t2%f'%(t1, t2))
        elif t2<t1:
            flag = 1
            print('t2<t1')


    for name in gateways:
        gateway = gateways[name]
        gateway_data = {}
        gateway_data['version'] = gateway['version']
        gateway_data['tagid'] = tagid
        phase_offset = np.array(gateway['phaseOffset'])
        gateway_data['phase'] = (-np.array(gateway['phase']) - phase_offset).tolist()
        gateway_data['rss'] = gateway['rss']
        gateway_data['createdTime'] = gateway['createdTime']
        gateway_data['sourcefile'] = gateway['sourcefile']
        created_time = gateway_data['createdTime']
        time_today = get_time_today(gateway_data['createdTime'])

        if name == 'gateway1':
            if flag != 1:
                if created_time == last_createdtime1:
                    time_num1 += 1
                else:
                    time_num1 = 0
                last_createdtime1 = created_time
                time_today = np.around(np.around(float(time_today),1) + 0.01*time_num1, 2)
            else:
                time_today = np.around(float(time_today),1)
            print('gateway1', time_today)
            if opti_time > time_today:
                opti_time = time_today

        elif name == 'gateway2':
            if flag != 2:
                if created_time == last_createdtime2:
                    time_num2 += 1
                else:
                    time_num2 = 0
                last_createdtime2 = created_time
                time_today = np.around(np.around(float(time_today),1) + 0.01*time_num2, 2)
            else:
                time_today = np.around(float(time_today),1)
            print('gateway2', time_today)
            if opti_time > time_today:
                opti_time = time_today
        elif name == 'gateway3':
            if created_time == last_createdtime3:
                time_num3 += 1
            else:
                time_num3 = 0
            last_createdtime3 = created_time
            time_today = np.around(np.around(float(time_today),1) + 0.01*time_num3, 2)
            opti_time3 = time_today

        data_upload[name] = json.dumps({time_today: gateway_data})
        # print(name, ':', time_today)

    opti_timetoday = np.around(opti_time, 1)
    print('optitrack', opti_timetoday)
    # optitrack
    optitrack_data = gateway_position.copy()
    optitrack_data['target'] = origin_data['truth']
    data_upload['optitrack'] = json.dumps({opti_timetoday: optitrack_data})

    return data_upload


def send_data(data):
    gateway_position = find_gateway(data)
    gateways = ['gateway1', 'gateway2']
    mc_list = {gateway: inite_mc(gateway) for gateway in gateways}
    mc_list['optitrack'] = inite_mc('optitrack')

    num = 0
    for each in data:
        num += 1
        data_upload = convert_data(each, gateway_position)
        for i in data_upload:
            if i != 'gateway3':
                mc_list[i].sendData(data_upload[i])
        # print('--------------------')
        #print(num)
        # print(data_upload)
        time.sleep(0.0001)
        # break
    # print(num)


def main():
    db = get_db()
    collist = db.list_collection_names()
    for i, element in enumerate(collist):
        print('[%d]:'%i, element)

    data_list = [2]
    for ii in data_list:
        col = db[collist[ii]]                  # tabledatas
        data = col.find({}, {'_id': 0}).sort([("phyTime",1)]).allow_disk_use(True)
        print(type(data))

        send_data(data)


if __name__ == '__main__':
    main()
