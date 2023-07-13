# -*- coding: utf-8 -*-
import json
import os
import sys
import time

sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import numpy as np
from pymongo import MongoClient
from mq.mq_publish import MQPublish_gateway

last_createdtime1 = ""
time_num1 = 0
last_createdtime2 = ""
time_num2 = 0
last_createdtime3 = ""
time_num3 = 0

def get_db():
    """Connect to MongoDB"""
    try:
        client = MongoClient('mongodb://tagsys:tagsys@158.132.255.118:27017/')
        db = client.LRT  # tabledatas
    except:
        raise Exception("can't connect to mongodb")
    return db


def inite_mc(name):
    """Initial Rabbitmq connection"""
    mc = MQPublish_gateway('158.132.255.118', '5672', name)
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
    """Find all gateway positions (Three gateways)"""
    gateway_position = {}
    for each in data:
        gateways = each['xServer'][0]['gateways']
        for name in gateways:
            if name not in gateway_position:
                gateway = gateways[name]
                gateway_position[name] = np.transpose(gateway['position']).tolist()
            if len(gateway_position) == 3:
                print(gateway_position)
                return gateway_position


def convert_data(origin_data, gateway_position):
    gateways = origin_data['xServer'][0]['gateways']
    data_upload = {}
    global time_num1,time_num2,time_num3
    global  last_createdtime1,last_createdtime2,last_createdtime3


    # gateways
    tagid = origin_data['tagId']
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
            if created_time == last_createdtime1:
                time_num1 += 1
            else:
                time_num1 = 0
            last_createdtime1 = created_time
            time_today = np.around(np.around(float(time_today),1) + 0.01*time_num1, 2)
        elif name == 'gateway2':
            if created_time == last_createdtime2:
                time_num2 += 1
            else:
                time_num2 = 0
            last_createdtime2 = created_time
            time_today = np.around(np.around(float(time_today),1) + 0.01*time_num2, 2)
        elif name == 'gateway3':
            if created_time == last_createdtime3:
                time_num3 += 1
            else:
                time_num3 = 0
            last_createdtime3 = created_time
            time_today = np.around(np.around(float(time_today),1) + 0.01*time_num3, 2)

        # print(name, time_today)


        data_upload[name] = json.dumps({time_today: gateway_data})
        # print(name, ':', time_today)

    opti_timetoday = get_time_today(gateway_data['createdTime'])
    opti_timetoday = np.around(opti_timetoday, 1)
    # optitrack
    optitrack_data = gateway_position.copy()
    optitrack_data['target'] = origin_data['truth']
    data_upload['optitrack'] = json.dumps({opti_timetoday: optitrack_data})

    return data_upload


def send_data(data):
    gateway_position = find_gateway(data)
    gateways = ['gateway1', 'gateway2', 'gateway3']
    mc_list = {gateway: inite_mc(gateway) for gateway in gateways}
    mc_list['optitrack'] = inite_mc('optitrack')

    num = 0
    for each in data:
        num += 1
        data_upload = convert_data(each, gateway_position)
        for i in data_upload:
            mc_list[i].sendData(data_upload[i])
        # print('--------------------')
        # print(num)
        # print(data_upload)
        time.sleep(0.0001)
        # break
    print(num)


def main():
    db = get_db()
    collist = db.list_collection_names()
    for i, element in enumerate(collist):
        print('[%d]:'%i, element)

    dataind = [10,13,6,15,24,0,4,20,5,12]

    col = db[collist[14]]                  # tabledatas
    data = col.find({}, {'_id': 0})
    print(data[0])

    send_data(data)


if __name__ == '__main__':
    main()
