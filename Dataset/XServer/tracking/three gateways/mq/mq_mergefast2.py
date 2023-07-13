# -*- coding: utf-8 -*-
"""合并多个gateway和optitrack的消息队列
"""
import datetime
import json
import os
import queue
import sys
import threading
import time
import traceback

sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import numpy as np
import pika
from utils.param_loader import ParamLoader as PM


class MsgWorker(threading.Thread):
    """Threading for getting message from Gateways Message Queue
    """
    def __init__(self, hostname, port, rb_queuename, mg_queue):
        super().__init__()
        self.credentials = pika.PlainCredentials(username='admin', password='admin')
        self.parameters = pika.ConnectionParameters(host=hostname, port=port, credentials=self.credentials, heartbeat=0)
        self.connection = pika.BlockingConnection(parameters=self.parameters)
        self.channel = self.connection.channel()

        self.channel.exchange_declare(exchange='direct_gateway', exchange_type='direct')
        self.channel.queue_declare(queue=rb_queuename, exclusive=True)
        self.channel.queue_bind(exchange='direct_gateway', queue=rb_queuename, routing_key=rb_queuename)

        self.mg_queue = mg_queue
        self.rb_queue = rb_queuename


    def run(self):
        num = 0

        while True:
            # blocking consume
            for _, _, body in self.channel.consume(self.rb_queue, auto_ack=True, inactivity_timeout=10):
                if body is not None:
                    num+=1
                    dataRecv = json.loads(body)
                    try:
                        self.mg_queue.put(dataRecv,timeout=10)
                    except queue.Full:
                        self.channel.cancel()
                        self.channel.queue_purge(self.rb_queue)
                        self.connection.process_data_events()
                else:
                    self.connection.process_data_events()            # heartbeat
                if num % 10 == 0:
                    self.channel.cancel()
                    self.channel.queue_purge(self.rb_queue)
                    self.connection.process_data_events()



def merge_data(hostname, port):
    """align timestamp of Message Queue, merge messages
    """
    rbqueue_names = PM.gateway_names.copy()
    if PM.optitrack:
        rbqueue_names.append('optitrack')
    # print(rbqueue_names)

    bfqueues = [queue.Queue(10) for _ in rbqueue_names]
    if 'optitrack' in rbqueue_names:
        HAVE_GT = True
    else:
        HAVE_GT = False
        opti_default = {'gateway1':[[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                        'gateway2':[[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                        'gateway3':[[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                        'target':[0,0,0]}

    # Create the Threading
    workers = [MsgWorker(hostname, port, rbqueue_name, bfqueue) for (rbqueue_name, bfqueue)
               in zip(rbqueue_names, bfqueues)]

    # Start the threading
    for worker in workers:
        worker.daemon = True
        worker.start()

    # gateway data dict, optitrack data dict
    gwdata_pop, opdata_pop = {}, {}

    while True:
        # if not gateway data in the buffer, get the data
        # print('-------------new round---------------------')
        if HAVE_GT:
            for rbqueue_name, bfqueue in zip(rbqueue_names[:-1], bfqueues[:-1]):
                if not rbqueue_name in gwdata_pop:
                    if not bfqueue.empty():
                        gwdata_pop[rbqueue_name] = bfqueue.get()
        else:
            for rbqueue_name, bfqueue in zip(rbqueue_names[:], bfqueues[:]):
                if not rbqueue_name in gwdata_pop:
                    if not bfqueue.empty():
                        gwdata_pop[rbqueue_name] = bfqueue.get()

        # get the gateway time from buffer
        usrptimes = np.array([],float)
        data_keys = np.array([],str)
        for queue_id, message in gwdata_pop.items():
            if isinstance(message, dict):
                usrptime = float(list(message.keys())[0])
                usrptimes = np.append(usrptimes, usrptime)
                data_keys = np.append(data_keys, queue_id)

        #找到最早的时间
        if usrptimes.size != 0:
            # print('get usrp time:', usrptimes, 'gateways', data_keys)
            usrptimes = np.around(usrptimes,2)                        # %.2f
            min_ind = np.where(usrptimes==min(usrptimes))[0]
            confidence = min_ind.size
            min_key = data_keys[min_ind]
            # print('merge timestamp!!!', usrptimes[min_ind])
            # print('merge gateway name', min_key)
            data_upload = {}
            for key in min_key:
                (_,data), = gwdata_pop[key].items()
                logTime = data['createdTime']
                data_upload.update({key:data})

            # do not have optitrack stream
            if not HAVE_GT:
                data_upload.update({'optitrack': opti_default})
                for key in min_key:
                    gwdata_pop.pop(key)

            # have optitrack stream
            else:
                #从optitrack的队列中找到groundtruth
                min_usrptime = np.around(min(usrptimes),1)           # .1f
                opti_queue = bfqueues[-1]

                if not opdata_pop:                #空字典
                    if not opti_queue.empty():    #非空队列
                        opdata_pop = opti_queue.get()
                        (opti_time,opti_data), = opdata_pop.items()
                        opti_time = np.around(float(opti_time), 1)
                        # print('get new optitrack time:', opti_time)

                        while opti_time <= min_usrptime:
                            if opti_time == min_usrptime:
                                data_upload.update({'optitrack':opti_data})
                                for key in min_key:
                                    gwdata_pop.pop(key)
                                break
                            if not opti_queue.empty():
                                opdata_pop = opti_queue.get()
                                (opti_time,opti_data), = opdata_pop.items()
                                opti_time = np.around(float(opti_time), 1)
                                # print('throw old time and get new optitrack time', opti_time)
                            else:
                                break
                        if opti_time > min_usrptime:
                            for key in min_key:
                                gwdata_pop.pop(key)

                else:   #optitrack 字典有数据
                    (opti_time,opti_data), = opdata_pop.items()
                    opti_time = np.around(float(opti_time), 1)
                    # print('last optitrack time not consume:', opti_time)

                    while opti_time <= min_usrptime:
                        if opti_time == min_usrptime:
                            data_upload.update({'optitrack':opti_data})
                            for key in min_key:
                                gwdata_pop.pop(key)
                            # opdata_pop.clear()
                            break
                        if not opti_queue.empty():
                            opdata_pop = opti_queue.get()
                            (opti_time,opti_data), = opdata_pop.items()
                            opti_time = np.around(float(opti_time), 1)
                            # print('throw old time and get new optitrack time', opti_time)
                        else:
                            break
                    if opti_time > min_usrptime:
                        for key in min_key:
                            gwdata_pop.pop(key)

            if 'optitrack' in data_upload:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')
                data_upload.update({'logTime':logTime, 'phyTime':timestamp})
                yield data_upload, confidence
        time.sleep(0.01)


if __name__ == "__main__":
    for i,j in merge_data('158.132.255.178', '5672'):
        # print("-------------confidence---------------:", j)

        time.sleep(1)
