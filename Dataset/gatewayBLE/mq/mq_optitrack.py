# -*- coding: utf-8 -*-
"""合并多个gateway和optitrack的消息队列
"""
import os
import sys

sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import datetime
import json
import queue
import threading
import time
import copy

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
        self.channel.cancel()
        self.channel.queue_purge(self.rb_queue)
        self.connection.process_data_events()

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
                if num % 10 == 0 and PM.realtime:
                    self.channel.cancel()
                    self.channel.queue_purge(self.rb_queue)
                    self.connection.process_data_events()



def merge_data(hostname, port):
    """align timestamp of Message Queue, merge messages
    """
    rbqueue_names = ['gateway1','gateway2', 'optitrack']

    bfqueues       = [queue.Queue(-1) for _ in rbqueue_names]

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
        print('-------------new round---------------------')
        # if not gateway data in the buffer, get the data
        for rbqueue_name, bfqueue in zip(rbqueue_names[:-1], bfqueues[:-1]):
            if not rbqueue_name in gwdata_pop:
                if not bfqueue.empty():
                    gwdata_pop[rbqueue_name] = bfqueue.get()

        usrptimes = np.array([],float)
        data_keys = np.array([],str)
        for queue_id, message in gwdata_pop.items():
            if isinstance(message, dict):
                usrptime = float(list(message.keys())[0])
                usrptimes = np.append(usrptimes, usrptime)
                data_keys = np.append(data_keys, queue_id)

        if usrptimes.size != 0:
            for i,data_key in enumerate(data_keys):
                print(data_key, ' time: ', usrptimes[i])
                gwdata_pop.pop(data_key)

        #从optitrack的队列中找到groundtruth
        opti_queue = bfqueues[-1]

        if not opdata_pop:                #空字典
            if not opti_queue.empty():    #非空队列
                opdata_pop = opti_queue.get()
                (opti_time,opti_data), = opdata_pop.items()
                opti_time = np.around(float(opti_time), 1)
                print('get new optitrack time:', opti_time)

        else:   #optitrack 字典有数据
            if not opti_queue.empty():
                opdata_pop = opti_queue.get()
                (opti_time,opti_data), = opdata_pop.items()
                opti_time = np.around(float(opti_time), 1)
                print('throw old time and get new optitrack time', opti_time)
        time.sleep(0.05)



if __name__ == "__main__":
    for i,j in merge_data('158.132.255.178', '5672'):
        print("-------------confidence---------------:", j)
        time.sleep(1)
