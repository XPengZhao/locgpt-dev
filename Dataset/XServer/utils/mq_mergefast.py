# -*- coding: utf-8 -*-
"""合并多个gateway和optitrack的消息队列
"""
import os
import sys

import yaml

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
    def __init__(self, queue_rb, queue_bf, **kwargs):
        super().__init__()
        self.credentials = pika.PlainCredentials(username=kwargs['user'], password=kwargs['password'])
        self.parameters = pika.ConnectionParameters(host=kwargs['hostname'],port=kwargs["port"],
                                                    credentials=self.credentials, heartbeat=0)
        self.connection = pika.BlockingConnection(parameters=self.parameters)
        self.channel = self.connection.channel()

        self.channel.exchange_declare(exchange=kwargs['exchange'], exchange_type='direct')
        self.channel.queue_declare(queue=queue_rb, auto_delete=True)
        self.channel.queue_bind(exchange='direct_gateway', queue=queue_rb, routing_key=queue_rb)

        self.queue_bf = queue_bf
        self.queue_rb = queue_rb

        # start listening
        self.channel.queue_purge(self.queue_rb)
        self.channel.basic_consume(queue=self.queue_rb, auto_ack=True, on_message_callback=self.callback)

        print('Waiting for messages. To exit press CTRL+C')
        self.channel.start_consuming()


    def callback(self, ch, method, properties, body):
        body_str = body.decode()  # Decode the byte string
        data = json.loads(body_str)  # Parse the JSON string
        print("Received %r" % data)
        if len(self.queue_bf) > 10:
            self.queue_bf.pop(0)
            self.queue_bf.append(data)
        else:
            self.queue_bf.append(data)



class SyncData(object):

    def __init__(self, **kwargs) -> None:
        realtime = True
        queue_gateway = kwargs['queue_gateway']
        queue_lidar = kwargs['queue_lidar']
        queues_rb = queue_gateway + queue_lidar
        queues_bf = [[] for _ in queues_rb]
        self.queues_dict = dict(zip(queues_rb, queues_bf))

        # Create the Threading
        workers = [MsgWorker(queue_rb, queue_bf, **kwargs) for (queue_rb, queue_bf)
                in zip(queues_rb, queues_bf)]
        # Start the threading
        for worker in workers:
            worker.daemon = True
            worker.start()



    def sync_by_sequence(self, tag_name):
        """Sync the same tag's data by the same sequence
        """
        queues_name_subset = [queue_name for queue_name in self.queues_dict.keys() if tag_name in queue_name]
        merged_data = []

        while len(merged_data) < len(queues_name_subset):
            first_queue = self.queues_dict[queues_name_subset[0]]

            if len(first_queue) > 0:
                ref_sequence = first_queue[0]['sequence']

                ## find ref_sequence in other queues
                for queue_name in queues_name_subset[1:]:
                    queue = self.queues_dict[queue_name]
                    for item in queue:
                        if item['sequence'] == ref_sequence:
                            merged_data.append(item)
                            break
                    else:
                        break

            for queue_name in queues_name_subset:
                frame = self.queues_dict[queue_name].get()
                if frame is not None:
                    merged_data.append(frame)
        for




    def merge_data(self):
        """align timestamp of Message Queue, merge messages
        """
        merged_data = {}

        for queue_name in self.queues_dict.keys():

            # Get the gateway number and tagid from the queue name
            gateway_name = queue_name.split('_')[0]
            tag_name = queue_name.split('_')[1][-1]

            frame = self.queues_dict[queue_name].get()
            if frame is not None:
                timestamp = datetime.strptime(frame['timestamp'], "%Y-%m-%dT%H:%M:%S:%f")

                if gateway_name not in merged_data:
                    merged_data[gateway_name] = []

                # Sync the same tag's data by the same sequence
                for item in merged_data[gateway_name]:
                    if item['tagid'] == data['tagid'] and item['sequence'] == data['sequence']:
                        break
                else:
                    # Sync the different tags' data by timestamp <= 100ms
                    for item in merged_data[gateway_number]:
                        item_timestamp = datetime.strptime(item['timestamp'], "%Y-%m-%dT%H:%M:%S:%f")
                        if abs(item_timestamp - timestamp) <= timedelta(milliseconds=100):
                            merged_data[gateway_number].append(data)
                            break















        # # gateway data dict, optitrack data dict
        # gwdata_pop, opdata_pop = {}, {}

        # while True:
        #     # if not gateway data in the buffer, get the data
        #     print('-------------new round---------------------')
        #     for rbqueue_name, bfqueue in zip(rbqueue_names[:-1], bfqueues[:-1]):
        #         if not rbqueue_name in gwdata_pop:
        #             if not bfqueue.empty():
        #                 gwdata_pop[rbqueue_name] = bfqueue.get()
        #             else:
        #                 time.sleep(0.1)
        #                 if not bfqueue.empty():
        #                     gwdata_pop[rbqueue_name] = bfqueue.get()

        #     # get the gateway time from buffer
        #     usrptimes = np.array([],float)
        #     data_keys = np.array([],str)
        #     for queue_id, message in gwdata_pop.items():
        #         if isinstance(message, dict):
        #             usrptime = float(list(message.keys())[0])
        #             usrptimes = np.append(usrptimes, usrptime)
        #             data_keys = np.append(data_keys, queue_id)

        #     #找到最早的时间
        #     if usrptimes.size != 0:
        #         print('get usrp time:', usrptimes, 'gateways', data_keys)
        #         usrptimes = np.around(usrptimes,2)
        #         min_time = min(usrptimes)
        #         min_ind = np.where(usrptimes==min_time)[0]             # minimum usrp timestamp
        #         merge_ind = np.where(abs(usrptimes-min_time)<=0.1)[0]  # merge timestamp min_time +- 0.2
        #         confidence = merge_ind.size
        #         min_key = data_keys[min_ind]
        #         merge_key = data_keys[merge_ind]
        #         print('merge timestamp!!!', usrptimes[merge_ind])
        #         print('merge gateway name', merge_key)
        #         data_upload = {}
        #         for key in merge_key:
        #             (_,data), = gwdata_pop[key].items()
        #             logTime = data['createdTime']
        #             if key != min_key[0]:
        #                 data_upload.update({key:copy.deepcopy(data)})
        #             else:
        #                 data_upload.update({key:data})

        #         #从optitrack的队列中找到groundtruth
        #         min_usrptime = np.around(min(usrptimes),1)           # .1f
        #         opti_queue = bfqueues[-1]

        #         if not opdata_pop:                #空字典
        #             if not opti_queue.empty():    #非空队列
        #                 opdata_pop = opti_queue.get()
        #                 (opti_time,opti_data), = opdata_pop.items()
        #                 opti_time = np.around(float(opti_time), 1)
        #                 print('get new optitrack time:', opti_time)

        #                 while opti_time <= min_usrptime:
        #                     if opti_time == min_usrptime:
        #                         data_upload.update({'optitrack':opti_data})
        #                         for key in min_key:
        #                             gwdata_pop.pop(key)
        #                         break
        #                     if not opti_queue.empty():
        #                         opdata_pop = opti_queue.get()
        #                         (opti_time,opti_data), = opdata_pop.items()
        #                         opti_time = np.around(float(opti_time), 1)
        #                         print('throw old time and get new optitrack time', opti_time)
        #                     else:
        #                         break
        #                 if opti_time > min_usrptime:
        #                     for key in min_key:
        #                         gwdata_pop.pop(key)

        #         else:   #optitrack 字典有数据
        #             (opti_time,opti_data), = opdata_pop.items()
        #             opti_time = np.around(float(opti_time), 1)
        #             print('last optitrack time not consume:', opti_time)

        #             while opti_time <= min_usrptime:
        #                 if opti_time == min_usrptime:
        #                     data_upload.update({'optitrack':opti_data})
        #                     for key in min_key:
        #                         gwdata_pop.pop(key)
        #                     # opdata_pop.clear()
        #                     break
        #                 if not opti_queue.empty():
        #                     opdata_pop = opti_queue.get()
        #                     (opti_time,opti_data), = opdata_pop.items()
        #                     opti_time = np.around(float(opti_time), 1)
        #                     print('throw old time and get new optitrack time', opti_time)
        #                 else:
        #                     break
        #             if opti_time > min_usrptime:
        #                 for key in min_key:
        #                     gwdata_pop.pop(key)

        #         if 'optitrack' in data_upload:
        #             timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')
        #             data_upload.update({'logTime':logTime, 'phyTime':timestamp})
        #             print('成功对齐时间戳')
        #             yield data_upload, confidence
        #     # time.sleep(0.05)


if __name__ == "__main__":

    with open("conf.yaml") as f:
        kwargs = yaml.safe_load(f)
        f.close()

    sync = SyncData(**kwargs["mq"])
