# -*- coding: utf-8 -*-
"""collect data offline
"""
import yaml
import json

import threading

import pika
from pymongo import MongoClient
from logger import logger
import time

# There are eight threading write into a same mongodb collection, which may cause problem. Could you add a lock for the threading to asure that at one time there is only one threading can insert into mongodb

class MsgWorker(threading.Thread):
    """Threading for getting message from Gateways Message Queue
    """
    def __init__(self, queue_rb, collection, lock, **kwargs):
        super().__init__()
        self.credentials = pika.PlainCredentials(username=kwargs['user'], password=kwargs['password'])
        self.parameters = pika.ConnectionParameters(host=kwargs['ip'],port=kwargs["port"],
                                                    credentials=self.credentials, heartbeat=0)
        self.connection = pika.BlockingConnection(parameters=self.parameters)
        self.channel = self.connection.channel()

        self.channel.exchange_declare(exchange=kwargs['exchange'], exchange_type='direct')
        self.channel.queue_declare(queue=queue_rb, auto_delete=True)
        self.channel.queue_bind(exchange=kwargs['exchange'], queue=queue_rb, routing_key=queue_rb)

        self.collection = collection
        self.queue_rb = queue_rb
        self.lock = lock  # Add the lock here

        # start listening
        self.channel.queue_purge(self.queue_rb)
        self.channel.basic_consume(queue=self.queue_rb, auto_ack=True, on_message_callback=self.callback)
        self.stop_flag = False  # Add this line

    def run(self):
        logger.info(f"Start listening {self.queue_rb}")
        # print('Waiting for messages. To exit press CTRL+C')
        self.channel.start_consuming()

    def stop(self):
        self.stop_flag = True

    def callback(self, ch, method, properties, body):

        if self.stop_flag:
            self.channel.stop_consuming()

        body_str = body.decode()  # Decode the byte string
        data = json.loads(body_str)  # Parse the JSON string
        data.update({"gateway": self.queue_rb})
        with self.lock:  # Use the lock here
            logger.debug("Received %r", data)
            self.collection.insert(data)


class DataCollect(object):

    def __init__(self, **kwargs) -> None:

        kwargs_db = kwargs['db']
        collection = kwargs_db['collection']
        kwargs_mq = kwargs['mq']

        ## create collection
        client = MongoClient(f"mongodb://tagsys:tagsys@{kwargs_db['ip']}:27017/")
        db = client[kwargs_db['database']]  # tabledatas
        self.collection = db[collection]
        logger.info(f"Create collection {collection}")


        queue_gateway = kwargs_mq['queue_gateway']
        queue_lidar = kwargs_mq['queue_lidar']
        queues_rb = queue_gateway + queue_lidar

        # Create the Threading
        self.lock = threading.Lock()  # Create the lock
        self.workers = [MsgWorker(queue_rb, self.collection, self.lock, **kwargs_mq) for queue_rb in queues_rb]


    def run(self):
        # Start the threading
        for worker in self.workers:
            worker.daemon = True
            worker.start()


    def stop(self):
        # Stop the threading
        for worker in self.workers:
            worker.stop()







if __name__ == "__main__":

    with open("conf.yaml") as f:
        kwargs = yaml.safe_load(f)
        f.close()

    collecter = DataCollect(**kwargs)

    print('Waiting for messages. To exit press CTRL+C')
    try:
        collecter.run()
        while True:
            # This loop will keep the main thread running, effectively
            # making it wait for the worker threads to finish.
            # The time.sleep call is there to prevent it from using 100% CPU.
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted by user, stopping workers...")
        collecter.stop()