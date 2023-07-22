#!user/bin/python
# -*- coding:utf-8 -*-
"""rabbitmq
"""

import json

import numpy as np
import pika


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



import pika

class MQConsumer():
    """RabbitMQ Consumer
    """
    def __init__(self, ip, port, user, password, exchange, routing_key, queue):
        """
        """
        self.exchange, self.routing_key, self.queue = exchange, routing_key, queue

        # connecting to server
        credentials = pika.PlainCredentials(username=user, password=password)
        parameters = pika.ConnectionParameters(host=ip, port=port, credentials=credentials)
        self.connection = pika.BlockingConnection(parameters=parameters)

        # create exchange
        self.channel = self.connection.channel()
        self.channel.exchange_declare(exchange=exchange, exchange_type='direct')

        # create queue
        self.channel.queue_declare(queue=queue, durable=True)
        self.channel.queue_bind(exchange=exchange, queue=queue, routing_key=routing_key)

    def callback(self, ch, method, properties, body):
        """Process received message
        """
        print("Received %r" % body)

    def start_consuming(self):
        """Start consuming messages
        """
        self.channel.basic_consume(queue=self.queue, on_message_callback=self.callback, auto_ack=True)
        print('Waiting for messages. To exit press CTRL+C')
        self.channel.start_consuming()

    def disconnect(self):
        """Close connection
        """
        self.connection.close()


class MQPublish():
    """RabbitMQ Producer
    """
    def __init__(self, ip, port, user, password, exchange, routing_key):
        """
        """
        self.exchange, self.routing_key = exchange, routing_key

        ## connecting to server
        credentials = pika.PlainCredentials(username=user, password=password)
        parameters = pika.ConnectionParameters(host=ip, port=port, credentials=credentials)
        self.connection = pika.BlockingConnection(parameters=parameters)

        # create exchange
        self.channel = self.connection.channel()
        self.channel.exchange_declare(exchange=exchange, exchange_type='direct')       #创建exchange


    def sendData(self, data):
        """发送消息
        """
        self.channel.basic_publish(exchange=self.exchange, routing_key=self.routing_key, body=data)


    def disconnect(self):
        """关闭连接
        """
        self.connection.close()
