# -*- coding:utf-8 -*-
"""rabbitmq 生产者，向GUI发送数据
"""

import json
import os

import numpy as np
import pika


class NpEncoder(json.JSONEncoder):
    """json dump data encoder
    """
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

class MQPublish():
    """Rabbitmq Publisher, send data to GUI Server
    """

    def __init__(self, hostname, port):
        """hostname: RabbitMQ server IP; port: RabbitMQ server Port
        """
        self.queue = "oss.url_test"
        self.routing_key = "url_test"
        self.exchange = "oss_test"
        self.hostname = hostname
        self.port = port
        self.channel = None
        self.connection = None


    def connect(self):
        """
        连接服务器
        """
        credentials = pika.PlainCredentials(username='admin', password='admin')
        parameters = pika.ConnectionParameters(host=self.hostname, port=self.port, credentials=credentials)
        self.connection = pika.BlockingConnection(parameters=parameters)

        # 创建通道
        self.channel = self.connection.channel()

        # 创建broker
        self.channel.exchange_declare(exchange=self.exchange, exchange_type='direct', durable=True)       #创建exchange
        self.channel.queue_declare(queue=self.queue, durable=True)                                        #创建队列
        self.channel.queue_bind(queue=self.queue, exchange=self.exchange, routing_key=self.routing_key)   #绑定exchange和队列
        self.channel.cancel()
        self.channel.queue_purge("oss.url_test")
        self.connection.process_data_events()

    def sendData(self, data):
        """发送消息
        """
        self.channel.basic_publish(exchange=self.exchange, routing_key=self.routing_key, body=data)
        # print("Send completed")


    def disconnect(self):
        """关闭连接
        """
        self.connection.close()


class MQPublish_gateway():
    """
    RabbitMQ客户端，对应接收机端
    """

    def __init__(self, hostname, port, routing_key):
        """hostname IP； port 端口；
        """
        self.routing_key = routing_key
        self.exchange = "direct_gateway"
        self.hostname = hostname
        self.port = port


    def connect(self):
        """Connect to Rabbitmq Server
        """
        credentials = pika.PlainCredentials(username='admin', password='admin')
        parameters = pika.ConnectionParameters(host=self.hostname, port=self.port, credentials=credentials)
        self.connection = pika.BlockingConnection(parameters=parameters)

        # 创建通道
        self.channel = self.connection.channel()

        # 创建broker
        self.channel.exchange_declare(exchange=self.exchange, exchange_type='direct')       #创建exchange


    def sendData(self, data):
        """发送消息
        """
        self.channel.basic_publish(exchange=self.exchange, routing_key=self.routing_key, body=data)
        # print("Send completed")


    def disconnect(self):
        """关闭连接
        """
        self.connection.close()
