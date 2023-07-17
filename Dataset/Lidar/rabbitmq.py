#!user/bin/python
# -*- coding:utf-8 -*-

"""rabbitmq 生产者
"""

import json

import numpy as np
import pika


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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
