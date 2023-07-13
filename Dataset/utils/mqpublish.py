#!user/bin/python
# -*- coding:utf-8 -*-

"""rabbitmq 生产者
"""

import pika


class MQClient():
    """
    RabbitMQ客户端，对应接收机端
    """
    def __init__(self, hostname, port, user, pwd):
        """hostname 服务器IP地址； port 服务器RabbitMQ服务端口
        """
        self.queue = "oss.url_test"      #消息队列名称
        self.routing_key = "url_test"
        self.exchange = "oss_test"
        self.hostname = hostname
        self.port = port
        self.user = user
        self.pwd = pwd
        self.channel = None
        self.connection = None


    def connect(self):
        """
        连接服务器
        """
        credentials = pika.PlainCredentials(username=self.user, password=self.pwd)
        parameters = pika.ConnectionParameters(host=self.hostname, port=self.port, credentials=credentials)
        self.connection = pika.BlockingConnection(parameters=parameters)

        # 创建通道
        self.channel = self.connection.channel()

        # 创建broker
        self.channel.exchange_declare(exchange=self.exchange, exchange_type='direct', durable=True)       #创建exchange
        self.channel.queue_declare(queue=self.queue, durable=True)                                        #创建队列
        self.channel.queue_bind(queue=self.queue, exchange=self.exchange, routing_key=self.routing_key)   #绑定exchange和队列

    def sendData(self, data):
        """发送消息
        """
        self.channel.basic_publish(exchange=self.exchange, routing_key=self.routing_key, body=data)
        # print("Send completed")


    def disconnect(self):
        """关闭连接
        """
        self.connection.close()


class MQPublish():
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
        self.channel = None
        self.connection = None


    def connect(self):
        """连接服务器
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
        print("Send completed")


    def disconnect(self):
        """关闭连接
        """
        self.connection.close()
