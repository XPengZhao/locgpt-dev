import json
import queue
import random
import threading
import time
import numpy as np
from paho.mqtt import client as mqtt_client
import pika


broker = 'localhost'
port = 1883
topic = "silabs/aoa/iq_report/ble-pd-0C4314F46BF8/ble-pd-0C4314EF65A1"
client_id = f'python-mqtt-{random.randint(0, 100)}'

# RabbitMQ setting
rabbitmq_user = 'guest'
rabbitmq_pwd = 'guest'
rabbitmq_host = 'localhost'
rabbitmq_port = 5672
rabbitmq_queue = 'mqtt_rabbitmq'

# MQTT
def connect_mqtt() -> mqtt_client:
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

# MQTT
def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        with open(filename, 'a') as f:
            f.write(f'{str(time.time())}\n{str(time.asctime())}\n{msg.payload.decode()}\n')
        data = json.loads(msg.payload.decode())
        for name, value in data.items():
            data_dict.setdefault(name, queue.LifoQueue()).put(value)

    client.subscribe(topic)
    client.on_message = on_message

def run():
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()


if __name__ == '__main__':
    thread1 = threading.Thread(target=run)