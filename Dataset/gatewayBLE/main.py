import json
import queue
import random
import time
import numpy as np
from paho.mqtt import client as mqtt_client
from utils.get_timestamp import get_s_today, get_timestamp
from utils.mqpublish import MQPublish
from utils.param_loader import Paramloader
from datetime import datetime

broker = 'localhost'
port = 1883

# gateway 1
topic = "silabs/aoa/iq_report/ble-pd-0C4314F46D2F/ble-pd-0C4314EF65A1"
topic2 = "silabs/aoa/iq_report/ble-pd-0C4314F46D2F/ble-pd-B43A31EEB7B6"

# gateway 2
# topic = "silabs/aoa/iq_report/ble-pd-0C4314F46D0A/ble-pd-0C4314EF65A1"
# topic2 = "silabs/aoa/iq_report/ble-pd-0C4314F46D0A/ble-pd-B43A31EEB7B6"

# gateway 3
# topic = "silabs/aoa/iq_report/ble-pd-0C4314F46D26/ble-pd-0C4314EF65A1"
# topic2 = "silabs/aoa/iq_report/ble-pd-0C4314F46D26/ble-pd-B43A31EEB7B6"

# gateway 4
# topic = "silabs/aoa/iq_report/ble-pd-0C4314F46DBF8/ble-pd-0C4314EF65A1"
# topic2 = "silabs/aoa/iq_report/ble-pd-0C4314F46DBF8/ble-pd-B43A31EEB7B6"

topic_array = [(topic, 0), (topic2, 1)]
client_id = f'python-mqtt-{random.randint(0, 100)}'
filename = r'./data.txt' 


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
def subscribe(client: mqtt_client, mc):
    def on_message(client, userdata, msg):
        # publish message to RabbitMQ
        # channel.basic_publish(exchange='', routing_key=key, body=msg.payload.decode('utf-8'))
        data = json.loads(msg.payload.decode())
        if msg.topic == topic:
            data_pub = {"id":1, "frequency": data["channel"], "timestamp": datetime.now().isoformat(), "rssi": data["rssi"], "samples": data["samples"]}
            print("tag 1")
        elif msg.topic == topic2:
            data_pub = {"id":2, "frequency": data["channel"], "timestamp": datetime.now().isoformat(), "rssi": data["rssi"], "samples": data["samples"]}
            print("tag 2")
        mc.sendData(json.dumps(data_pub))
        print("Message received from MQTT and sent to RabbitMQ")
        # with open(filename, 'a') as f:
        #     f.write(f'{str(time.time())}\n{str(time.asctime())}\n{msg.payload.decode()}\n')
        # data = json.loads(msg.payload.decode())
        # for name, value in data.items():
        #     data_dict.setdefault(name, queue.LifoQueue()).put(value)

    client.subscribe(topic_array)
    client.on_message = on_message
    

if __name__ == '__main__':
    # RabbitMQ
    PM = Paramloader()
    mc = MQPublish(PM.ip, PM.port, PM.gateway_name)
    try:
        mc.connect()
    except:
        print(PM.ip)
        raise Exception("can't connect to server")
    
    broker = 'localhost'
    port = 1883
 
    client_id = f'python-mqtt-{random.randint(0, 100)}'
    filename = r'./data.txt' 
    print(client_id)
    # MQTT collect raw iq data
    client = connect_mqtt()
    subscribe(client, mc)
    client.loop_forever()