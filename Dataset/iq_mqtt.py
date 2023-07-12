import json
import queue
import random
import threading
import time
import matplotlib.pyplot as plt
import numpy as np
from paho.mqtt import client as mqtt_client

import test
broker = 'localhost'
port = 1883
topic = "silabs/aoa/iq_report/ble-pd-90395E4B4F3F/ble-pd-6C5CB145BB3C"
topic = "silabs/aoa/iq_report/ble-pd-0C4314F468E5/ble-pd-6C5CB145BB3C"
#topic = "silabs/aoa/iq_report/ble-pd-0C4314F468E5/ble-pd-30FB10D78788"
filename = r'C:\Users\linqi\Desktop\620Data\基站倾角60度3米高无防水罩标签2.4米正对\ble_200m.txt'
# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 100)}'
data_dict = {}
diff = []
diff1 = []
ax = plt.subplot(1, 5, 1)
#
ax2 = plt.subplot(1, 5, 2)
ax3 = plt.subplot(1, 5, 3)
ax4 = plt.subplot(1, 5, 4, projection='polar')
ax5 = plt.subplot(1, 5, 5)
x = 0
m = []
m_new = []
xs = []
ys = []
m_l = []
rssi_l = []
points = []
points1 = []
samples = []
channel = 0
queueLock = threading.Lock()
inten = 0
i_l = []
weight = 0
weights = []
w_l = []


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


def draw():
    global x, m, xs, ys, inten, weight, m_new
    while True:
        time.sleep(0.1)
        if len(data_dict) < 4:
            continue
        sa = data_dict["samples"].get()
        cha = data_dict["channel"].get()
        rssi = data_dict["rssi"].get()
        rssi_l.append(rssi)
        data_dict.clear()
        if cha >= 0:
            m, n, el, intensity = test.cal(sa, cha)
            if intensity < 2:
                continue
            m_l.append(m)
            # if len(m_l) > 1:
            #     t = dict(zip(m_new, m))
            #     m_new = [(x + y) for x, y in t.items()]
            m_new = m
            diff.append(n)
            inten += intensity
            i_l.append(intensity)
            tmp = intensity * (np.cos(n / 180 * np.pi) + 1j * np.sin(n / 180 * np.pi))
            weight += tmp
            weights.append(tmp)
            if len(weights) > 10:
                # t = dict(zip(m_new, m_l[0]))
                # m_new = [(x - y) for x, y in t.items()]
                # m_l.pop(0)
                weight -= weights[0]
                inten -= i_l[0]
                weights.pop(0)
                i_l.pop(0)
                index = weights.index(max(weights))
                #print('**********', diff[index])
                #w_l.append(np.angle(weight / intensity))
                w_l.append(n/180*np.pi)
            x += 1
            points.append((x, diff[-1]))
            xs, ys = zip(*points)


def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        global samples
        global channel
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
    thread = threading.Thread(target=draw, daemon=True)
    thread.start()
    thread1.start()
    while True:
        ax.clear()
        ax.scatter(xs, ys)
        ax2.clear()
        # ax2.grid()
        if len(m) >0:
            ax2.imshow(m)
        ax3.clear()
        ax3.plot(w_l)
        if len(w_l) > 0:
            ax4.clear()
            ax4.scatter(w_l[-1], 1, s=100)
        ax5.clear()
        ax5.plot(rssi_l)
        plt.pause(0.001)