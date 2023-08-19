# -*- coding: utf-8 -*-
"""read mongodb database and save csv
"""
import time
from datetime import datetime

import numpy as np
from pymongo import MongoClient
from pyquaternion import Quaternion
from bartlett import Bartlett
import matplotlib.image as plm
from tqdm import tqdm
import matplotlib.pyplot as plt


mod_phase = lambda x: np.mod(x + np.pi, 2 * np.pi) - np.pi

database = "LocGPT"
ip = "158.132.255.110"
merge_collection = "pq504_exp2_s46_merge"
gateways = ["gateway1", "gateway2", "gateway3", "gateway4"]

## BLE channel
ble_ch_ind = [37]+[i for i in range(0, 11)] + [38] +[i for i in range(11,37)] + [39]
ble_ch_freq = np.linspace(2402e6, 2480e6, 40)
ble_freq_dict = dict(zip(ble_ch_ind, ble_ch_freq))
print(ble_freq_dict)




def get_tag_position(position, orientation):
    """get tag position
    """

    # Position and orientation of p0
    position_p0 = np.array([position[0], position[1], position[2]])
    orientation_p0 = Quaternion(orientation[3], orientation[0],
                                orientation[1], orientation[2])

    # Relative positions of p1 and p2 to p0
    relative_position_tag1 = np.array([0.3, 0, 0])
    relative_position_tag2 = np.array([-0.3, 0, 0])

    # Calculate global positions of p1 and p2
    position_tag1 = position_p0 + orientation_p0.rotate(relative_position_tag1)
    position_tag2 = position_p0 + orientation_p0.rotate(relative_position_tag2)

    return position_tag1, position_tag2



def timestamp2uxtime(timestamp):

    # Convert the timestamp string to a datetime object
    timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")
    unix_time = time.mktime(timestamp.timetuple())
    unix_time = unix_time + timestamp.microsecond / 1e6

    return unix_time


def get_cfo_phase(gateway_constant_phase):
    """get cfo phase per sample (8 samples)
    """
    N = len(gateway_constant_phase)
    cfo_phase = mod_phase(gateway_constant_phase[1:] - gateway_constant_phase[:-1])
    cfo_phase = np.mean(cfo_phase)

    compensate_phase = cfo_phase * np.arange(N)
    phase_after_compensate = mod_phase(gateway_constant_phase - compensate_phase - gateway_constant_phase[0])
    print(f"phase_after_compensate: {np.mean(phase_after_compensate)}")
    cfo = cfo_phase / (2 * np.pi * 1e-6)
    print(f"cfo: {cfo}")

    # plt.figure()
    # plt.plot(gateway_constant_phase)
    # plt.plot(mod_phase(gateway_constant_phase - compensate_phase))
    # plt.savefig("cfo.png")

    return cfo_phase


if __name__ == '__main__':

    ##
    spt_worker =Bartlett()

    ## create collection
    client = MongoClient(f"mongodb://tagsys:tagsys@{ip}:27017/")
    db = client[database]
    collection = db[merge_collection]
    data_len = collection.count_documents({})
    print(f"{merge_collection} has {data_len} records")
    data = collection.find({}, {'_id':0}).sort([("timestamp",1)]).allow_disk_use(True)

    # timestamp, freq,  area, position, phase
    csv_data = np.zeros((data_len*2, 1+1+1+3+4*16))

    for i, item in enumerate(data):
        timestamp = timestamp2uxtime(item['timestamp'])
        tag1_position, tag2_position = get_tag_position(item['position'], item['orientation'])
        tag1_freq = ble_freq_dict[item['gateway1'][0]['frequency']]
        tag2_freq = ble_freq_dict[item['gateway1'][1]['frequency']]

        csv_data[i*2:(i+1)*2, 0] = timestamp
        csv_data[i*2,1], csv_data[i*2+1, 1] = tag1_freq, tag2_freq
        csv_data[i*2,3:6], csv_data[i*2+1, 3:6] = tag1_position, tag2_position


        for j, gateway in enumerate(gateways):
            gateway_tag1_samples = np.array(item[gateway][0]['samples']).reshape((-1,2))  # [82, 2] IQ samples
            gateway_tag1_cfo_samples = gateway_tag1_samples[:8] # [8, 2]
            gateway_tag1_cfo_phase = np.angle(gateway_tag1_cfo_samples[:,0]+1j*gateway_tag1_cfo_samples[:,1])  # [8]
            gateway_tag1_cfo_phase = get_cfo_phase(gateway_tag1_cfo_phase)  # (1,)
            gateway_tag1_compensate_phase = gateway_tag1_cfo_phase * np.arange(0,32,2)
            gateway_tag1_samplesV = gateway_tag1_samples[9:9+32:2]  # [16, 2]
            gateway_tag1_phase = np.angle(gateway_tag1_samplesV[:,0]+1j*gateway_tag1_samplesV[:,1])  # [16]
            gateway_tag1_phase = gateway_tag1_phase - gateway_tag1_compensate_phase  # [16]
            gateway_tag1_phase = mod_phase(gateway_tag1_phase - gateway_tag1_phase[0])  # ralative to the first phase


            gateway_tag2_samples = np.array(item[gateway][1]['samples']).reshape((-1,2))  # [82, 2] IQ samples
            gateway_tag2_cfo_samples = gateway_tag2_samples[:8] # [8, 2]
            gateway_tag2_cfo_phase = np.angle(gateway_tag2_cfo_samples[:,0]+1j*gateway_tag2_cfo_samples[:,1])  # [8]
            gateway_tag2_cfo_phase = get_cfo_phase(gateway_tag2_cfo_phase)  # (1,)
            gateway_tag2_compensate_phase = gateway_tag2_cfo_phase * np.arange(0,32,2)
            gateway_tag2_samplesV = gateway_tag2_samples[9:9+32:2]  # [16, 2]
            gateway_tag2_phase = np.angle(gateway_tag2_samplesV[:,0]+1j*gateway_tag2_samplesV[:,1])  # [16]
            gateway_tag2_phase = gateway_tag2_phase - gateway_tag2_compensate_phase  # [16]
            gateway_tag2_phase = (gateway_tag2_phase - gateway_tag2_phase[0])  # ralative to the first phase
            gateway_tag2_phase = np.mod(gateway_tag2_phase + np.pi, 2 * np.pi) - np.pi  # [-pi, pi]



            csv_data[i*2, 6+16*j:6+16*(j+1)] = gateway_tag1_phase
            csv_data[i*2+1, 6+16*j:6+16*(j+1)] = gateway_tag2_phase

    csv_data[:,0] = csv_data[:,0] - csv_data[0,0]  # time offset

    csv_header = "timestamp, freq, area, pos_x, pos_y, pos_z"
    phase_header = ["g%d_p%d"%(i,j) for i in range(1,5) for j in range(1,17)]
    phase_header = ", ".join(phase_header)
    csv_header = csv_header + ", " + phase_header

    np.savetxt(f"{merge_collection}.csv", csv_data, delimiter=",", header=csv_header, fmt='%.03f', comments="")


        # spt = spt_worker.get_aoa_heatmap(gateway1_tag1_phase.reshape((1,16)), tag1_freq)
        # figure = np.zeros((9,36,3))
        # figure[:,:,0] = spt.cpu().numpy()
        # plm.imsave(f"test.png", figure)
        # gateway1_tag2_samples = item['gateway1'][1]['samples']





