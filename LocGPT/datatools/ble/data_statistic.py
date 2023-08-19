# -*- coding: utf-8 -*-
"""statistic ble data
"""
import numpy as np
from pymongo import MongoClient

## gateway position

gateway1_pos = [-1.69, -2.90, -0.1]
gateway2_pos = [1.20, -1.60, -0.01]
gateway3_pos = [3.07, 8.22, 0.07]
gateway4_pos = [-2.23, 4.70, -0.1]

# gateway_pos = np.array([gateway1_pos, gateway2_pos, gateway3_pos, gateway4_pos])
# center_pos = np.mean(gateway_pos, axis=0)

# dis_to_center = np.linalg.norm(gateway_pos - center_pos, axis=1)
# print(dis_to_center)
# dis = np.mean(dis_to_center)
# print(gateway_pos/dis)


database = "LocGPT"
ip = "158.132.255.110"
merge_collection = "pq504_exp4_s48_merge"
gateways = ["gateway1", "gateway2", "gateway3", "gateway4"]

def get_mean_rss():
    """get mean rss
    """

    client = MongoClient(f"mongodb://tagsys:tagsys@{ip}:27017/")
    db = client[database]
    collection = db[merge_collection]
    data_len = collection.count_documents({})
    print(f"{merge_collection} has {data_len} records")
    data = collection.find({}, {'_id':0}).sort([("timestamp",1)]).allow_disk_use(True)
    all_pos = np.zeros((data_len, 3))
    all_rssi = []
    for i, item in enumerate(data):
        all_pos[i] = np.array(item['position'])
        for j, gateway in enumerate(gateways):
            rssi = np.array(item[gateway][0]['rssi'])
            all_rssi.append(rssi)

    all_rssi = np.array(all_rssi)
    print("mean rssi: ", np.mean(all_rssi))
    pos_to_gateway1 = np.mean(np.linalg.norm(all_pos - gateway1_pos, axis=1))
    pos_to_gateway2 = np.mean(np.linalg.norm(all_pos - gateway2_pos, axis=1))
    pos_to_gateway3 = np.mean(np.linalg.norm(all_pos - gateway3_pos, axis=1))
    pos_to_gateway4 = np.mean(np.linalg.norm(all_pos - gateway4_pos, axis=1))
    pos_to_gateway = np.mean([pos_to_gateway1, pos_to_gateway2, pos_to_gateway3, pos_to_gateway4])

    area = (np.max(all_pos[:,0]) - np.min(all_pos[:,0])) * (np.max(all_pos[:,1]) - np.min(all_pos[:,1]))
    print("area: ", area)
    print("pos_to_gateway: ", pos_to_gateway)


if __name__ == '__main__':
    get_mean_rss()