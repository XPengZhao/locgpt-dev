import json
import pika
import pymongo
from datetime import datetime, timedelta

# Establish a connection with MongoDB
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["mydatabase"]  # replace with your database name
collection = db["mycollection"]  # replace with your collection name

# Establish a connection with RabbitMQ
credentials = pika.PlainCredentials('guest', 'guest')  # replace with your credentials
parameters = pika.ConnectionParameters('localhost', 5672, '/', credentials)  # replace with your host and port
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

gateway_queues = ['gateway1_queue1', 'gateway1_queue2', 'gateway2_queue1', 'gateway2_queue2',
                  'gateway3_queue1', 'gateway3_queue2', 'gateway4_queue1', 'gateway4_queue2']

merged_data = {}

for queue in gateway_queues:
    # Get the gateway number and tagid from the queue name
    gateway_number = queue.split('_')[0]
    tagid = queue.split('_')[1][-1]

    method_frame, header_frame, body = channel.basic_get(queue)
    if method_frame:
        data = json.loads(body)
        timestamp = datetime.strptime(data['timestamp'], "%Y-%m-%dT%H:%M:%S:%f")

        if gateway_number not in merged_data:
            merged_data[gateway_number] = []

        # Sync the same tag's data by the same sequence
        for item in merged_data[gateway_number]:
            if item['tagid'] == data['tagid'] and item['sequence'] == data['sequence']:
                break
        else:
            # Sync the different tags' data by timestamp <= 100ms
            for item in merged_data[gateway_number]:
                item_timestamp = datetime.strptime(item['timestamp'], "%Y-%m-%dT%H:%M:%S:%f")
                if abs(item_timestamp - timestamp) <= timedelta(milliseconds=100):
                    merged_data[gateway_number].append(data)
                    break

# Merge the lidar data into altimate data by timestamp < 100ms
lidar_data = [...]  # replace with your lidar data
for lidar in lidar_data:
    lidar_timestamp = datetime.strptime(lidar['timestamp'], "%Y-%m-%dT%H:%M:%S:%f")
    for gateway in merged_data.values():
        for item in gateway:
            item_timestamp = datetime.strptime(item['timestamp'], "%Y-%m-%dT%H:%M:%S:%f")
            if abs(item_timestamp - lidar_timestamp) < timedelta(milliseconds=100):
                item.update(lidar)

# Store the merged data in MongoDB
collection.insert_one(merged_data)

# Close the connection with RabbitMQ
connection.close()
