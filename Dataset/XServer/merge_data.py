# -*- coding: utf-8 -*-
"""sync data from mongodb by sequence and timestamp
"""
from pymongo import MongoClient
import yaml
from logger import logger

class DataMerge():

    def __init__(self, **kwargs) -> None:
        kwargs_db = kwargs['db']
        collection = kwargs_db['collection']
        self.devices = kwargs_db['devices']

        ## create collection
        client = MongoClient(f"mongodb://tagsys:tagsys@{kwargs_db['ip']}:27017/")
        db = client[kwargs_db['database']]  # tabledatas
        self.collection = db[collection]




    def load_data(self, **kwargs) -> None:
        """load data from mongodb
        """

        ## statistics
        datalen = self.collection.count_documents({})
        logger.info(f"Reading for collection {self.collection.name} with {datalen} records")
        for device in self.devices:
            device_len = self.collection.count_documents({"gateway": device})
            logger.info(f"{device} has {device_len} records")

        gateway1 = self.collection.find({"gateway":"gateway1"},
                                        {'_id': 0}).sort([("timestamp",1)]).allow_disk_use(True)
        gateway2 = self.collection.find({"gateway":"gateway2"},
                                        {'_id': 0}).sort([("timestamp",1)]).allow_disk_use(True)
        gateway3 = self.collection.find({"gateway":"gateway2"},
                                        {'_id': 0}).sort([("timestamp",1)]).allow_disk_use(True)
        lidar = self.collection.find({"gateway":"lidar"},
                                     {'_id': 0}).sort([("timestamp",1)]).allow_disk_use(True)

        for i in gateway1:
            print(i['id'],i["sequence"])






if __name__ == "__main__":

    with open("conf.yaml") as f:
        kwargs = yaml.safe_load(f)
        f.close()

    sync_worker = DataMerge(**kwargs)

    sync_worker.load_data()