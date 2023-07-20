import json
from datetime import datetime

import numpy as np
import rospy
import yaml
from nav_msgs.msg import Odometry

from rabbitmq import MQPublish, NumpyEncoder
from logger import logger



class Lidar_sniffer():

    def __init__(self, **kwargs) -> None:

        try:
            self.mq = MQPublish(**kwargs['mq'])
            logger.info("connected to rabbitmq server")
        except:
            raise Exception("can't connect to localhost")

        # for ros node
        self.node_name = kwargs['ros_node']['node_name']
        self.sniff_topic = kwargs['ros_node']['sniff_topic']


    def callback(self, data):

        timestamp = data.header.stamp
        timestamp = timestamp.to_sec()
        timestamp = datetime.fromtimestamp(timestamp).isoformat()
        position, orientation = data.pose.pose.position, data.pose.pose.orientation

        logger.debug("Current timestamp %s - position: x: %s, y: %s, z: %s", timestamp, position.x, position.y, position.z)

        # prepare the message as a JSON object
        message = json.dumps({
            'timestamp': timestamp,  # add current timestamp
            'position': [position.x, position.y, position.z],
            'orientation': [orientation.x, orientation.y, orientation.z, orientation.w]
        }, cls=NumpyEncoder)

        self.mq.sendData(message)
        logger.debug(" [x] Sent %r" % message)


    def listener(self):
        rospy.init_node(self.node_name, anonymous=True)
        rospy.Subscriber(self.sniff_topic, Odometry, self.callback)
        rospy.spin()



if __name__ == '__main__':

    with open("conf.yaml") as f:
        kwargs = yaml.safe_load(f)
        f.close()

    sniffer = Lidar_sniffer(**kwargs)
    sniffer.listener()