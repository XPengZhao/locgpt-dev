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
        position, rotation = data.pose.pose.position, data.pose.pose.rotation
        logger.debug("Current position: x: %s, y: %s, z: %s", position.x, position.y, position.z)

        target_pos = np.array([position.x, position.y, position.z])
        # prepare the message as a JSON object
        message = json.dumps({
            'target': target_pos,
            'timestamp': datetime.now().isoformat()  # add current timestamp
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