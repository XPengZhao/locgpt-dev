import rospy
from nav_msgs.msg import Odometry
import pika
import json
from datetime import datetime
import numpy as np


# rabbitmq server
mc = MQPublish(PM.ip, PM.port, 'optitrack')
try:
    mc.connect()
    print("connect to server")
except:
    raise Exception("can't connect to localhost")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

def callback(data):
    position = data.pose.pose.position
    print("Current position: x: {}, y: {}, z: {}".format(position.x, position.y, position.z))
    send_to_rabbitmq(position)


def send_to_rabbitmq(position):
    target_pos = np.array([position.x, position.y, position.z])
    # prepare the message as a JSON object
    message = json.dumps({
        'target': target_pos,
        'timestamp': datetime.now().isoformat()  # add current timestamp
    }, cls=NumpyEncoder)

    mc.sendData(message) # replace 'lio_sam_position' with your desired queue name
    print(" [x] Sent %r" % message)


def listener():
    rospy.init_node('lio_sam_listener', anonymous=True)
    rospy.Subscriber('/lio_sam/mapping/odom', Odometry, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
