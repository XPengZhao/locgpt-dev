import rospy
from nav_msgs.msg import Odometry


def callback(data):
    position = data.pose.pose.position
    print("Current position: x: {}, y: {}, z: {}".format(position.x, position.y, position.z))

def listener():
    rospy.init_node('lio_sam_listener', anonymous=True)
    rospy.Subscriber('/lio_sam/mapping/odom', Odometry, callback)
    rospy.spin()


if __name__ == '__main__':
    listener()
