import numpy as np


def rosPoseToPosQuat(ros_pose):
    pos = np.zeros((3,))
    quat = np.zeros((4,))
    pos[0] = ros_pose.position.x
    pos[1] = ros_pose.position.y
    pos[2] = ros_pose.position.z
    quat[0] = ros_pose.orientation.x
    quat[1] = ros_pose.orientation.y
    quat[2] = ros_pose.orientation.z
    quat[3] = ros_pose.orientation.w

    return pos, quat
