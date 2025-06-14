import os
import re

import mujoco.viewer
import numpy as np
import rospkg
from dm_control import mjcf, mujoco
from scipy.spatial.transform import Rotation as sciR

import leap_utils.mingrui.utils_calc as ucalc
from leap_model_based.leaphand_pinocchio import LeapHandPinocchio


# -----------------------------------------------------------
class Arena:
    def __init__(self):
        rospack = rospkg.RosPack()

        self.model = mjcf.RootElement(model="arena")

        scene = mjcf.from_file(os.path.join(rospack.get_path("my_robot_description"), "urdf/objects/scene.xml"))
        self.model.attach(scene)

        xml_file_path = os.path.join(rospack.get_path("my_robot_description"), "urdf/leaphand_xml/leaphand_mujoco.xml")
        with open(xml_file_path) as file:
            xml_content = file.read()
        xml_content = xml_content.replace(
            "./leaphand.xml", os.path.join(os.path.dirname(xml_file_path), "leaphand.xml")
        )
        self.leap_hand = mjcf.from_xml_string(xml_content)
        self.model.attach(self.leap_hand)

        xml_file_path = os.path.join(rospack.get_path("my_robot_description"), "urdf/objects/cylinder_mujoco.xml")
        with open(xml_file_path) as file:
            xml_content = file.read()
        xml_content = xml_content.replace(
            "./cylinder.xml", os.path.join(os.path.dirname(xml_file_path), "cylinder.xml")
        )
        self.object = mjcf.from_xml_string(xml_content)

        object_quat = ucalc.quatXYZW2WXYZ(
            sciR.from_euler("xyz", [90, 0, 0], degrees=True).as_quat()
        )  # vertical cylinder
        object_site = self.model.worldbody.add(
            "site", name="object_site", pos=[-0.025, 0.035, 0.15], quat=object_quat, rgba=[0, 0, 0, 0]
        )
        object_site.attach(self.object).add("joint", type="free", damping="0.0001")

        self.model.worldbody.add(
            "site", name="desired_pos", pos=[0, 0, 1.0], rgba=[0, 1, 0, 0.8], type="sphere", size=[0.005]
        )


# -----------------------------------------------------------
class Simulation:
    def __init__(self, robot_model):
        self.P = robot_model
        self.initialMujoco()

        self.viewer_fps = 20
        self.n_step = 0

    # --------------------------------------
    def initialMujoco(self):
        self.arena = Arena()
        self.model = self.arena.model
        self.physics = mjcf.Physics.from_mjcf_model(self.model)
        self.timestep = 0.0005  # unit: s
        self.physics.model.opt.timestep = self.timestep
        self.viewer = mujoco.viewer.launch_passive(self.physics.model.ptr, self.physics.data.ptr)
        self.viewer.cam.distance = 0.7  # change camera position

        self.mujoco_joints = {}
        for part_joints_name in list(self.P.part_joints_name.values()):
            for joint_name in part_joints_name:
                self.mujoco_joints[joint_name] = self.arena.leap_hand.find("joint", joint_name)

        self.mujoco_actuators = {}
        for joint_name in self.P.part_joints_name["hand"]:
            number = re.search(R"\d+$", joint_name).group()
            self.mujoco_actuators[joint_name] = self.arena.leap_hand.find(
                "actuator", "actuator_" + number
            )  # take care of the names of the actuators and joints

        self.hand_kp = 20.0
        self.hand_target_joint_pos = self.getHandJointPos().copy()

    # --------------------------------------
    def step(self, refresh=False):
        self.physics.step()  # move to the next step

        viewer_fps = self.viewer_fps
        if self.n_step % (int(1.0 / self.timestep) / viewer_fps) == 0 or refresh:
            self.viewer.sync()
        # time.sleep(self.timestep * 0.8)

        self.n_step += 1

    # --------------------------------------
    def getHandJointPos(self):
        joint_pos = np.zeros((len(self.P.part_joints_name["hand"]),))

        for i, joint_name in enumerate(self.P.part_joints_name["hand"]):
            joint_pos[i] = self.physics.bind(self.mujoco_joints[joint_name]).qpos[0]

        return joint_pos

    # --------------------------------------
    def ctrlHandJointPos(self, target_joint_pos):
        self.P.checkJointDim("hand", target_joint_pos)

        for i, joint_name in enumerate(self.P.part_joints_name["hand"]):
            self.physics.bind(self.mujoco_actuators[joint_name]).ctrl = target_joint_pos[i]

    # --------------------------------------
    def getObjectPose(self):
        object = self.arena.object.find("body", "object")
        pos = np.array(self.physics.bind(object).xpos)
        quat = ucalc.quatWXYZ2XYZW(self.physics.bind(object).xquat)
        return pos, quat

    # --------------------------------------
    def getQRCodePose(self):
        qr_code_center = self.physics.bind(self.arena.object.find("site", "QR_code_pos"))
        pos = np.array(qr_code_center.xpos)
        _, object_quat = self.getObjectPose()
        return pos, object_quat

    # --------------------------------------
    def visDesiredPos(self, pos):
        self.physics.bind(self.model.find("site", "desired_pos")).pos = pos


# --------------------------------------
def test1():
    rospack = rospkg.RosPack()
    urdf_path = os.path.join(rospack.get_path("my_robot_description"), "urdf/leaphand_taskA.urdf")
    robot_model = LeapHandPinocchio(urdf_path=urdf_path)

    sim = Simulation(robot_model=robot_model)

    while True:
        sim.step()


# ----------------------------------------------
if __name__ == "__main__":
    test1()
