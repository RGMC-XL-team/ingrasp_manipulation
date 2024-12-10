#!/usr/bin/env python
from __future__ import annotations

import os
import sys
import time
from math import radians
from pathlib import Path
from typing import Literal

import numpy as np
import rospkg
import yaml

# Manually add path if not in ROS
if "ROS_MASTER_URI" not in os.environ:
    current_file = Path(__file__).resolve()
    repo_root_dir = current_file.parent.parent.parent
    sys.path.append(str(repo_root_dir / "leap_model_based" / "src"))
    sys.path.append(str(repo_root_dir / "leap_utils" / "src"))
    os.environ["ROS_PACKAGE_PATH"] = str(repo_root_dir)

from leaphand_mujoco import Simulation
from leap_model_based.leaphand_pinocchio import LeapHandPinocchio
from leap_utils.mingrui.utils_calc import posQuat2Isometry3d

move_option = Literal["from_real", "from_last_target"]

def vec_normalize(vec):
    return vec / np.linalg.norm(vec)

class LeapHandControl:
    def __init__(self, robot_model: LeapHandPinocchio, use_real_hardware: bool = False) -> None:
        self.use_real_hardware = use_real_hardware
        self.robot_model = robot_model

        self.control_rate = 50
        self.last_start_time = None

        # hyper-options / parameters
        self.back_to_initial_config = True
        self.second_finger_id = "finger2"  # "finger1" or "finger2"
        self.T = 3  # the trajectory length is (T+1)
        self.max_control_time = 20
        self.max_control_iter = 5 # (N_replan + 1)

        self.env = Simulation(robot_model=robot_model)
        self.hand_target_joint_pos = self.env.getHandJointPos().copy()

        if int(1.0 / self.env.timestep) % self.control_rate != 0:
            raise NameError("Simulation rate % control rate != 0")

    def taskGoalCb(self, msg):
        self.task_goal = msg

    def updateCurrentHandJointPos(self):
        self.hand_curr_joint_pos = self.env.getHandJointPos().copy()

    def getFingerJointPos(self, finger_name, option: move_option = "from_last_target"):
        if option == "from_real":
            finger_joint_pos = self.hand_curr_joint_pos[self.robot_model.finger_joints_id_in_hand[finger_name]]
        elif option == "from_last_target":
            finger_joint_pos = self.hand_target_joint_pos[self.robot_model.finger_joints_id_in_hand[finger_name]]
        else:
            raise NameError("Invalid option.")
        return finger_joint_pos

    def getFingerGlobalPose(self, finger_name, local_position=None, option: move_option = "from_last_target"):
        return self.robot_model.getTcpGlobalPose(
            finger_name, self.getFingerJointPos(finger_name, option), local_position=local_position
        )  # finger_tcp_pos, finger_tcp_quat

    def moveHandToJointPos(self, target_joint_pos, max_speed=radians(45), option: move_option = "from_last_target"):
        """
        Move the hand to the target joint positions along a joint-space interpolated path
        """
        if option == "from_real":
            self.updateCurrentHandJointPos()
            current_joint_pos = self.hand_curr_joint_pos.copy()
        elif option == "from_last_target":
            current_joint_pos = self.hand_target_joint_pos.copy()
        else:
            raise NameError("Invalid option.")

        target_joint_pos = np.array(target_joint_pos)
        max_step_size = max_speed * (1.0 / self.control_rate)

        n_steps = np.max(np.abs(target_joint_pos - current_joint_pos) / max_step_size)
        n_steps = np.ceil(n_steps)

        for i in range(1, int(n_steps) + 1):
            t = float(i) / n_steps
            temp_target_joint_pos = current_joint_pos * (1 - t) + target_joint_pos * t
            self.env.ctrlHandJointPos(temp_target_joint_pos)
            for _ in range(int(1.0 / self.control_rate / self.env.timestep)):
                self.env.step()

        self.hand_target_joint_pos = target_joint_pos

    def moveHandToInitialConfig(self):
        target_hand_joint_pos = np.array([0, 0, 0, 0, 0, -0.3, 0, 0, 0, 0, 0, 0, np.pi / 2, 0, 0, 0])
        self.moveHandToJointPos(target_hand_joint_pos, max_speed=radians(90), option="from_last_target")

    def initialGrasping(self):
        """
        Initial grasping strategy for the known cylinder object
        """
        self.updateCurrentHandJointPos()
        object_pos, object_quat = self.env.getObjectPose()
        fingertip_radius = 0.008

        # First, move the hand to a config near the grasping config, but no contact
        # Second, move the hand to the grasping config and make the contact
        for radius in [0.03 + 0.012, 0.03 + fingertip_radius]:
            finger0_grasp_offset = np.array(
                [radius / np.sqrt(2) - 0.004, -0.02, radius / np.sqrt(2)]
            )  # manually defined
            finger1_grasp_offset = np.array([radius / np.sqrt(2), -0.02, -radius / np.sqrt(2)])
            thumb_grasp_offset = np.array([-radius, -0.02, 0.005])

            traj_hand_joint_pos, _ = self.robot_model.trajOptOurs(
                T=1,
                delta_t=1.0,
                object_target_pose=posQuat2Isometry3d(object_pos, object_quat),
                thumb_target_rel_pose=posQuat2Isometry3d(thumb_grasp_offset, [0, 0, 0, 1]),
                finger0_target_rel_pose=posQuat2Isometry3d(finger0_grasp_offset, [0, 0, 0, 1]),
                finger1_target_rel_pose=posQuat2Isometry3d(finger1_grasp_offset, [0, 0, 0, 1])
                if self.second_finger_id == "finger1"
                else None,
                finger2_target_rel_pose=posQuat2Isometry3d(finger1_grasp_offset, [0, 0, 0, 1])
                if self.second_finger_id == "finger2"
                else None,
                weights_object_pose=[100, 100, 100, 10, 10, 10],
                weights_rel_pose=[10, 10, 10, 0, 0, 0],
                weights_joint_vel=1e-4,
                object_pose_init=posQuat2Isometry3d(object_pos, object_quat),
                hand_joint_pos_init=self.hand_target_joint_pos.copy(),
            )

            hand_target_joint_pos = traj_hand_joint_pos[-1, :]
            self.moveHandToJointPos(target_joint_pos=hand_target_joint_pos, max_speed=radians(90))

            for i in range(int(1.0 / self.env.timestep)):
                self.env.step()

        self.moveHandToJointPos(target_joint_pos=hand_target_joint_pos, max_speed=radians(90))
        for i in range(int(1.0 / self.env.timestep)):
            self.env.step()

    def moveObject(self, target_object_pos=None, target_object_quat=None, target_rel_movement=None):
        """
        Move the object to a desired position
        """
        if target_rel_movement is not None:
            object_pos, object_quat = self.env.getQRCodePose()
            target_object_pos = object_pos + target_rel_movement
            target_object_quat = object_quat.copy()
        if target_object_pos is not None:
            target_object_pos = target_object_pos.copy()
            target_object_quat = target_object_quat.copy()

        self.env.visDesiredPos(target_object_pos)
        self.env.step(refresh=True)

        control_errs = []
        plan_errs = []
        plan_times = []
        all_traj_hand_joint_pos = []

        # re-plan and re-execute to improve the final accuracy
        for control_iter in range(self.max_control_iter):
            self.updateCurrentHandJointPos()

            object_pos, object_quat = self.env.getQRCodePose()
            object_pose = posQuat2Isometry3d(object_pos, object_quat)
            finger0_pos, finger0_quat = self.getFingerGlobalPose(
                "finger0", local_position=[0, 0, 0], option="from_last_target"
            )
            finger0_pose = posQuat2Isometry3d(finger0_pos, finger0_quat)
            finger1_pos, finger1_quat = self.getFingerGlobalPose(
                self.second_finger_id, local_position=[0, 0, 0], option="from_last_target"
            )
            finger1_pose = posQuat2Isometry3d(finger1_pos, finger1_quat)
            thumb_pos, thumb_quat = self.getFingerGlobalPose(
                "thumb", local_position=[0, 0, 0], option="from_last_target"
            )
            thumb_pose = posQuat2Isometry3d(thumb_pos, thumb_quat)

            t_opt_start = time.perf_counter()
            thumb_pose_in_object = np.linalg.inv(object_pose) @ thumb_pose
            finger0_pose_in_object = np.linalg.inv(object_pose) @ finger0_pose
            finger1_pose_in_object = np.linalg.inv(object_pose) @ finger1_pose
            traj_hand_joint_pos, planned_object_err = self.robot_model.trajOptOurs(
                T=self.T if control_iter == 0 else 1,
                delta_t=1.5 / self.T if control_iter == 0 else 0.2,
                object_target_pose=posQuat2Isometry3d(target_object_pos, target_object_quat),
                thumb_target_rel_pose=thumb_pose_in_object,
                finger0_target_rel_pose=finger0_pose_in_object,
                finger1_target_rel_pose=finger1_pose_in_object if self.second_finger_id == "finger1" else None,
                finger2_target_rel_pose=finger1_pose_in_object if self.second_finger_id == "finger2" else None,
                weights_object_pose=[10, 10, 10, 0.01, 0.01, 0.00],
                weights_rel_pose=[10, 10, 10, 0.001, 0.001, 0.001],
                weights_joint_vel=1e-4 if control_iter == 0 else 2e-4,
                object_pose_init=object_pose,
                hand_joint_pos_init=self.hand_target_joint_pos.copy(),
                b_print_opt_details=False,
            )

            # record the time cost of trajectory optimization
            t_opt_end = time.perf_counter()
            opt_time = t_opt_end - t_opt_start

            # actuate the hand to move along the planned trajectory
            for i, hand_joint_pos in enumerate(traj_hand_joint_pos):
                self.moveHandToJointPos(target_joint_pos=hand_joint_pos, max_speed=radians(20))
            all_traj_hand_joint_pos.extend(traj_hand_joint_pos)

            # wait for 0.5 second
            for _ in range(int(0.5 / self.env.timestep)):
                self.env.step()

            object_pos, _ = self.env.getQRCodePose()
            control_err = np.linalg.norm(object_pos - target_object_pos)
            control_errs.append(float(control_err))
            plan_errs.append(float(planned_object_err))
            plan_times.append(float(opt_time))
            print(
                f"replan_iter: {control_iter}, planned_err: {planned_object_err}, actual_err: {control_err}, opt_time: {opt_time}"
            )

            # the criterion for switching to the next waypoint
            if control_err < planned_object_err:
                print("Stop: control_err < planned_object_err")
                break
            if time.time() - self.last_start_time > self.max_control_time:
                print("Stop: timeout")
                break

        self.last_start_time = time.time()

        # back to the intial configuration (currently, sliding easily happens in simulation and leads to failure)
        if self.back_to_initial_config:
            for i, hand_joint_pos in enumerate(np.flip(all_traj_hand_joint_pos, axis=0)):
                self.moveHandToJointPos(target_joint_pos=hand_joint_pos, max_speed=radians(30))
            for i in range(int(1.0 / self.env.timestep)):  # wait for one second
                self.env.step()

        return control_errs, plan_errs, plan_times

# ------------------------------------------------------------------
def read_configs():
    rospack = rospkg.RosPack()
    urdf_path = Path(rospack.get_path("my_robot_description")) / "urdf" / "leaphand_taskA.urdf"
    robot_model = LeapHandPinocchio(urdf_path=str(urdf_path))

    task_cfg_path = Path(rospack.get_path("leap_task_A")) / "config" / "taskA_corners.yaml"
    with open(task_cfg_path) as stream:
        task_cfg = yaml.safe_load(stream)
    target_waypoints = task_cfg["waypoints"]
    return robot_model, target_waypoints


# ------------------------------------------------------------------
def sim_test():
    robot_model, target_waypoints = read_configs()

    ctrl = LeapHandControl(robot_model=robot_model, use_real_hardware=False)

    for i in range(1):  # repeat the whole task
        ctrl.moveHandToInitialConfig()
        ctrl.initialGrasping()

        ctrl.last_start_time = time.time()

        object_pos, object_quat = ctrl.env.getQRCodePose()
        all_control_err = []
        all_plan_err = []
        all_plan_time = []

        for waypoint_idx, target_waypoint in enumerate(target_waypoints):
            print(f"------ waypoint_idx: {waypoint_idx} ------")

            target_pos = np.array([target_waypoint["x"], target_waypoint["y"], target_waypoint["z"]])
            target_object_pos = object_pos + target_pos

            control_errs, plan_errs, plan_times = ctrl.moveObject(
                target_object_pos=target_object_pos, target_object_quat=object_quat
            )
            all_control_err.append(control_errs)
            all_plan_err.append(plan_errs)
            all_plan_time.append(plan_times)

        ctrl.env.physics.reset()


# ----------------------------------------------
if __name__ == "__main__":
    sim_test()
