import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pinocchio as pin
import rospkg
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as sciR

# Manually add path if not in ROS
if "ROS_MASTER_URI" not in os.environ:
    current_file = Path(__file__).resolve()
    repo_root_dir = current_file.parents[3]
    sys.path.append(str(repo_root_dir / "leap_utils" / "src"))
    os.environ["ROS_PACKAGE_PATH"] = str(repo_root_dir)

import leap_utils.mingrui.utils_calc as ucalc


class LeapHandPinocchio:
    def __init__(self, urdf_path: str) -> None:
        self.urdf_path = urdf_path
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        self.part_joints_name = {}
        self.part_joints_name["finger0"] = ["joint_0", "joint_1", "joint_2", "joint_3"]
        self.part_joints_name["finger1"] = ["joint_4", "joint_5", "joint_6", "joint_7"]
        self.part_joints_name["finger2"] = ["joint_8", "joint_9", "joint_10", "joint_11"]
        self.part_joints_name["thumb"] = ["joint_12", "joint_13", "joint_14", "joint_15"]
        self.part_joints_name["hand"] = (
            self.part_joints_name["finger0"]
            + self.part_joints_name["finger1"]
            + self.part_joints_name["finger2"]
            + self.part_joints_name["thumb"]
        )

        self.tcp_links_name = {}
        self.tcp_links_name["finger0"] = "finger1_tip_center"
        self.tcp_links_name["finger1"] = "finger2_tip_center"
        self.tcp_links_name["finger2"] = "finger3_tip_center"
        self.tcp_links_name["thumb"] = "thumb_tip_center"

        self.finger_joints_id_in_hand = {
            "finger0": [0, 1, 2, 3],
            "finger1": [4, 5, 6, 7],
            "finger2": [8, 9, 10, 11],
            "thumb": [12, 13, 14, 15],
        }

        self.joints_name_to_id = {}
        for id, name in enumerate(self.model.names):
            if name != "universe":
                self.joints_name_to_id[name] = id - 1  # only used when interacting with pinocchio

        self.part_joints_id = {}
        for part_name, joints_name in self.part_joints_name.items():
            joints_id = [self.joints_name_to_id[name] for name in joints_name]
            self.part_joints_id[part_name] = joints_id

        self.tcp_links_id = {}
        for part_name, link_name in self.tcp_links_name.items():
            self.tcp_links_id[part_name] = self.model.getFrameId(link_name)

        # id: 0 ~ 15, used for specifying input joint pos
        self.joints_id_to_lower_limit = self.model.lowerPositionLimit
        self.joints_id_to_upper_limit = self.model.upperPositionLimit
        self.joints_id_to_upper_limit[self.joints_name_to_id["joint_12"]] = 1.9  # manually adjusted

    def renewModelAndData(self):
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()

    def checkJointDim(self, part_name, joints):
        if isinstance(joints, list):
            return len(joints) == len(self.part_joints_id[part_name])
        else:
            return joints.size == len(self.part_joints_id[part_name])

    def jointOrderPartUserToAllPin(self, part_name, q_part_normal):
        q_all_pin = np.zeros((len(self.joints_name_to_id),))
        q_all_pin[self.part_joints_id[part_name]] = q_part_normal
        return q_all_pin

    def updateFK(self, part_name, part_joint_pos):
        """
        input:
            hand_joint_pos: the joint order is 0, 1, 2, ... 15, the same as the joint name
        """
        if not self.checkJointDim(part_name, part_joint_pos):
            raise NameError("The dim of input joint is wrong !")

        q_pin = self.jointOrderPartUserToAllPin(part_name, part_joint_pos)
        pin.framesForwardKinematics(self.model, self.data, q_pin)

    def updateFKvel(self, part_name, part_joint_pos, part_joint_vel):
        """
        input:
            hand_joint_pos: the joint order is 0, 1, 2, ... 15, the same as the joint name
            hand_joint_vel: the joint order is 0, 1, 2, ... 15, the same as the joint name
        """
        if not self.checkJointDim(part_name, part_joint_pos):
            raise NameError("The dim of input joint is wrong !")

        if not self.checkJointDim(part_name, part_joint_vel):
            raise NameError("The dim of input joint is wrong !")

        q_pin = self.jointOrderPartUserToAllPin(part_name, part_joint_pos)
        v_pin = self.jointOrderPartUserToAllPin(part_name, part_joint_vel)
        # This only updates joint data
        pin.forwardKinematics(self.model, self.data, q_pin, v_pin)
        # Update frame data
        pin.updateFramePlacements(self.model, self.data)

    def getTcpGlobalPose(self, part_name, part_joint_pos=None, local_position=None):
        if part_joint_pos is not None:
            self.updateFK(part_name, part_joint_pos)
        return self.getFrameGlobalPose(self.tcp_links_name[part_name])

    def getFrameGlobalPose(self, frame_name):
        data = self.data.oMf[self.model.getFrameId(frame_name)]
        pos = data.translation
        rot_mat = data.rotation
        return np.array(pos), sciR.from_matrix(rot_mat).as_quat()

    def getFrameGlobalVelocity(self, frame_name):
        """
        This should be called after updateFKvel
        """
        data = pin.getFrameVelocity(
            self.model, self.data, self.model.getFrameId(frame_name), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        linear = data.linear
        angular = data.angular
        return linear, angular

    def updateJacobians(self, part_name, part_joint_pos):
        if not self.checkJointDim(part_name, part_joint_pos):
            raise NameError("The dim of input joint is wrong !")
        if part_name != "hand":
            raise NameError("The part_name must be 'hand'.")

        q_pin = self.jointOrderPartUserToAllPin(part_name, part_joint_pos)
        pin.computeJointJacobians(self.model, self.data, q_pin)  # call FK internally
        pin.updateFramePlacements(self.model, self.data)

    def getGlobalJacobian(self, part_name, part_joint_pos=None, local_position=None, joint_part_name=None):
        """
        input:
            part_name: which part the target tcp belongs to.
            joint_part_name: which part the joints belong to.
        """
        if part_joint_pos is not None:
            q_pin = self.jointOrderPartUserToAllPin(part_name, part_joint_pos)
            jaco_pin = pin.computeFrameJacobian(
                self.model,
                self.data,
                q=q_pin,
                frame_id=self.tcp_links_id[part_name],
                reference_frame=pin.LOCAL_WORLD_ALIGNED,
            )
        else:
            jaco_pin = pin.getFrameJacobian(
                self.model, self.data, frame_id=self.tcp_links_id[part_name], reference_frame=pin.LOCAL_WORLD_ALIGNED
            )

        return jaco_pin[:, self.part_joints_id[joint_part_name]]

    def getFrameGlobalJacobian(self, frame_name, joint_part_name=None):
        jaco_pin = pin.getFrameJacobian(
            self.model, self.data, frame_id=self.model.getFrameId(frame_name), reference_frame=pin.LOCAL_WORLD_ALIGNED
        )
        return jaco_pin[:, self.part_joints_id[joint_part_name]]

    def calcFingerPoseError(self, finger_name, finger_target_pose, finger_joint_pos, local_position):
        err = np.zeros((6,))
        finger_target_pos = finger_target_pose[0:3]
        finger_target_ori = sciR.from_quat(finger_target_pose[3:7])

        self.updateFK(finger_name, finger_joint_pos)
        finger_pos, finger_quat = self.getTcpGlobalPose(finger_name, local_position=local_position)
        err[0:3] = finger_pos - finger_target_pos
        err[3:6] = (sciR.from_quat(finger_quat) * finger_target_ori.inv()).as_rotvec()

        err = err.reshape(-1, 1)
        self.curr_err = err

        return err

    def fingerIKSQP(self, finger_name, finger_target_pose, weights, finger_joint_pos_init, local_position=None):  # noqa: PLR0913
        t_ik = perf_counter()

        finger_target_pos, finger_target_ori = ucalc.isometry3dToPosOri(finger_target_pose)
        self.curr_err = None

        def calcTargetError(finger_joint_pos):
            t1 = perf_counter()
            err = np.zeros((6,))

            self.updateFK(finger_name, finger_joint_pos)
            finger_pos, finger_quat = self.getTcpGlobalPose(finger_name, local_position=local_position)
            err[0:3] = finger_pos - finger_target_pos
            err[3:6] = (sciR.from_quat(finger_quat) * finger_target_ori.inv()).as_rotvec()

            err = err.reshape(-1, 1)
            self.curr_err = err

            return err

        def objectFunction(finger_joint_pos):
            err = calcTargetError(finger_joint_pos)
            cost = 1.0 / 2.0 * err.T @ weights @ err
            return cost[0, 0]

        def objectJacobian(finger_joint_pos):
            t1 = perf_counter()
            err = self.curr_err.copy()

            # get all jacobians
            jaco = self.getGlobalJacobian(
                finger_name, finger_joint_pos, local_position=local_position, joint_part_name=finger_name
            )

            object_jaco = err.T @ weights @ jaco

            return object_jaco.reshape(-1)

        joint_pos_init = finger_joint_pos_init.copy()
        # bounds
        joint_pos_lb = np.array(self.joints_id_to_lower_limit)[self.part_joints_id[finger_name]]
        joint_pos_ub = np.array(self.joints_id_to_upper_limit)[self.part_joints_id[finger_name]]
        joint_pos_bounds = [(joint_pos_lb[i], joint_pos_ub[i]) for i in range(joint_pos_lb.shape[0])]

        res = minimize(
            fun=objectFunction,
            jac=objectJacobian,
            x0=joint_pos_init,
            bounds=joint_pos_bounds,
            method="SLSQP",
            options={"ftol": 1e-8, "disp": False},
        )
        res_joint_pos = res.x.reshape(-1)

        return res_joint_pos

    def trajOptOurs(  # noqa: PLR0913, PLR0915
        self,
        T,
        delta_t,
        object_target_pose,
        thumb_target_rel_pose,
        finger0_target_rel_pose=None,
        finger1_target_rel_pose=None,
        finger2_target_rel_pose=None,
        weights_object_pose=None,
        weights_rel_pose=None,
        weights_joint_vel=None,
        object_pose_init=None,
        hand_joint_pos_init=[0] * 16,
        finger0_local_position=None,
        finger1_local_position=None,
        finger2_local_position=None,
        thumb_local_position=None,
        b_print_opt_details=False,
    ):
        """
        Our proposed in-grasp trajectory optimization approach.
        """
        hand_joint_pos_init = np.asarray(hand_joint_pos_init)
        object_target_pos, object_target_ori = ucalc.isometry3dToPosOri(object_target_pose)

        fingers_target_rel_pose = {}
        if thumb_target_rel_pose is not None:
            fingers_target_rel_pose["thumb"] = thumb_target_rel_pose
        if finger0_target_rel_pose is not None:
            fingers_target_rel_pose["finger0"] = finger0_target_rel_pose
        if finger1_target_rel_pose is not None:
            fingers_target_rel_pose["finger1"] = finger1_target_rel_pose
        if finger2_target_rel_pose is not None:
            fingers_target_rel_pose["finger2"] = finger2_target_rel_pose
        fingers_target_rel_pos_ori = {}
        for finger_name, target_rel_pose in fingers_target_rel_pose.items():
            fingers_target_rel_pos_ori[finger_name] = ucalc.isometry3dToPosOri(target_rel_pose)

        fingers_local_position = {}
        fingers_local_position["finger0"] = finger0_local_position
        fingers_local_position["finger1"] = finger1_local_position
        fingers_local_position["finger2"] = finger2_local_position
        fingers_local_position["thumb"] = thumb_local_position

        # init value
        object_pos_init, object_rotvec_init = ucalc.isometry3dToPosRotVec(object_pose_init)
        val_init = np.concatenate([object_pos_init, object_rotvec_init, hand_joint_pos_init], axis=0)
        traj_val_init = np.tile(val_init, T + 1)
        # bounds
        joint_pos_lb = np.array(self.joints_id_to_lower_limit)[self.part_joints_id["hand"]]
        joint_pos_ub = np.array(self.joints_id_to_upper_limit)[self.part_joints_id["hand"]]
        val_bounds = [(-10, 10) for _ in range(6)] + [
            (joint_pos_lb[i], joint_pos_ub[i]) for i in range(joint_pos_lb.shape[0])
        ]
        traj_val_bounds = [val_bound for _ in range(T + 1) for val_bound in val_bounds]

        weights = np.diag(
            weights_object_pose
            + weights_rel_pose * (T * len(fingers_target_rel_pose))
            + [weights_joint_vel] * (T * len(self.part_joints_id["hand"]))
        )
        self.curr_err = None

        def calcTargetError(val):
            t1 = perf_counter()
            val = val.reshape(T + 1, -1)
            traj_object_pos, traj_object_rotvec, traj_hand_joint_pos = val[:, 0:3], val[:, 3:6], val[:, 6 : 6 + 16]

            err_list = []
            # ------------ object pose in world frame at {T} -------------
            object_pos, object_rotvec = traj_object_pos[-1, :], traj_object_rotvec[-1, :]
            err_list.extend(object_pos - object_target_pos)
            err_list.extend((sciR.from_rotvec(object_rotvec) * object_target_ori.inv()).as_rotvec())

            # ------------ finger pose in object frame at {t = 1,...T} ------------
            fingers_pos = {}
            fingers_quat = {}
            for finger_name in fingers_target_rel_pose.keys():
                fingers_pos[finger_name] = np.zeros((T, 3))
                fingers_quat[finger_name] = np.zeros((T, 4))

            for t in range(1, T + 1):
                self.updateFK("hand", traj_hand_joint_pos[t, :])
                for finger_name in fingers_target_rel_pose.keys():
                    fingers_pos[finger_name][t - 1, :], fingers_quat[finger_name][t - 1, :] = self.getTcpGlobalPose(
                        finger_name, local_position=fingers_local_position[finger_name]
                    )

            object_pose_inv = ucalc.batchIsometry3dInverse(
                ucalc.batchPosRotVec2Isometry3d(traj_object_pos[1:, :], traj_object_rotvec[1:, :])
            )

            for finger_name, (finger_target_rel_pos, finger_target_rel_ori) in fingers_target_rel_pos_ori.items():
                finger_pos, finger_quat = fingers_pos[finger_name], fingers_quat[finger_name]
                finger_pose = ucalc.batchPosQuat2Isometry3d(finger_pos, finger_quat)
                finger_rel_pose = np.matmul(object_pose_inv, finger_pose)
                finger_rel_pos_err = finger_rel_pose[:, 0:3, 3] - np.tile(finger_target_rel_pos, (T, 1))
                finger_rel_ori_err = (
                    sciR.from_matrix(finger_rel_pose[:, 0:3, 0:3]) * finger_target_rel_ori.inv()
                ).as_rotvec()
                finger_rel_pose_err = np.hstack([finger_rel_pos_err, finger_rel_ori_err])
                err_list.extend(finger_rel_pose_err.reshape(-1))

            # ------------ joint vel between t and t-1 (t = 1,...,T) ------------
            traj_hand_joint_vel = (traj_hand_joint_pos[1:, :] - traj_hand_joint_pos[0:-1, :]) / delta_t
            err_list.extend(traj_hand_joint_vel.reshape(-1))

            err = np.asarray(err_list).reshape(-1, 1)
            self.curr_err = err
            return err

        def objectFunction(val):
            err = calcTargetError(val)
            cost = 1.0 / 2.0 * err.T @ weights @ err
            return cost[0, 0]

        def objectJacobian(val):
            val = val.reshape(T + 1, -1)
            traj_object_pos, traj_object_rotvec, traj_hand_joint_pos = val[:, 0:3], val[:, 3:6], val[:, 6 : 6 + 16]

            err = self.curr_err  # to avoid double computing the error, slightly reducing optimization time cost
            jaco_list = []

            # ------------ object pose in world frame at {T} -------------
            whole_jaco = np.zeros((6, T + 1, 6 + 16))
            whole_jaco[0:3, -1, 0:3] = np.eye(3)

            object_rotvec = traj_object_rotvec[-1, :]
            object_ori_err = (sciR.from_rotvec(object_rotvec) * object_target_ori.inv()).as_rotvec()
            whole_jaco[3:6, -1, 3:6] = ucalc.jacoLeftBCHInverse(object_ori_err) @ ucalc.jacoLeftBCH(object_rotvec)
            jaco_list.append(whole_jaco.reshape(6, -1))

            # ------------ finger pose in object frame at {t = 1,...T} ------------
            fingers_pos = {}
            fingers_quat = {}
            fingers_jaco = {}
            for finger_name in fingers_target_rel_pose.keys():
                fingers_pos[finger_name] = np.zeros((T, 3))
                fingers_quat[finger_name] = np.zeros((T, 4))
                fingers_jaco[finger_name] = np.zeros((T, 6, 4))

            for t in range(1, T + 1):
                self.updateJacobians("hand", traj_hand_joint_pos[t, :])  # will call FK internally
                for finger_name in fingers_target_rel_pose.keys():
                    fingers_pos[finger_name][t - 1, :], fingers_quat[finger_name][t - 1, :] = self.getTcpGlobalPose(
                        finger_name, local_position=fingers_local_position[finger_name]
                    )
                    fingers_jaco[finger_name][t - 1, :, :] = self.getGlobalJacobian(
                        finger_name, local_position=fingers_local_position[finger_name], joint_part_name=finger_name
                    )

            object_pose_inv = ucalc.batchIsometry3dInverse(
                ucalc.batchPosRotVec2Isometry3d(traj_object_pos[1:, :], traj_object_rotvec[1:, :])
            )
            object_jaco = np.zeros((T, 6, 6))  # T = 1, ..., T
            object_jaco[:, 0:3, 0:3] = np.tile(np.eye(3), (T, 1, 1))
            object_jaco[:, 3:6, 3:6] = ucalc.jacoLeftBCH(traj_object_rotvec[1:, :])
            transform_mat = ucalc.batchDiagRotMat(object_pose_inv[:, 0:3, 0:3])
            transformed_object_jaco = np.matmul(transform_mat, object_jaco)

            for finger_name, (finger_target_rel_pos, finger_target_rel_ori) in fingers_target_rel_pos_ori.items():
                finger_pose = ucalc.batchPosQuat2Isometry3d(fingers_pos[finger_name], fingers_quat[finger_name])
                finger_rel_pose = np.matmul(object_pose_inv, finger_pose)
                finger_rel_pos = finger_rel_pose[:, 0:3, 3]
                finger_rel_ori_err = (
                    sciR.from_matrix(finger_rel_pose[:, 0:3, 0:3]) * finger_target_rel_ori.inv()
                ).as_rotvec()

                temp_jaco = np.zeros((T, 6, 6 + 16))  # the variable only include the joint positions at time t
                temp_jaco[:, :, 0:6] = -np.matmul(
                    ucalc.wrenchTransformationMatrix(finger_rel_pos), transformed_object_jaco
                )
                temp_jaco[:, :, 6 + np.asarray(self.finger_joints_id_in_hand[finger_name])] = np.matmul(
                    transform_mat, fingers_jaco[finger_name]
                )
                # partial error / partial phi (spatial twist)
                temp_jaco[:, 3:6, :] = np.matmul(ucalc.jacoLeftBCHInverse(finger_rel_ori_err), temp_jaco[:, 3:6, :])

                whole_jaco = np.zeros((T, 6, T + 1, 6 + 16))
                for t in range(T):
                    whole_jaco[t, :, t + 1, :] = temp_jaco[t]

                jaco_list.append(whole_jaco.reshape(T * 6, -1))

            # ------------ joint vel between t and t-1 (t = 1,...,T) ------------
            whole_jaco = np.zeros((T, 16, T + 1, 6 + 16))
            for t in range(1, T + 1):
                whole_jaco[t - 1, :, t, 6:] = 1.0 / delta_t * np.eye(16)
                whole_jaco[t - 1, :, t - 1, 6:] = -1.0 / delta_t * np.eye(16)
            jaco_list.append(whole_jaco.reshape(T * 16, -1))

            jaco = np.vstack(jaco_list)
            object_jaco = err.T @ weights @ jaco

            return object_jaco.reshape(-1)

        def x0EqConstraint(val):
            val = val.reshape(T + 1, -1)
            eq = val[0, :] - val_init  # = 0
            return eq

        def x0EqConstraintJaco(val):
            jaco = np.zeros((22, val.size))
            jaco[:, 0:22] = np.eye(22)
            return jaco

        def collisionConstraint(val):
            val = val.reshape(T + 1, -1)
            traj_hand_joint_pos = val[:, 6 : 6 + 16]

            traj_constraints = np.zeros((T, 4))
            for t in range(1, T + 1):
                self.updateFK("hand", traj_hand_joint_pos[t, :])
                critical_frame_0_0_pos, _ = self.getFrameGlobalPose(frame_name="critical_link_0_0")
                critical_frame_1_0_pos, _ = self.getFrameGlobalPose(frame_name="critical_link_1_0")
                critical_frame_0_1_pos, _ = self.getFrameGlobalPose(frame_name="critical_link_0_1")
                critical_frame_1_1_pos, _ = self.getFrameGlobalPose(frame_name="critical_link_1_1")
                dist_0 = np.linalg.norm(critical_frame_0_0_pos - critical_frame_1_0_pos)
                dist_1 = np.linalg.norm(critical_frame_0_0_pos - critical_frame_1_1_pos)
                dist_2 = np.linalg.norm(critical_frame_0_1_pos - critical_frame_1_0_pos)
                dist_3 = np.linalg.norm(critical_frame_0_1_pos - critical_frame_1_1_pos)

                traj_constraints[t - 1, 0] = dist_0 - 0.04  # >= 0
                traj_constraints[t - 1, 1] = dist_1 - 0.04
                traj_constraints[t - 1, 2] = dist_2 - 0.04
                traj_constraints[t - 1, 3] = dist_3 - 0.04

            return traj_constraints.reshape(-1)  # >= 0

        def collisionConstraintJaco(val):
            val = val.reshape(T + 1, -1)
            traj_hand_joint_pos = val[:, 6 : 6 + 16]

            whole_jaco = np.zeros((T, 4, T + 1, 6 + 16))
            for t in range(1, T + 1):
                self.updateJacobians("hand", traj_hand_joint_pos[t, :])  # will call FK internally
                critical_frame_0_0_pos, _ = self.getFrameGlobalPose(frame_name="critical_link_0_0")
                critical_frame_1_0_pos, _ = self.getFrameGlobalPose(frame_name="critical_link_1_0")
                critical_frame_0_1_pos, _ = self.getFrameGlobalPose(frame_name="critical_link_0_1")
                critical_frame_1_1_pos, _ = self.getFrameGlobalPose(frame_name="critical_link_1_1")

                critical_frame_0_0_jaco = self.getFrameGlobalJacobian("critical_link_0_0", joint_part_name="hand")
                critical_frame_1_0_jaco = self.getFrameGlobalJacobian("critical_link_1_0", joint_part_name="hand")
                critical_frame_0_1_jaco = self.getFrameGlobalJacobian("critical_link_0_1", joint_part_name="hand")
                critical_frame_1_1_jaco = self.getFrameGlobalJacobian("critical_link_1_1", joint_part_name="hand")

                diff = critical_frame_0_0_pos - critical_frame_1_0_pos
                whole_jaco[t - 1, 0, t, 6:] = (diff / np.linalg.norm(diff)).reshape(1, -1) @ (
                    critical_frame_0_0_jaco[0:3, :] - critical_frame_1_0_jaco[0:3, :]
                )
                diff = critical_frame_0_0_pos - critical_frame_1_1_pos
                whole_jaco[t - 1, 1, t, 6:] = (diff / np.linalg.norm(diff)).reshape(1, -1) @ (
                    critical_frame_0_0_jaco[0:3, :] - critical_frame_1_1_jaco[0:3, :]
                )
                diff = critical_frame_0_1_pos - critical_frame_1_0_pos
                whole_jaco[t - 1, 2, t, 6:] = (diff / np.linalg.norm(diff)).reshape(1, -1) @ (
                    critical_frame_0_1_jaco[0:3, :] - critical_frame_1_0_jaco[0:3, :]
                )
                diff = critical_frame_0_1_pos - critical_frame_1_1_pos
                whole_jaco[t - 1, 3, t, 6:] = (diff / np.linalg.norm(diff)).reshape(1, -1) @ (
                    critical_frame_0_1_jaco[0:3, :] - critical_frame_1_1_jaco[0:3, :]
                )

            return whole_jaco.reshape(T * 4, (T + 1) * (6 + 16))

        constraints_list = [
            dict(type="eq", fun=x0EqConstraint, jac=x0EqConstraintJaco),
            dict(type="ineq", fun=collisionConstraint, jac=collisionConstraintJaco),
        ]

        res = minimize(
            fun=objectFunction,
            jac=objectJacobian,
            constraints=constraints_list,
            x0=traj_val_init,
            bounds=traj_val_bounds,
            method="SLSQP",
            options={"ftol": 1e-10, "disp": b_print_opt_details, "maxiter": 1000},
        )
        res_val = res.x.reshape(-1)

        traj_object_pos = res_val.reshape(T + 1, -1)[:, 0:3]
        planned_object_err = np.linalg.norm(traj_object_pos[-1] - object_target_pos)
        traj_hand_joint_pos = res_val.reshape(T + 1, -1)[:, 6:]
        return traj_hand_joint_pos, planned_object_err