import math

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# obtained from running `scripts/lqr_gain_calculator.py`
LQR_K = np.array([-8.13602854295, -0.662641087853, -5.33513485128e-17, 2.2360679775], dtype=float)
WHEEL_RADIUS = 0.037
MAX_MOTOR_VEL = 500.0  # rad/s

# CAD/URDF-derived model convention:
# - wheel hinge axes are +/-Y
# - forward/backward body pitch is therefore rotation about Y, not X
PITCH_AXIS_INDEX = 1
PITCH_SIGN = 1.0


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


def quat_wxyz_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return MuJoCo free-joint quaternion order [w, x, y, z]."""
    quat_xyzw = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)


class RobotLqr:
    """
    Basic LQR controller implementation for the CAD/URDF-derived MuJoCo model.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data

        self.velocity_angular = 0.0
        self.velocity_linear_set_point = 0.0
        self.yaw = 0.0

        self.pitch_dot_filtered = 0.0
        self.velocity_angular_filtered = 0.0

    def set_velocity_linear_set_point(self, vel: float) -> None:
        self.velocity_linear_set_point = vel

    def set_yaw(self, yaw: float) -> None:
        self.yaw = yaw

    def get_pitch(self) -> float:
        quat = self.data.body("robot_body").xquat
        if quat[0] == 0:
            return 0.0

        rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy: [x, y, z, w]
        angles = rotation.as_euler("xyz", degrees=False)

        # CAD model: pitch is rotation about Y.
        return PITCH_SIGN * angles[PITCH_AXIS_INDEX]

    def get_pitch_dot(self) -> float:
        angular = self.data.joint("robot_body_joint").qvel[-3:]

        # Free-joint angular velocity components are treated as [wx, wy, wz].
        return PITCH_SIGN * angular[PITCH_AXIS_INDEX]

    def get_wheel_velocity(self) -> float:
        vel_l = self.data.joint("torso_l_wheel").qvel[0]
        vel_r = self.data.joint("torso_r_wheel").qvel[0]

        # Because the left and right wheel hinge axes are opposite signs.
        # Verify this sign convention with a motor sign test.
        return (-vel_l + vel_r) / 2.0

    def calculate_lqr_velocity(self) -> float:
        # Keep the original controller sign convention.
        # If the robot immediately drives itself into the fall direction,
        # first try PITCH_SIGN = -1.0, then check motor signs.
        pitch = -self.get_pitch()
        pitch_dot = self.get_pitch_dot()

        self.pitch_dot_filtered = (self.pitch_dot_filtered * 0.975) + (pitch_dot * 0.025)
        self.velocity_angular_filtered = (
            self.velocity_angular_filtered * 0.975
        ) + (self.get_wheel_velocity() * 0.025)

        velocity_linear_error = self.velocity_linear_set_point - self.velocity_angular_filtered * WHEEL_RADIUS

        lqr_v = (
            LQR_K[0] * (0.0 - pitch)
            + LQR_K[1] * self.pitch_dot_filtered
            + LQR_K[2] * 0.0
            + LQR_K[3] * velocity_linear_error
        )

        return -lqr_v / WHEEL_RADIUS

    def update_motor_speed(self) -> None:
        vel = self.calculate_lqr_velocity()
        vel = clamp(vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)

        # Verify this convention with a motor sign test.
        self.data.actuator("motor_l_wheel").ctrl = [-vel + self.yaw]
        self.data.actuator("motor_r_wheel").ctrl = [vel + self.yaw]

    def _default_body_z(self) -> float:
        """Place wheel collision cylinders near the floor plane."""
        try:
            floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            floor_z = float(self.model.geom_pos[floor_gid, 2]) if floor_gid >= 0 else -0.1
        except Exception:
            floor_z = -0.1

        try:
            lgid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_geom")
            rgid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_geom")

            radius = float(0.5 * (self.model.geom_size[lgid, 0] + self.model.geom_size[rgid, 0]))

            l_body_id = int(self.model.geom_bodyid[lgid])
            r_body_id = int(self.model.geom_bodyid[rgid])
            wheel_body_z = float(0.5 * (self.model.body_pos[l_body_id, 2] + self.model.body_pos[r_body_id, 2]))
            wheel_geom_z = float(0.5 * (self.model.geom_pos[lgid, 2] + self.model.geom_pos[rgid, 2]))

            return floor_z + radius - (wheel_body_z + wheel_geom_z)
        except Exception:
            # Current CAD model fallback: -0.1 + 0.037 - (-0.201) = 0.138
            return 0.138

    def reset(self):
        self.velocity_angular = 0.0
        self.velocity_linear_set_point = 0.0
        self.yaw = 0.0
        self.pitch_dot_filtered = 0.0
        self.velocity_angular_filtered = 0.0

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        # Free joint position: [x, y, z]
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = self._default_body_z()

        # For the CAD model:
        # roll  = X rotation, keep near 0
        # pitch = Y rotation, small perturbation
        # yaw   = Z rotation, random heading
        roll = 0.0
        pitch = (np.random.random() - 0.5) * 0.20
        yaw = (np.random.random() - 0.5) * 2.0 * math.pi

        self.data.qpos[3:7] = quat_wxyz_from_euler_xyz(roll, pitch, yaw)

        # Wheel joint positions if present.
        if self.data.qpos.size >= 9:
            self.data.qpos[7] = 0.0
            self.data.qpos[8] = 0.0

        mujoco.mj_forward(self.model, self.data)
