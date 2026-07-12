#!/usr/bin/env python3
"""
CAD/URDF-aware LQR dataset collector for the MuJoCo self-balancing robot.

This collector is updated for the current CAD-based model convention:

- wheel hinge axes are +/-Y
- forward/backward balancing pitch is Euler-Y
- forward translation is approximately body/world X
- wheel radius is read from l_wheel_geom / r_wheel_geom
- initial free-joint body z is computed from floor height, wheel radius, and wheel body z
- logging is one row per LQR control update by default

Typical usage from repository root:

    python scripts/collect_lqr_dataset.py \
        --model src/simulation/scene.xml \
        --out datasets/lqr_cad_dataset.csv \
        --episodes 50 \
        --duration 10 \
        --control-hz 200 \
        --cmd-mode random_step \
        --max-speed 0.10 \
        --max-yaw 0.40

For a quiet balancing-only dataset:

    python scripts/collect_lqr_dataset.py --cmd-mode zero --episodes 20
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Import robot_lqr.py robustly whether this script lives in scripts/ or
# src/simulation/.
# ---------------------------------------------------------------------------

_THIS_FILE = pathlib.Path(__file__).resolve()
_CANDIDATE_IMPORT_DIRS = [
    _THIS_FILE.parent,
    _THIS_FILE.parent.parent,
    _THIS_FILE.parent.parent / "src" / "simulation",
    pathlib.Path.cwd(),
    pathlib.Path.cwd() / "src" / "simulation",
]
for _p in _CANDIDATE_IMPORT_DIRS:
    if _p.exists():
        sys.path.insert(0, str(_p))

try:
    from robot_lqr import LQR_K, MAX_MOTOR_VEL, WHEEL_RADIUS, RobotLqr, clamp
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Could not import robot_lqr.py. Run from the repository root or place this "
        "file where robot_lqr.py is importable."
    ) from exc


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SensorNoiseConfig:
    """Simple pseudo-sensor noise model for EKF/MLP dataset creation."""

    gyro_bias_std: float = 0.015          # rad/s, episode-wise gyro bias
    gyro_noise_std: float = 0.010         # rad/s, per-sample gyro noise
    encoder_pos_noise_std: float = 0.001  # rad
    encoder_vel_noise_std: float = 0.020  # rad/s
    accel_noise_std: float = 0.10         # m/s^2
    vision_pos_noise_std: float = 0.005   # m
    vision_yaw_noise_std: float = 0.010   # rad


@dataclass
class EpisodeRandomizationConfig:
    """Initial-state randomization around a stable balancing pose."""

    init_pitch_max_rad: float = math.radians(8.0)   # Euler-Y pitch
    init_roll_max_rad: float = math.radians(0.0)    # keep 0 unless explicitly needed
    init_yaw_max_rad: float = math.pi
    init_forward_vel_max_m_s: float = 0.20          # body-X forward velocity
    init_pitch_rate_max_rad_s: float = 0.50         # qvel angular Y


@dataclass
class ModelGeometry:
    floor_z: float
    wheel_radius: float
    wheel_body_z: float
    initial_body_z: float


# ---------------------------------------------------------------------------
# Path/model helpers
# ---------------------------------------------------------------------------

def resolve_model_path(model_arg: str | None) -> pathlib.Path:
    if model_arg:
        path = pathlib.Path(model_arg).expanduser()
        if not path.is_absolute():
            path = pathlib.Path.cwd() / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        return path

    candidates = [
        pathlib.Path.cwd() / "src" / "simulation" / "scene.xml",
        pathlib.Path.cwd() / "scene.xml",
        _THIS_FILE.parent / "scene.xml",
        _THIS_FILE.parent.parent / "scene.xml",
        _THIS_FILE.parent.parent / "src" / "simulation" / "scene.xml",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()

    raise FileNotFoundError("Could not find scene.xml automatically. Pass --model path/to/scene.xml")


def name2id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return idx


def infer_model_geometry(
    model: mujoco.MjModel,
    floor_geom: str = "floor",
    left_wheel_geom: str = "l_wheel_geom",
    right_wheel_geom: str = "r_wheel_geom",
    left_wheel_body: str = "l_wheel",
    right_wheel_body: str = "r_wheel",
) -> ModelGeometry:
    """Infer floor height, wheel radius, and stable initial body z.

    For the current CAD model:
        body_z = floor_z + wheel_radius - wheel_body_z

    because wheel center world z = body_z + wheel_body_z.
    """
    floor_id = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_geom)
    floor_z = float(model.geom_pos[floor_id, 2])

    lgid = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, left_wheel_geom)
    rgid = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, right_wheel_geom)
    wheel_radius = float(0.5 * (model.geom_size[lgid, 0] + model.geom_size[rgid, 0]))

    lbid = name2id(model, mujoco.mjtObj.mjOBJ_BODY, left_wheel_body)
    rbid = name2id(model, mujoco.mjtObj.mjOBJ_BODY, right_wheel_body)
    wheel_body_z = float(0.5 * (model.body_pos[lbid, 2] + model.body_pos[rbid, 2]))

    initial_body_z = floor_z + wheel_radius - wheel_body_z

    return ModelGeometry(
        floor_z=floor_z,
        wheel_radius=wheel_radius,
        wheel_body_z=wheel_body_z,
        initial_body_z=initial_body_z,
    )


# ---------------------------------------------------------------------------
# Rotation/kinematics helpers
# ---------------------------------------------------------------------------

def quat_wxyz_to_euler_xyz(quat_wxyz: np.ndarray) -> Tuple[float, float, float]:
    """MuJoCo quaternion [w, x, y, z] -> scipy Euler xyz [roll, pitch, yaw]."""
    q = np.asarray(quat_wxyz, dtype=float)
    rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
    roll, pitch, yaw = rot.as_euler("xyz", degrees=False)
    return float(roll), float(pitch), float(yaw)


def euler_xyz_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """scipy Euler xyz -> MuJoCo quaternion [w, x, y, z]."""
    q_xyzw = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)


def reset_lqr_internal_state(robot: RobotLqr) -> None:
    robot.velocity_angular = 0.0
    robot.velocity_linear_set_point = 0.0
    robot.yaw = 0.0
    robot.pitch_dot_filtered = 0.0
    robot.velocity_angular_filtered = 0.0


def reset_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot: RobotLqr,
    rng: np.random.Generator,
    geom: ModelGeometry,
    randomize: EpisodeRandomizationConfig,
) -> None:
    """Reset MuJoCo state with CAD-coordinate-safe initial conditions."""
    mujoco.mj_resetData(model, data)
    reset_lqr_internal_state(robot)

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0

    # Free joint position.
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = geom.initial_body_z

    # CAD convention: roll=X, pitch=Y, yaw=Z.
    init_roll = rng.uniform(-randomize.init_roll_max_rad, randomize.init_roll_max_rad)
    init_pitch = rng.uniform(-randomize.init_pitch_max_rad, randomize.init_pitch_max_rad)
    init_yaw = rng.uniform(-randomize.init_yaw_max_rad, randomize.init_yaw_max_rad)
    data.qpos[3:7] = euler_xyz_to_quat_wxyz(init_roll, init_pitch, init_yaw)

    # Wheel joint positions.
    if data.qpos.size >= 9:
        data.qpos[7] = 0.0
        data.qpos[8] = 0.0

    # Initial body-X forward velocity, rotated into world coordinates.
    v_forward_body = rng.uniform(
        -randomize.init_forward_vel_max_m_s,
        randomize.init_forward_vel_max_m_s,
    )
    rot = Rotation.from_euler("xyz", [init_roll, init_pitch, init_yaw], degrees=False)
    v_world = rot.apply([v_forward_body, 0.0, 0.0])
    data.qvel[:3] = v_world

    # Free-joint qvel is [vx, vy, vz, wx, wy, wz]. CAD pitch rate is wy.
    data.qvel[4] = rng.uniform(
        -randomize.init_pitch_rate_max_rad_s,
        randomize.init_pitch_rate_max_rad_s,
    )

    mujoco.mj_forward(model, data)


def get_kinematics(data: mujoco.MjData) -> Dict[str, float | np.ndarray]:
    quat = np.asarray(data.body("robot_body").xquat, dtype=float).copy()
    roll, pitch, yaw = quat_wxyz_to_euler_xyz(quat)

    body_pos = np.asarray(data.body("robot_body").xpos, dtype=float).copy()
    free_qvel = np.asarray(data.joint("robot_body_joint").qvel, dtype=float).copy()
    linear_vel_world = free_qvel[:3].copy()
    angular_vel = free_qvel[-3:].copy()

    rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    linear_vel_body = rot.inv().apply(linear_vel_world)

    wheel_pos_l = float(data.joint("torso_l_wheel").qpos[0])
    wheel_pos_r = float(data.joint("torso_r_wheel").qpos[0])
    wheel_vel_l = float(data.joint("torso_l_wheel").qvel[0])
    wheel_vel_r = float(data.joint("torso_r_wheel").qvel[0])

    # Match RobotLqr.get_wheel_velocity().
    wheel_vel_avg = (-wheel_vel_l + wheel_vel_r) / 2.0
    encoder_forward_vel = wheel_vel_avg * WHEEL_RADIUS

    return {
        "body_pos": body_pos,
        "quat_wxyz": quat,
        "roll_rad": float(roll),
        "pitch_rad": float(pitch),       # CAD balancing pitch
        "yaw_rad": float(yaw),
        "linear_vel_world": linear_vel_world,
        "linear_vel_body": linear_vel_body,
        "angular_vel": angular_vel,
        "wheel_pos_l_rad": wheel_pos_l,
        "wheel_pos_r_rad": wheel_pos_r,
        "wheel_vel_l_rad_s": wheel_vel_l,
        "wheel_vel_r_rad_s": wheel_vel_r,
        "wheel_vel_avg_rad_s": float(wheel_vel_avg),
        "encoder_forward_vel_m_s": float(encoder_forward_vel),
    }


# ---------------------------------------------------------------------------
# Control/command helpers
# ---------------------------------------------------------------------------

def compute_and_apply_lqr(
    robot: RobotLqr,
    speed_setpoint: float,
    yaw_setpoint: float,
) -> Dict[str, float | int]:
    """One LQR update while logging the internal terms.

    This reproduces robot_lqr.py's calculate_lqr_velocity/update_motor_speed
    logic without calling calculate_lqr_velocity() twice.
    """
    robot.set_velocity_linear_set_point(speed_setpoint)
    robot.set_yaw(yaw_setpoint)

    pitch_for_lqr = -robot.get_pitch()
    pitch_dot = robot.get_pitch_dot()

    robot.pitch_dot_filtered = robot.pitch_dot_filtered * 0.975 + pitch_dot * 0.025
    wheel_vel_avg = robot.get_wheel_velocity()
    robot.velocity_angular_filtered = (
        robot.velocity_angular_filtered * 0.975 + wheel_vel_avg * 0.025
    )

    velocity_linear_error = robot.velocity_linear_set_point - (
        robot.velocity_angular_filtered * WHEEL_RADIUS
    )
    lqr_v = (
        LQR_K[0] * (0.0 - pitch_for_lqr)
        + LQR_K[1] * robot.pitch_dot_filtered
        + LQR_K[2] * 0.0
        + LQR_K[3] * velocity_linear_error
    )

    motor_velocity_raw = -lqr_v / WHEEL_RADIUS
    motor_velocity_cmd = clamp(motor_velocity_raw, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)

    ctrl_l = -motor_velocity_cmd + robot.yaw
    ctrl_r = motor_velocity_cmd + robot.yaw
    robot.data.actuator("motor_l_wheel").ctrl = [ctrl_l]
    robot.data.actuator("motor_r_wheel").ctrl = [ctrl_r]

    return {
        "lqr_pitch_rad": float(robot.get_pitch()),
        "pitch_for_lqr_rad": float(pitch_for_lqr),
        "pitch_dot_raw_rad_s": float(pitch_dot),
        "pitch_dot_filtered_rad_s": float(robot.pitch_dot_filtered),
        "wheel_vel_avg_raw_rad_s": float(wheel_vel_avg),
        "wheel_vel_avg_filtered_rad_s": float(robot.velocity_angular_filtered),
        "velocity_linear_error_m_s": float(velocity_linear_error),
        "lqr_motor_vel_raw_rad_s": float(motor_velocity_raw),
        "lqr_motor_vel_cmd_rad_s": float(motor_velocity_cmd),
        "lqr_saturated": int(abs(motor_velocity_raw) > MAX_MOTOR_VEL),
        "ctrl_l_rad_s": float(ctrl_l),
        "ctrl_r_rad_s": float(ctrl_r),
    }


def sample_commands(
    mode: str,
    t: float,
    rng: np.random.Generator,
    current_speed: float,
    current_yaw: float,
    next_change_t: float,
    max_speed: float,
    max_yaw: float,
    change_interval: float,
) -> Tuple[float, float, float]:
    """Return speed setpoint, yaw setpoint, next command change time."""
    if mode == "zero":
        return 0.0, 0.0, math.inf

    if mode == "sine":
        speed = 0.5 * max_speed * math.sin(2.0 * math.pi * 0.20 * t)
        yaw = 0.5 * max_yaw * math.sin(2.0 * math.pi * 0.10 * t)
        return speed, yaw, math.inf

    if mode == "random_step":
        if t >= next_change_t:
            current_speed = float(rng.uniform(-max_speed, max_speed))
            current_yaw = float(rng.uniform(-max_yaw, max_yaw))
            next_change_t = t + change_interval
        return current_speed, current_yaw, next_change_t

    raise ValueError(f"Unknown cmd mode: {mode}")


# ---------------------------------------------------------------------------
# Sensor model / row construction
# ---------------------------------------------------------------------------

def make_sensor_measurements(
    kin: Dict[str, float | np.ndarray],
    prev_linear_vel_world: np.ndarray,
    dt: float,
    gyro_bias: np.ndarray,
    noise: SensorNoiseConfig,
    rng: np.random.Generator,
) -> Dict[str, float]:
    quat = np.asarray(kin["quat_wxyz"], dtype=float)
    rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])

    linear_vel_world = np.asarray(kin["linear_vel_world"], dtype=float)
    angular_vel = np.asarray(kin["angular_vel"], dtype=float)

    # Accelerometer specific force: R^T * (a_world - g_world).
    acc_world = (linear_vel_world - prev_linear_vel_world) / max(dt, 1e-9)
    gravity_world = np.array([0.0, 0.0, -9.81])
    specific_force_body = rot.inv().apply(acc_world - gravity_world)
    accel_meas = specific_force_body + rng.normal(0.0, noise.accel_noise_std, size=3)

    gyro_meas = angular_vel + gyro_bias + rng.normal(0.0, noise.gyro_noise_std, size=3)

    wheel_pos_l_meas = float(kin["wheel_pos_l_rad"]) + rng.normal(0.0, noise.encoder_pos_noise_std)
    wheel_pos_r_meas = float(kin["wheel_pos_r_rad"]) + rng.normal(0.0, noise.encoder_pos_noise_std)
    wheel_vel_l_meas = float(kin["wheel_vel_l_rad_s"]) + rng.normal(0.0, noise.encoder_vel_noise_std)
    wheel_vel_r_meas = float(kin["wheel_vel_r_rad_s"]) + rng.normal(0.0, noise.encoder_vel_noise_std)
    encoder_forward_vel_meas = ((-wheel_vel_l_meas + wheel_vel_r_meas) / 2.0) * WHEEL_RADIUS

    body_pos = np.asarray(kin["body_pos"], dtype=float)
    vision_pos = body_pos + rng.normal(0.0, noise.vision_pos_noise_std, size=3)
    vision_yaw = float(kin["yaw_rad"]) + rng.normal(0.0, noise.vision_yaw_noise_std)

    # CAD model forward direction is body-X.
    true_forward_vel_body_x = float(np.asarray(kin["linear_vel_body"], dtype=float)[0])

    return {
        "gyro_x_meas_rad_s": float(gyro_meas[0]),
        "gyro_y_meas_rad_s": float(gyro_meas[1]),
        "gyro_z_meas_rad_s": float(gyro_meas[2]),
        "gyro_bias_x_rad_s": float(gyro_bias[0]),
        "gyro_bias_y_rad_s": float(gyro_bias[1]),
        "gyro_bias_z_rad_s": float(gyro_bias[2]),
        "acc_x_meas_m_s2": float(accel_meas[0]),
        "acc_y_meas_m_s2": float(accel_meas[1]),
        "acc_z_meas_m_s2": float(accel_meas[2]),
        "enc_l_pos_meas_rad": float(wheel_pos_l_meas),
        "enc_r_pos_meas_rad": float(wheel_pos_r_meas),
        "enc_l_vel_meas_rad_s": float(wheel_vel_l_meas),
        "enc_r_vel_meas_rad_s": float(wheel_vel_r_meas),
        "enc_forward_vel_meas_m_s": float(encoder_forward_vel_meas),
        "vision_x_meas_m": float(vision_pos[0]),
        "vision_y_meas_m": float(vision_pos[1]),
        "vision_z_meas_m": float(vision_pos[2]),
        "vision_yaw_meas_rad": float(vision_yaw),
        # Supervised residual labels for later EKF/MLP experiments.
        "target_encoder_forward_vel_error_m_s": float(true_forward_vel_body_x - encoder_forward_vel_meas),
        "target_gyro_x_error_rad_s": float(angular_vel[0] - gyro_meas[0]),
        "target_gyro_y_error_rad_s": float(angular_vel[1] - gyro_meas[1]),
        "target_gyro_z_error_rad_s": float(angular_vel[2] - gyro_meas[2]),
        "target_vision_x_error_m": float(body_pos[0] - vision_pos[0]),
        "target_vision_y_error_m": float(body_pos[1] - vision_pos[1]),
        "target_vision_z_error_m": float(body_pos[2] - vision_pos[2]),
        "target_vision_yaw_error_rad": float(float(kin["yaw_rad"]) - vision_yaw),
    }


def build_row(
    episode: int,
    control_step: int,
    physics_step: int,
    data: mujoco.MjData,
    control_dt: float,
    speed_setpoint: float,
    yaw_setpoint: float,
    lqr_info: Dict[str, float | int],
    kin: Dict[str, float | np.ndarray],
    sensor: Dict[str, float],
    is_fallen: int,
) -> Dict[str, float | int]:
    body_pos = np.asarray(kin["body_pos"], dtype=float)
    quat = np.asarray(kin["quat_wxyz"], dtype=float)
    v_world = np.asarray(kin["linear_vel_world"], dtype=float)
    v_body = np.asarray(kin["linear_vel_body"], dtype=float)
    w = np.asarray(kin["angular_vel"], dtype=float)

    row: Dict[str, float | int] = {
        "episode": episode,
        "control_step": control_step,
        "physics_step": physics_step,
        "time_s": float(data.time),
        "control_dt_s": float(control_dt),
        "speed_setpoint_m_s": float(speed_setpoint),
        "yaw_setpoint_rad_s": float(yaw_setpoint),
        "is_fallen": int(is_fallen),
        "x_m": float(body_pos[0]),
        "y_m": float(body_pos[1]),
        "z_m": float(body_pos[2]),
        "quat_w": float(quat[0]),
        "quat_x": float(quat[1]),
        "quat_y": float(quat[2]),
        "quat_z": float(quat[3]),
        "roll_rad": float(kin["roll_rad"]),
        "pitch_rad": float(kin["pitch_rad"]),
        "yaw_rad": float(kin["yaw_rad"]),
        "vx_world_m_s": float(v_world[0]),
        "vy_world_m_s": float(v_world[1]),
        "vz_world_m_s": float(v_world[2]),
        "vx_body_m_s": float(v_body[0]),
        "vy_body_m_s": float(v_body[1]),
        "vz_body_m_s": float(v_body[2]),
        "wx_rad_s": float(w[0]),
        "wy_rad_s": float(w[1]),
        "wz_rad_s": float(w[2]),
        "wheel_l_pos_rad": float(kin["wheel_pos_l_rad"]),
        "wheel_r_pos_rad": float(kin["wheel_pos_r_rad"]),
        "wheel_l_vel_rad_s": float(kin["wheel_vel_l_rad_s"]),
        "wheel_r_vel_rad_s": float(kin["wheel_vel_r_rad_s"]),
        "wheel_avg_vel_rad_s": float(kin["wheel_vel_avg_rad_s"]),
        "encoder_forward_vel_true_m_s": float(kin["encoder_forward_vel_m_s"]),
    }
    row.update(lqr_info)
    row.update(sensor)
    return row


FIELDNAMES: List[str] = [
    "episode",
    "control_step",
    "physics_step",
    "time_s",
    "control_dt_s",
    "speed_setpoint_m_s",
    "yaw_setpoint_rad_s",
    "is_fallen",
    "x_m",
    "y_m",
    "z_m",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
    "roll_rad",
    "pitch_rad",
    "yaw_rad",
    "vx_world_m_s",
    "vy_world_m_s",
    "vz_world_m_s",
    "vx_body_m_s",
    "vy_body_m_s",
    "vz_body_m_s",
    "wx_rad_s",
    "wy_rad_s",
    "wz_rad_s",
    "wheel_l_pos_rad",
    "wheel_r_pos_rad",
    "wheel_l_vel_rad_s",
    "wheel_r_vel_rad_s",
    "wheel_avg_vel_rad_s",
    "encoder_forward_vel_true_m_s",
    "lqr_pitch_rad",
    "pitch_for_lqr_rad",
    "pitch_dot_raw_rad_s",
    "pitch_dot_filtered_rad_s",
    "wheel_vel_avg_raw_rad_s",
    "wheel_vel_avg_filtered_rad_s",
    "velocity_linear_error_m_s",
    "lqr_motor_vel_raw_rad_s",
    "lqr_motor_vel_cmd_rad_s",
    "lqr_saturated",
    "ctrl_l_rad_s",
    "ctrl_r_rad_s",
    "gyro_x_meas_rad_s",
    "gyro_y_meas_rad_s",
    "gyro_z_meas_rad_s",
    "gyro_bias_x_rad_s",
    "gyro_bias_y_rad_s",
    "gyro_bias_z_rad_s",
    "acc_x_meas_m_s2",
    "acc_y_meas_m_s2",
    "acc_z_meas_m_s2",
    "enc_l_pos_meas_rad",
    "enc_r_pos_meas_rad",
    "enc_l_vel_meas_rad_s",
    "enc_r_vel_meas_rad_s",
    "enc_forward_vel_meas_m_s",
    "vision_x_meas_m",
    "vision_y_meas_m",
    "vision_z_meas_m",
    "vision_yaw_meas_rad",
    "target_encoder_forward_vel_error_m_s",
    "target_gyro_x_error_rad_s",
    "target_gyro_y_error_rad_s",
    "target_gyro_z_error_rad_s",
    "target_vision_x_error_m",
    "target_vision_y_error_m",
    "target_vision_z_error_m",
    "target_vision_yaw_error_rad",
]


def check_fallen(
    robot: RobotLqr,
    data: mujoco.MjData,
    fall_pitch_rad: float,
    min_z: float,
) -> bool:
    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
        return True
    if abs(robot.get_pitch()) > fall_pitch_rad:
        return True
    if float(data.body("robot_body").xpos[2]) < min_z:
        return True
    return False


def collect_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot: RobotLqr,
    episode: int,
    duration_s: float,
    control_hz: float,
    cmd_mode: str,
    max_speed: float,
    max_yaw: float,
    command_interval_s: float,
    rng: np.random.Generator,
    writer: csv.DictWriter,
    log_every_physics_step: bool,
    geom: ModelGeometry,
    randomize: EpisodeRandomizationConfig,
    fall_pitch_deg: float,
) -> Tuple[int, int, bool]:
    noise = SensorNoiseConfig()
    reset_episode(model, data, robot, rng, geom, randomize)

    physics_dt = float(model.opt.timestep)
    control_dt = 1.0 / control_hz
    substeps = max(1, int(round(control_dt / physics_dt)))
    effective_control_dt = substeps * physics_dt

    gyro_bias = rng.normal(0.0, noise.gyro_bias_std, size=3)

    speed_setpoint = 0.0
    yaw_setpoint = 0.0
    next_change_t = 0.0

    n_control_steps = int(math.ceil(duration_s / effective_control_dt))
    physics_step = 0
    rows_written = 0
    fell = False

    prev_kin = get_kinematics(data)
    prev_linear_vel_world = np.asarray(prev_kin["linear_vel_world"], dtype=float).copy()

    fall_pitch_rad = math.radians(fall_pitch_deg)
    min_z = geom.floor_z - 0.05

    for control_step in range(n_control_steps):
        speed_setpoint, yaw_setpoint, next_change_t = sample_commands(
            cmd_mode,
            float(data.time),
            rng,
            speed_setpoint,
            yaw_setpoint,
            next_change_t,
            max_speed,
            max_yaw,
            command_interval_s,
        )

        lqr_info = compute_and_apply_lqr(robot, speed_setpoint, yaw_setpoint)

        if log_every_physics_step:
            for _ in range(substeps):
                kin = get_kinematics(data)
                fell_now = check_fallen(robot, data, fall_pitch_rad, min_z)
                sensor = make_sensor_measurements(
                    kin,
                    prev_linear_vel_world,
                    physics_dt,
                    gyro_bias,
                    noise,
                    rng,
                )
                writer.writerow(
                    build_row(
                        episode,
                        control_step,
                        physics_step,
                        data,
                        physics_dt,
                        speed_setpoint,
                        yaw_setpoint,
                        lqr_info,
                        kin,
                        sensor,
                        int(fell_now),
                    )
                )
                rows_written += 1
                if fell_now:
                    fell = True
                    break

                prev_linear_vel_world = np.asarray(kin["linear_vel_world"], dtype=float).copy()
                mujoco.mj_step(model, data)
                physics_step += 1

            if fell:
                break
        else:
            kin = get_kinematics(data)
            fell_now = check_fallen(robot, data, fall_pitch_rad, min_z)
            sensor = make_sensor_measurements(
                kin,
                prev_linear_vel_world,
                effective_control_dt,
                gyro_bias,
                noise,
                rng,
            )
            writer.writerow(
                build_row(
                    episode,
                    control_step,
                    physics_step,
                    data,
                    effective_control_dt,
                    speed_setpoint,
                    yaw_setpoint,
                    lqr_info,
                    kin,
                    sensor,
                    int(fell_now),
                )
            )
            rows_written += 1
            if fell_now:
                fell = True
                break

            prev_linear_vel_world = np.asarray(kin["linear_vel_world"], dtype=float).copy()

            for _ in range(substeps):
                mujoco.mj_step(model, data)
                physics_step += 1

            if check_fallen(robot, data, fall_pitch_rad, min_z):
                fell = True
                break

    return rows_written, physics_step, fell


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect CAD-aware LQR MuJoCo balancing dataset as CSV.")
    parser.add_argument("--model", type=str, default=None, help="Path to scene.xml. Auto-detected if omitted.")
    parser.add_argument("--out", type=str, default="datasets/lqr_cad_dataset.csv", help="Output CSV path.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to collect.")
    parser.add_argument("--duration", type=float, default=10.0, help="Episode duration in seconds.")
    parser.add_argument("--control-hz", type=float, default=200.0, help="LQR/logging control rate in Hz.")

    parser.add_argument(
        "--cmd-mode",
        choices=["zero", "random_step", "sine"],
        default="random_step",
        help="Setpoint generation mode.",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=0.10,
        help="Max random/sine speed setpoint in m/s. Current controller may track about 2x this value.",
    )
    parser.add_argument(
        "--max-yaw",
        type=float,
        default=0.40,
        help="Max random/sine yaw motor offset. Keep small until yaw behavior is validated.",
    )
    parser.add_argument("--command-interval", type=float, default=2.0, help="Seconds between random command changes.")

    parser.add_argument("--init-pitch-deg", type=float, default=8.0, help="Max random initial Euler-Y pitch [deg].")
    parser.add_argument("--init-roll-deg", type=float, default=0.0, help="Max random initial Euler-X roll [deg].")
    parser.add_argument("--init-forward-vel", type=float, default=0.20, help="Max random initial body-X velocity [m/s].")
    parser.add_argument("--init-pitch-rate", type=float, default=0.50, help="Max random initial pitch rate wy [rad/s].")
    parser.add_argument("--fall-pitch-deg", type=float, default=60.0, help="Stop episode if abs(LQR pitch) exceeds this.")

    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--append", action="store_true", help="Append to existing CSV instead of overwriting.")
    parser.add_argument(
        "--log-every-physics-step",
        action="store_true",
        help="Log every MuJoCo physics step instead of every control step. Creates very large CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = resolve_model_path(args.model)
    out_path = pathlib.Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = pathlib.Path.cwd() / out_path
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    robot = RobotLqr(model, data)
    rng = np.random.default_rng(args.seed)

    geom = infer_model_geometry(model)

    randomize = EpisodeRandomizationConfig(
        init_pitch_max_rad=math.radians(args.init_pitch_deg),
        init_roll_max_rad=math.radians(args.init_roll_deg),
        init_yaw_max_rad=math.pi,
        init_forward_vel_max_m_s=args.init_forward_vel,
        init_pitch_rate_max_rad_s=args.init_pitch_rate,
    )

    print("=== CAD-aware LQR dataset collector ===")
    print(f"model                  : {model_path}")
    print(f"out                    : {out_path}")
    print(f"episodes               : {args.episodes}")
    print(f"duration               : {args.duration:.3f} s")
    print(f"control_hz             : {args.control_hz:.1f} Hz")
    print(f"cmd_mode               : {args.cmd_mode}")
    print(f"max_speed              : {args.max_speed:.3f} m/s")
    print(f"max_yaw                : {args.max_yaw:.3f}")
    print(f"floor_z                : {geom.floor_z:.4f} m")
    print(f"wheel_radius           : {geom.wheel_radius:.4f} m")
    print(f"wheel_body_z           : {geom.wheel_body_z:.4f} m")
    print(f"computed initial_body_z: {geom.initial_body_z:.4f} m")
    print(f"robot_lqr WHEEL_RADIUS : {WHEEL_RADIUS:.4f} m")
    print(f"LQR_K                  : {LQR_K}")
    print()

    if abs(geom.wheel_radius - WHEEL_RADIUS) > 1e-6:
        print(
            "[WARN] XML wheel radius and robot_lqr.WHEEL_RADIUS differ. "
            "Dataset will still collect, but velocity labels may be inconsistent."
        )

    mode = "a" if args.append else "w"
    need_header = (not args.append) or (not out_path.exists()) or out_path.stat().st_size == 0

    total_rows = 0
    total_physics_steps = 0
    fallen_episodes = 0

    with out_path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if need_header:
            writer.writeheader()

        for ep in range(args.episodes):
            rows, physics_steps, fell = collect_episode(
                model=model,
                data=data,
                robot=robot,
                episode=ep,
                duration_s=args.duration,
                control_hz=args.control_hz,
                cmd_mode=args.cmd_mode,
                max_speed=args.max_speed,
                max_yaw=args.max_yaw,
                command_interval_s=args.command_interval,
                rng=rng,
                writer=writer,
                log_every_physics_step=args.log_every_physics_step,
                geom=geom,
                randomize=randomize,
                fall_pitch_deg=args.fall_pitch_deg,
            )
            total_rows += rows
            total_physics_steps += physics_steps
            fallen_episodes += int(fell)

            status = "FELL" if fell else "OK"
            print(
                f"[episode {ep:04d}] {status} | rows={rows}, "
                f"physics_steps={physics_steps}, sim_time={data.time:.3f}s"
            )

    print()
    print(f"Saved: {out_path}")
    print(f"Total rows: {total_rows}")
    print(f"Total physics steps: {total_physics_steps}")
    print(f"Fallen episodes: {fallen_episodes}/{args.episodes}")
    print(f"Model: {model_path}")


if __name__ == "__main__":
    main()
