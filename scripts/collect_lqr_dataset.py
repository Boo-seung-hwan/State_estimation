#!/usr/bin/env python3
"""
Collect LQR balancing data from the MuJoCo self-balancing robot simulation.

This script turns a headless LQR balance check into a research dataset collector.
It saves one CSV row per control timestep by default. The MuJoCo physics timestep
in the provided scene is very small, so logging every physics step can create a
huge CSV; use --log-every-physics-step only when you really need that.

Example:
    python scripts/collect_lqr_dataset.py \
        --model src/simulation/scene.xml \
        --out datasets/lqr_dataset.csv \
        --episodes 50 \
        --duration 10 \
        --control-hz 200 \
        --cmd-mode random_step
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


# Make this file usable both from repository root and from scripts/.
_THIS_FILE = pathlib.Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1] if len(_THIS_FILE.parents) > 1 else pathlib.Path.cwd()
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
except Exception as exc:  # pragma: no cover - helpful runtime error for local repo differences
    raise RuntimeError(
        "Could not import robot_lqr.py. Put collect_lqr_dataset.py where it can find "
        "robot_lqr.py, or run it from the repository root."
    ) from exc


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
    init_lqr_pitch_max: float = math.radians(8.0)  # LQR pitch axis, rad
    init_yaw_max: float = math.pi                  # rad
    init_forward_vel_max: float = 0.20             # m/s, body y-axis approx.
    init_pitch_rate_max: float = 0.50              # rad/s


def resolve_model_path(model_arg: str | None) -> pathlib.Path:
    if model_arg is not None:
        path = pathlib.Path(model_arg).expanduser().resolve()
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
    raise FileNotFoundError(
        "Could not find scene.xml automatically. Pass --model path/to/scene.xml"
    )


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
    randomize: EpisodeRandomizationConfig,
) -> None:
    """Reset MuJoCo state with controlled random initial conditions.

    The existing RobotLqr.reset() randomizes attitude, but this collector sets the
    free-joint quaternion explicitly in MuJoCo's [w, x, y, z] order so the dataset
    initial conditions are reproducible and easy to audit.
    """
    mujoco.mj_resetData(model, data)
    reset_lqr_internal_state(robot)

    # In the current controller, RobotLqr.get_pitch() reads Euler x, so this
    # variable is named lqr_pitch even though Euler-x is conventionally roll.
    init_lqr_pitch = rng.uniform(-randomize.init_lqr_pitch_max, randomize.init_lqr_pitch_max)
    init_yaw = rng.uniform(-randomize.init_yaw_max, randomize.init_yaw_max)
    data.qpos[3:7] = euler_xyz_to_quat_wxyz(init_lqr_pitch, 0.0, init_yaw)

    # qvel for a free joint is [vx, vy, vz, wx, wy, wz]. The wheel axis is x,
    # so forward motion is approximately body/world y for small yaw.
    data.qvel[:] = 0.0
    data.qvel[1] = rng.uniform(-randomize.init_forward_vel_max, randomize.init_forward_vel_max)
    data.qvel[3] = rng.uniform(-randomize.init_pitch_rate_max, randomize.init_pitch_rate_max)

    data.actuator("motor_l_wheel").ctrl = [0.0]
    data.actuator("motor_r_wheel").ctrl = [0.0]
    mujoco.mj_forward(model, data)


def compute_and_apply_lqr(robot: RobotLqr, speed_setpoint: float, yaw_setpoint: float) -> Dict[str, float]:
    """One LQR control update, matching RobotLqr.update_motor_speed().

    We do not call calculate_lqr_velocity() and update_motor_speed() separately,
    because calculate_lqr_velocity() updates internal filters. Calling it twice
    would corrupt the logged data.
    """
    robot.set_velocity_linear_set_point(speed_setpoint)
    robot.set_yaw(yaw_setpoint)

    # This reproduces robot_lqr.py exactly.
    pitch_for_lqr = -robot.get_pitch()
    pitch_dot = robot.get_pitch_dot()

    robot.pitch_dot_filtered = robot.pitch_dot_filtered * 0.975 + pitch_dot * 0.025
    wheel_vel_avg = robot.get_wheel_velocity()
    robot.velocity_angular_filtered = robot.velocity_angular_filtered * 0.975 + wheel_vel_avg * 0.025

    velocity_linear_error = robot.velocity_linear_set_point - robot.velocity_angular_filtered * WHEEL_RADIUS
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
    wheel_vel_avg = (-wheel_vel_l + wheel_vel_r) / 2.0
    encoder_forward_vel = wheel_vel_avg * WHEEL_RADIUS

    return {
        "body_pos": body_pos,
        "quat_wxyz": quat,
        "roll_rad": float(roll),
        "pitch_rad": float(pitch),
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

    true_forward_vel_body_y = float(np.asarray(kin["linear_vel_body"], dtype=float)[1])

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
        # Useful supervised-learning labels for MLP residual/error estimation.
        "target_encoder_forward_vel_error_m_s": float(true_forward_vel_body_y - encoder_forward_vel_meas),
        "target_gyro_x_error_rad_s": float(angular_vel[0] - gyro_meas[0]),
        "target_vision_x_error_m": float(body_pos[0] - vision_pos[0]),
        "target_vision_y_error_m": float(body_pos[1] - vision_pos[1]),
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
    lqr_info: Dict[str, float],
    kin: Dict[str, float | np.ndarray],
    sensor: Dict[str, float],
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
    "target_vision_x_error_m",
    "target_vision_y_error_m",
    "target_vision_yaw_error_rad",
]


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
) -> Tuple[int, int]:
    noise = SensorNoiseConfig()
    randomize = EpisodeRandomizationConfig()
    reset_episode(model, data, robot, rng, randomize)

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

    prev_kin = get_kinematics(data)
    prev_linear_vel_world = np.asarray(prev_kin["linear_vel_world"], dtype=float).copy()

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
                sensor = make_sensor_measurements(
                    kin,
                    prev_linear_vel_world,
                    physics_dt,
                    gyro_bias,
                    noise,
                    rng,
                )
                row = build_row(
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
                )
                writer.writerow(row)
                rows_written += 1
                prev_linear_vel_world = np.asarray(kin["linear_vel_world"], dtype=float).copy()
                mujoco.mj_step(model, data)
                physics_step += 1
        else:
            kin = get_kinematics(data)
            sensor = make_sensor_measurements(
                kin,
                prev_linear_vel_world,
                effective_control_dt,
                gyro_bias,
                noise,
                rng,
            )
            row = build_row(
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
            )
            writer.writerow(row)
            rows_written += 1
            prev_linear_vel_world = np.asarray(kin["linear_vel_world"], dtype=float).copy()

            for _ in range(substeps):
                mujoco.mj_step(model, data)
                physics_step += 1

        # Early stop if the robot has clearly fallen. This keeps bad episodes from
        # dominating the dataset, while still recording the failure trajectory.
        kin_after = get_kinematics(data)
        if abs(float(kin_after["roll_rad"])) > math.radians(60.0):
            break

    return rows_written, physics_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect LQR MuJoCo balancing dataset as CSV.")
    parser.add_argument("--model", type=str, default=None, help="Path to scene.xml. Auto-detected if omitted.")
    parser.add_argument("--out", type=str, default="datasets/lqr_dataset.csv", help="Output CSV path.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to collect.")
    parser.add_argument("--duration", type=float, default=10.0, help="Episode duration in seconds.")
    parser.add_argument("--control-hz", type=float, default=200.0, help="LQR/logging control rate in Hz.")
    parser.add_argument(
        "--cmd-mode",
        choices=["zero", "random_step", "sine"],
        default="random_step",
        help="Setpoint generation mode.",
    )
    parser.add_argument("--max-speed", type=float, default=0.6, help="Max random/sine speed setpoint in m/s.")
    parser.add_argument("--max-yaw", type=float, default=2.0, help="Max random/sine yaw setpoint in rad/s-like motor offset.")
    parser.add_argument("--command-interval", type=float, default=2.0, help="Seconds between random command changes.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--append", action="store_true", help="Append to existing CSV instead of overwriting.")
    parser.add_argument(
        "--log-every-physics-step",
        action="store_true",
        help="Log every MuJoCo physics step instead of every control step. Creates large CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = resolve_model_path(args.model)
    out_path = pathlib.Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    robot = RobotLqr(model, data)
    rng = np.random.default_rng(args.seed)

    mode = "a" if args.append else "w"
    need_header = (not args.append) or (not out_path.exists()) or out_path.stat().st_size == 0

    total_rows = 0
    total_physics_steps = 0
    with out_path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if need_header:
            writer.writeheader()

        for ep in range(args.episodes):
            rows, physics_steps = collect_episode(
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
            )
            total_rows += rows
            total_physics_steps += physics_steps
            print(
                f"[episode {ep:04d}] rows={rows}, physics_steps={physics_steps}, "
                f"sim_time={data.time:.3f}s"
            )

    print(f"\nSaved: {out_path}")
    print(f"Total rows: {total_rows}")
    print(f"Total physics steps: {total_physics_steps}")
    print(f"Model: {model_path}")


if __name__ == "__main__":
    main()
