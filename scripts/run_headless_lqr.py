#!/usr/bin/env python3

import argparse
import csv
import math
import sys
from pathlib import Path

import mujoco
import numpy as np


# Repo root:
# balance-robot-mujoco-sim/
#   scripts/run_headless_lqr.py
#   src/simulation/scene.xml
#   src/simulation/robot_lqr.py
REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = REPO_ROOT / "src" / "simulation"
MODEL_PATH = SIM_DIR / "scene.xml"

sys.path.insert(0, str(SIM_DIR))
from robot_lqr import RobotLqr  # noqa: E402


def quat_wxyz_from_x_angle(angle_rad: float) -> np.ndarray:
    """
    MuJoCo free joint quaternion order is [w, x, y, z].
    In this robot code, rotation about x-axis is treated as pitch.
    """
    half = 0.5 * angle_rad
    return np.array([math.cos(half), math.sin(half), 0.0, 0.0], dtype=float)


def set_initial_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_z: float,
    initial_pitch_deg: float,
) -> None:
    """
    Set the robot near wheel-ground contact and give it a small initial pitch.

    scene.xml puts the floor plane at z = -0.1.
    In robot-02.xml, the wheel center is at body z + 0.034 and wheel radius is 0.034,
    so body_z = -0.1 places the wheel bottom at the floor plane.
    """
    mujoco.mj_resetData(model, data)

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0

    # Free joint position: [x, y, z]
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = body_z

    # Free joint orientation quaternion: [w, x, y, z]
    pitch_rad = math.radians(initial_pitch_deg)
    data.qpos[3:7] = quat_wxyz_from_x_angle(pitch_rad)

    # Wheel joint positions
    data.qpos[7] = 0.0
    data.qpos[8] = 0.0

    mujoco.mj_forward(model, data)


def get_snapshot(robot: RobotLqr, data: mujoco.MjData) -> dict:
    pitch = robot.get_pitch()
    pitch_dot = robot.get_pitch_dot()
    wheel_vel = robot.get_wheel_velocity()

    return {
        "time": float(data.time),
        "x": float(data.qpos[0]),
        "y": float(data.qpos[1]),
        "z": float(data.qpos[2]),
        "pitch_rad": float(pitch),
        "pitch_deg": float(math.degrees(pitch)),
        "pitch_dot": float(pitch_dot),
        "wheel_l_pos": float(data.qpos[7]),
        "wheel_r_pos": float(data.qpos[8]),
        "wheel_l_vel": float(data.joint("torso_l_wheel").qvel[0]),
        "wheel_r_vel": float(data.joint("torso_r_wheel").qvel[0]),
        "wheel_vel_avg": float(wheel_vel),
        "cmd_l": float(data.ctrl[0]),
        "cmd_r": float(data.ctrl[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Headless MuJoCo LQR simulation for the two-wheel balancing robot."
    )
    parser.add_argument("--duration", type=float, default=5.0, help="Simulation duration [s]")
    parser.add_argument("--control-hz", type=float, default=200.0, help="LQR update rate [Hz]")
    parser.add_argument("--print-hz", type=float, default=10.0, help="Console print rate [Hz]")
    parser.add_argument("--log-hz", type=float, default=100.0, help="CSV logging rate [Hz]")
    parser.add_argument("--speed", type=float, default=0.0, help="Target linear velocity [m/s]")
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw command added to wheel commands")
    parser.add_argument("--initial-pitch-deg", type=float, default=5.0, help="Initial x-axis pitch [deg]")
    parser.add_argument("--body-z", type=float, default=-0.1, help="Initial free-joint body z position")
    parser.add_argument("--fall-pitch-deg", type=float, default=60.0, help="Stop if abs(pitch) exceeds this [deg]")
    parser.add_argument(
        "--log-csv",
        type=str,
        default="logs/headless_lqr.csv",
        help="CSV log path. Use empty string to disable logging.",
    )
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    robot = RobotLqr(model, data)

    # Do not call robot.reset() here.
    # The original reset uses scipy Rotation.as_quat(), whose order is [x, y, z, w],
    # while MuJoCo qpos expects [w, x, y, z].
    # We explicitly initialize qpos[3:7] in MuJoCo's [w, x, y, z] order instead.
    set_initial_pose(
        model=model,
        data=data,
        body_z=args.body_z,
        initial_pitch_deg=args.initial_pitch_deg,
    )

    robot.velocity_linear_set_point = 0.0
    robot.yaw = 0.0
    robot.pitch_dot_filtered = 0.0
    robot.velocity_angular_filtered = 0.0

    dt = float(model.opt.timestep)
    total_steps = int(args.duration / dt)

    control_steps = max(1, int(round((1.0 / args.control_hz) / dt)))
    print_steps = max(1, int(round((1.0 / args.print_hz) / dt)))
    log_steps = max(1, int(round((1.0 / args.log_hz) / dt)))

    print("=== Headless LQR simulation ===")
    print(f"model path      : {MODEL_PATH}")
    print(f"MuJoCo timestep : {dt:.8f} s")
    print(f"duration        : {args.duration:.3f} s")
    print(f"total steps     : {total_steps}")
    print(f"control rate    : {args.control_hz:.1f} Hz, every {control_steps} sim steps")
    print(f"initial pitch   : {args.initial_pitch_deg:.3f} deg")
    print(f"target speed    : {args.speed:.3f} m/s")
    print(f"yaw command     : {args.yaw:.3f}")
    print()

    writer = None
    csv_file = None

    if args.log_csv:
        log_path = REPO_ROOT / args.log_csv
        log_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(log_path, "w", newline="")
        fieldnames = list(get_snapshot(robot, data).keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        print(f"logging to      : {log_path}")
        print()

    fall_pitch_rad = math.radians(args.fall_pitch_deg)
    fell = False

    try:
        for step in range(total_steps + 1):
            if step % control_steps == 0:
                robot.set_velocity_linear_set_point(args.speed)
                robot.set_yaw(args.yaw)
                robot.update_motor_speed()

            mujoco.mj_step(model, data)

            if writer is not None and step % log_steps == 0:
                writer.writerow(get_snapshot(robot, data))

            if step % print_steps == 0:
                s = get_snapshot(robot, data)
                print(
                    f"t={s['time']:7.3f} | "
                    f"pos=({s['x']:+.3f}, {s['y']:+.3f}, {s['z']:+.3f}) | "
                    f"pitch={s['pitch_deg']:+8.3f} deg | "
                    f"pitch_dot={s['pitch_dot']:+8.3f} rad/s | "
                    f"wheel_vel={s['wheel_vel_avg']:+8.3f} rad/s | "
                    f"cmd=({s['cmd_l']:+8.3f}, {s['cmd_r']:+8.3f})"
                )

            pitch = robot.get_pitch()
            if abs(pitch) > fall_pitch_rad:
                s = get_snapshot(robot, data)
                print()
                print(
                    f"[STOP] Robot likely fell: "
                    f"t={s['time']:.3f}s, pitch={s['pitch_deg']:.2f} deg"
                )
                fell = True
                break

    finally:
        if csv_file is not None:
            csv_file.close()

    final = get_snapshot(robot, data)
    print()
    print("=== Final state ===")
    print(f"time       : {final['time']:.3f} s")
    print(f"position   : x={final['x']:+.4f}, y={final['y']:+.4f}, z={final['z']:+.4f}")
    print(f"pitch      : {final['pitch_deg']:+.4f} deg")
    print(f"pitch_dot  : {final['pitch_dot']:+.4f} rad/s")
    print(f"wheel_vel  : {final['wheel_vel_avg']:+.4f} rad/s")
    print(f"cmd        : L={final['cmd_l']:+.4f}, R={final['cmd_r']:+.4f}")

    if fell:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
