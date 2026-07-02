#!/usr/bin/env python3

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np


# Expected location when executed from this repository:
# balance-robot-mujoco-sim/
#   src/simulation/run_headless_lqr.py
#   src/simulation/scene.xml
#   src/simulation/robot_lqr.py
REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = REPO_ROOT / "simulation"
MODEL_PATH = SIM_DIR / "scene.xml"

sys.path.insert(0, str(SIM_DIR))
from robot_lqr import RobotLqr  # noqa: E402


ROBOT_BODY_NAME = "robot_body"
FLOOR_GEOM_NAME = "floor"
LEFT_WHEEL_BODY_NAME = "l_wheel"
RIGHT_WHEEL_BODY_NAME = "r_wheel"
LEFT_WHEEL_GEOM_NAME = "l_wheel_geom"
RIGHT_WHEEL_GEOM_NAME = "r_wheel_geom"


_AXIS_TO_VECTOR = {
    "x": np.array([1.0, 0.0, 0.0], dtype=float),
    "y": np.array([0.0, 1.0, 0.0], dtype=float),
    "z": np.array([0.0, 0.0, 1.0], dtype=float),
}


def _name_to_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"MuJoCo object not found: {name!r}")
    return obj_id


def quat_wxyz_from_axis_angle(axis_name: str, angle_rad: float) -> np.ndarray:
    """
    Return a MuJoCo free-joint quaternion in [w, x, y, z] order.

    The CAD/URDF-derived model uses wheel hinge axes along +/-Y, so the physically
    correct forward/backward pitch axis may be y. However, the current RobotLqr
    may still read x-axis pitch. Keep this selectable with --initial-pitch-axis.
    """
    axis = _AXIS_TO_VECTOR[axis_name]
    half = 0.5 * angle_rad
    return np.array(
        [math.cos(half), *(math.sin(half) * axis)],
        dtype=float,
    )


def get_floor_z(model: mujoco.MjModel, floor_z_arg: Optional[float]) -> float:
    """Infer the floor plane z-position unless the user explicitly gives one."""
    if floor_z_arg is not None:
        return float(floor_z_arg)

    floor_gid = _name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME)
    return float(model.geom_pos[floor_gid][2])


def get_wheel_radius(model: mujoco.MjModel) -> float:
    """Use the collision cylinder radius from the left/right wheel geoms."""
    l_gid = _name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, LEFT_WHEEL_GEOM_NAME)
    r_gid = _name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, RIGHT_WHEEL_GEOM_NAME)
    l_radius = float(model.geom_size[l_gid][0])
    r_radius = float(model.geom_size[r_gid][0])
    if not math.isclose(l_radius, r_radius, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"Left/right wheel radii differ: {l_radius:.9f} vs {r_radius:.9f}"
        )
    return l_radius


def infer_body_z_for_wheel_contact(
    model: mujoco.MjModel,
    floor_z: float,
    ground_clearance: float,
) -> float:
    """
    Compute free-joint body z so the wheel cylinders start at floor contact.

    Formula for the current CAD model:
        body_z = floor_z + wheel_radius - wheel_local_z + clearance

    where wheel_local_z is the wheel body's local z-offset from robot_body plus
    the wheel collision geom's local z-offset inside the wheel body.
    """
    l_bid = _name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, LEFT_WHEEL_BODY_NAME)
    r_bid = _name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_WHEEL_BODY_NAME)
    l_gid = _name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, LEFT_WHEEL_GEOM_NAME)
    r_gid = _name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, RIGHT_WHEEL_GEOM_NAME)

    l_local_z = float(model.body_pos[l_bid][2] + model.geom_pos[l_gid][2])
    r_local_z = float(model.body_pos[r_bid][2] + model.geom_pos[r_gid][2])
    wheel_local_z = 0.5 * (l_local_z + r_local_z)
    wheel_radius = get_wheel_radius(model)

    return float(floor_z + wheel_radius - wheel_local_z + ground_clearance)


def reset_filters_and_commands(robot: RobotLqr) -> None:
    robot.velocity_linear_set_point = 0.0
    robot.yaw = 0.0
    robot.pitch_dot_filtered = 0.0
    robot.velocity_angular_filtered = 0.0


def set_initial_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_z: float,
    initial_pitch_deg: float,
    initial_pitch_axis: str,
) -> None:
    """Set free-joint pose and wheel joint positions for the headless test."""
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
    data.qpos[3:7] = quat_wxyz_from_axis_angle(initial_pitch_axis, pitch_rad)

    # Wheel joint positions. The current model has two wheel hinge joints after
    # the free joint, so these are qpos[7] and qpos[8].
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


def print_contact_summary(model: mujoco.MjModel, data: mujoco.MjData, label: str) -> None:
    print(f"{label} contacts : {data.ncon}")
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        print(f"  {i}: {g1} <-> {g2} | dist={c.dist:+.6e}")


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
    parser.add_argument(
        "--initial-pitch-deg",
        type=float,
        default=5.0,
        help="Initial pitch perturbation [deg]",
    )
    parser.add_argument(
        "--initial-pitch-axis",
        choices=("x", "y", "z"),
        default="y",
        help=(
            "Axis used to apply initial pitch. For the CAD/URDF-derived model, "
            "forward/backward pitch is usually about the y-axis."
        ),
    )
    parser.add_argument(
        "--body-z",
        type=float,
        default=None,
        help=(
            "Initial free-joint body z position. If omitted, it is inferred from "
            "floor height, wheel radius, and wheel local z-offset."
        ),
    )
    parser.add_argument(
        "--floor-z",
        type=float,
        default=None,
        help="Floor z-position. If omitted, infer from geom named 'floor'.",
    )
    parser.add_argument(
        "--ground-clearance",
        type=float,
        default=0.0,
        help="Extra initial clearance above wheel-ground contact [m].",
    )
    parser.add_argument("--fall-pitch-deg", type=float, default=60.0, help="Stop if abs(pitch) exceeds this [deg]")
    parser.add_argument(
        "--print-initial-contacts",
        action="store_true",
        help="Print contacts immediately after initialization.",
    )
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

    floor_z = get_floor_z(model, args.floor_z)
    wheel_radius = get_wheel_radius(model)
    body_z = (
        float(args.body_z)
        if args.body_z is not None
        else infer_body_z_for_wheel_contact(
            model=model,
            floor_z=floor_z,
            ground_clearance=args.ground_clearance,
        )
    )

    robot = RobotLqr(model, data)

    # Do not call robot.reset() here.
    # scipy Rotation.as_quat() returns [x, y, z, w], while MuJoCo free-joint qpos
    # expects [w, x, y, z]. We explicitly initialize qpos[3:7] in MuJoCo order.
    set_initial_pose(
        model=model,
        data=data,
        body_z=body_z,
        initial_pitch_deg=args.initial_pitch_deg,
        initial_pitch_axis=args.initial_pitch_axis,
    )
    reset_filters_and_commands(robot)

    initial_snapshot = get_snapshot(robot, data)
    if abs(initial_snapshot["pitch_deg"] - args.initial_pitch_deg) > 1.0:
        print(
            "[WARN] RobotLqr.get_pitch() does not match the requested initial pitch.\n"
            f"       requested={args.initial_pitch_deg:+.3f} deg about {args.initial_pitch_axis}-axis, "
            f"RobotLqr reads={initial_snapshot['pitch_deg']:+.3f} deg.\n"
            "       This usually means RobotLqr is reading a different pitch axis."
        )
        print()

    if args.print_initial_contacts:
        print_contact_summary(model, data, "initial")
        print()

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
    print(f"floor z         : {floor_z:+.4f} m")
    print(f"wheel radius    : {wheel_radius:.4f} m")
    print(f"initial body z  : {body_z:+.4f} m")
    print(f"initial pitch   : {args.initial_pitch_deg:.3f} deg about {args.initial_pitch_axis}-axis")
    print(f"RobotLqr pitch  : {initial_snapshot['pitch_deg']:+.3f} deg")
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
