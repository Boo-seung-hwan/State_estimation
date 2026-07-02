#!/usr/bin/env python3
"""
Calculate a simplified LQR gain for the two-wheel balancing robot.

This script is intentionally conservative about output names:

    LQR_K                      = real 1x4 feedback gain to paste into robot_lqr.py
    closed-loop eigenvalues    = diagnostic poles of A - B @ K, may be complex

If you see complex numbers in "closed-loop eigenvalues", that is normal. Do NOT
copy eigenvalues into robot_lqr.py. Only copy LQR_K.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np
from scipy.linalg import solve_continuous_are


SCRIPT_DIR = Path(__file__).resolve().parent


def default_model_path() -> Path:
    """
    Prefer the usual repo layout:
        balance-robot-mujoco-sim/scripts/lqr_gain_calculator.py
        balance-robot-mujoco-sim/src/simulation/scene.xml

    Fall back to scene.xml next to this file for ad-hoc copies.
    """
    candidates = [
        SCRIPT_DIR.parent / "src" / "simulation" / "scene.xml",
        SCRIPT_DIR / "scene.xml",
        Path.cwd() / "src" / "simulation" / "scene.xml",
    ]
    for p in candidates:
        if p.exists():
            return p
    return Path("src/simulation/scene.xml")


DEFAULT_MODEL = default_model_path()


def resolve_existing_path(path: Path) -> Path:
    """Resolve a model path robustly and print a useful error if it is missing."""
    candidates = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                Path.cwd() / path,
                SCRIPT_DIR / path,
                SCRIPT_DIR.parent / path,
            ]
        )

    seen = set()
    unique_candidates = []
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            unique_candidates.append(rp)
            seen.add(rp)

    for p in unique_candidates:
        if p.exists():
            return p

    msg = ["Could not find model XML. Tried:"]
    msg.extend(f"  - {p}" for p in unique_candidates)
    msg.append("")
    msg.append("Run from the repository root, for example:")
    msg.append("  cd /workspace/balance-robot-mujoco-sim")
    msg.append("  python scripts/lqr_gain_calculator.py --model src/simulation/scene.xml")
    raise FileNotFoundError("\n".join(msg))


def name2id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return idx


def compute_robot_properties(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_wheel_geom: str,
    right_wheel_geom: str,
    left_wheel_joint: str,
    right_wheel_joint: str,
) -> dict[str, float | np.ndarray]:
    """Extract simplified physical parameters from the loaded MuJoCo model."""
    mujoco.mj_forward(model, data)

    # Total mass and full-body COM from inertial frames.
    body_ids = range(1, model.nbody)  # skip world body 0
    masses = np.array([model.body_mass[i] for i in body_ids], dtype=float)
    xipos = np.array([data.xipos[i] for i in body_ids], dtype=float)

    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError("Total robot mass is zero or invalid. Check inertial tags.")

    com = np.sum(xipos * masses[:, None], axis=0) / total_mass

    # Wheel radius from cylinder collision geoms.
    lgid = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, left_wheel_geom)
    rgid = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, right_wheel_geom)
    wheel_radius = float(0.5 * (model.geom_size[lgid, 0] + model.geom_size[rgid, 0]))

    # Wheel-axis location from hinge joint anchors.
    ljid = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, left_wheel_joint)
    rjid = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, right_wheel_joint)
    wheel_axis_pos = 0.5 * (data.xanchor[ljid] + data.xanchor[rjid])

    # Simplified inverted-pendulum length: vertical COM distance above wheel axis.
    l_com = float(com[2] - wheel_axis_pos[2])
    if l_com <= 0.0:
        raise ValueError(
            f"COM is not above wheel axis: l={l_com:.6g}. "
            "Check body orientation, inertial tags, and initial pose."
        )

    return {
        "m": total_mass,
        "r": wheel_radius,
        "l": l_com,
        "com": com,
        "wheel_axis_pos": wheel_axis_pos,
    }


def continuous_lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
    """Solve continuous-time LQR using SciPy."""
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    eigvals = np.linalg.eigvals(A - B @ K)
    return K, P, eigvals


def build_simplified_model(g: float, m: float, l: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Simplified linear inverted-pendulum model used by the original script.

    x = [theta, theta_dot, position, velocity]^T
    x_dot = A x + B u
    """
    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [g / l, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-g / m, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    B = np.array(
        [
            [0.0],
            [-1.0 / (m * l**2)],
            [0.0],
            [1.0 / m],
        ],
        dtype=float,
    )
    return A, B


def controllability_rank(A: np.ndarray, B: np.ndarray) -> int:
    n = A.shape[0]
    ctrb = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])
    return int(np.linalg.matrix_rank(ctrb))


def real_lqr_gain(K: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """
    Return K as a real vector.

    Complex closed-loop eigenvalues are normal, but K itself must be real for
    this real-valued model. If K has meaningful imaginary parts, something is
    wrong and we should not silently copy it into robot_lqr.py.
    """
    K = np.asarray(K)
    imag_max = float(np.max(np.abs(np.imag(K)))) if np.iscomplexobj(K) else 0.0
    if imag_max > tol:
        raise ValueError(
            f"LQR gain K has non-negligible imaginary component: max |imag(K)|={imag_max:.3e}. "
            "Do not use this gain directly."
        )
    return np.real(K).reshape(-1)


def fmt_real_vector(v: Iterable[float], precision: int = 12) -> str:
    return "[" + ", ".join(f"{float(x): .{precision}g}" for x in v) + "]"


def fmt_eigvals(eigvals: np.ndarray, precision: int = 6) -> str:
    parts = []
    for z in np.asarray(eigvals).reshape(-1):
        re = float(np.real(z))
        im = float(np.imag(z))
        if abs(im) < 1e-12:
            parts.append(f"{re:.{precision}g}")
        else:
            sign = "+" if im >= 0 else "-"
            parts.append(f"{re:.{precision}g}{sign}{abs(im):.{precision}g}j")
    return "[" + ", ".join(parts) + "]"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate a simplified LQR gain from the current MuJoCo model."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to MuJoCo scene.xml")
    parser.add_argument("--g", type=float, default=9.81)
    parser.add_argument("--m-override", type=float, default=None)
    parser.add_argument("--l-override", type=float, default=None)
    parser.add_argument("--r-override", type=float, default=None)
    parser.add_argument("--q-theta", type=float, default=0.7)
    parser.add_argument("--q-theta-dot", type=float, default=0.05)
    parser.add_argument("--q-position", type=float, default=0.0)
    parser.add_argument("--q-velocity", type=float, default=2.5)
    parser.add_argument("--r-control", type=float, default=0.5)
    parser.add_argument("--left-wheel-geom", default="l_wheel_geom")
    parser.add_argument("--right-wheel-geom", default="r_wheel_geom")
    parser.add_argument("--left-wheel-joint", default="torso_l_wheel")
    parser.add_argument("--right-wheel-joint", default="torso_r_wheel")
    args = parser.parse_args()

    model_path = resolve_existing_path(args.model)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    props = compute_robot_properties(
        model,
        data,
        left_wheel_geom=args.left_wheel_geom,
        right_wheel_geom=args.right_wheel_geom,
        left_wheel_joint=args.left_wheel_joint,
        right_wheel_joint=args.right_wheel_joint,
    )

    m = float(args.m_override if args.m_override is not None else props["m"])
    l = float(args.l_override if args.l_override is not None else props["l"])
    r = float(args.r_override if args.r_override is not None else props["r"])

    if m <= 0.0:
        raise ValueError(f"Invalid mass m={m}")
    if l <= 0.0:
        raise ValueError(f"Invalid COM length l={l}")
    if r <= 0.0:
        raise ValueError(f"Invalid wheel radius r={r}")
    if args.r_control <= 0.0:
        raise ValueError("--r-control must be positive")

    A, B = build_simplified_model(g=args.g, m=m, l=l)
    Q = np.diag([args.q_theta, args.q_theta_dot, args.q_position, args.q_velocity])
    R = np.array([[args.r_control]], dtype=float)

    rank = controllability_rank(A, B)
    if rank < A.shape[0]:
        print(f"[WARN] controllability rank is {rank}/{A.shape[0]}. LQR result may be problematic.")

    K, _, eigvals = continuous_lqr(A, B, Q, R)
    K_vec = real_lqr_gain(K)

    print("=== MuJoCo-derived simplified LQR gain ===")
    print(f"model              : {model_path}")
    print(f"total mass m       : {m:.6f} kg")
    print(f"COM world          : {np.asarray(props['com'])}")
    print(f"wheel axis world   : {np.asarray(props['wheel_axis_pos'])}")
    print(f"COM height l       : {l:.6f} m")
    print(f"wheel radius r     : {r:.6f} m")
    print(f"controllability    : rank {rank}/{A.shape[0]}")
    print()
    print("A =")
    print(A)
    print("B =")
    print(B)
    print("Q =")
    print(Q)
    print("R =")
    print(R)
    print()
    print("# Copy ONLY these real values into robot_lqr.py:")
    print(f"LQR_K = np.array({fmt_real_vector(K_vec)}, dtype=float)")
    print(f"WHEEL_RADIUS = {r:.12g}")
    print()
    print("# Diagnostic only. These are NOT gains; complex values here are normal.")
    print(f"closed_loop_eigenvalues = {fmt_eigvals(eigvals)}")

    if abs(args.q_position) < 1e-15:
        print()
        print(
            "[NOTE] q_position is 0, so the controller does not try to return x to the origin. "
            "A near-zero closed-loop eigenvalue can be expected for the free position mode."
        )

    print()
    print("Next checks: pitch axis/sign in robot_lqr.py, then left/right motor sign.")


if __name__ == "__main__":
    main()
