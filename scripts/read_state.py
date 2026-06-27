import mujoco
import numpy as np


MODEL_PATH = "src/simulation/scene.xml"


def quat_to_euler_wxyz(q):
    """
    MuJoCo free joint quaternion order: [w, x, y, z]
    Return roll, pitch, yaw in radians.
    """
    w, x, y, z = q

    # roll
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # yaw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

for step in range(1000):
    mujoco.mj_step(model, data)

    if step % 100 == 0:
        pos = data.qpos[0:3]
        quat = data.qpos[3:7]
        wheel_l = data.qpos[7]
        wheel_r = data.qpos[8]

        vel = data.qvel[0:3]
        ang_vel = data.qvel[3:6]
        wheel_l_vel = data.qvel[6]
        wheel_r_vel = data.qvel[7]

        roll, pitch, yaw = quat_to_euler_wxyz(quat)

        print(f"step={step}")
        print(f"  pos       = {pos}")
        print(f"  rpy[deg]  = {np.rad2deg([roll, pitch, yaw])}")
        print(f"  lin_vel   = {vel}")
        print(f"  ang_vel   = {ang_vel}")
        print(f"  wheel_pos = L {wheel_l:.4f}, R {wheel_r:.4f}")
        print(f"  wheel_vel = L {wheel_l_vel:.4f}, R {wheel_r_vel:.4f}")
