import mujoco
import numpy as np


MODEL_PATH = "src/simulation/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

for step in range(2000):
    # 좌우 바퀴에 같은 torque 입력
    data.ctrl[0] = 0.1
    data.ctrl[1] = 0.1

    mujoco.mj_step(model, data)

    if step % 100 == 0:
        print(
            f"step={step:4d} | "
            f"x={data.qpos[0]:+.4f}, z={data.qpos[2]:+.4f}, "
            f"wheel_l={data.qpos[7]:+.4f}, wheel_r={data.qpos[8]:+.4f}, "
            f"wheel_l_vel={data.qvel[6]:+.4f}, wheel_r_vel={data.qvel[7]:+.4f}, "
            f"ctrl={data.ctrl}"
        )
