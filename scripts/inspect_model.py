import glob
import mujoco

xml_files = glob.glob("**/*.xml", recursive=True) + glob.glob("**/*.mjcf", recursive=True)

if not xml_files:
    raise FileNotFoundError("No XML/MJCF model file found.")

print("Found model files:")
for i, path in enumerate(xml_files):
    print(f"[{i}] {path}")

model_path = xml_files[0]
print(f"\nLoading: {model_path}")

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

print("\n=== Model summary ===")
print("nq:", model.nq)
print("nv:", model.nv)
print("nu:", model.nu)
print("nbody:", model.nbody)
print("njnt:", model.njnt)
print("ngeom:", model.ngeom)

print("\n=== Joints ===")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    qpos_adr = model.jnt_qposadr[i]
    dof_adr = model.jnt_dofadr[i]
    jtype = model.jnt_type[i]
    print(f"{i}: name={name}, type={jtype}, qpos_adr={qpos_adr}, dof_adr={dof_adr}")

print("\n=== Actuators ===")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"{i}: name={name}")

print("\nStepping simulation...")
for _ in range(100):
    mujoco.mj_step(model, data)

print("qpos:", data.qpos)
print("qvel:", data.qvel)
print("ctrl:", data.ctrl)
