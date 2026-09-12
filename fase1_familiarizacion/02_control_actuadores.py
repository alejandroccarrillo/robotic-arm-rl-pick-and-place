"""
Fase 1 - Script 2: Control de actuadores via data.ctrl.

Objetivo: mover el brazo a una pose objetivo escribiendo en los actuadores
(7 del brazo + 1 del gripper) y observar el efecto en el visor.
"""
import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path('mujoco_menagerie/franka_emika_panda/scene.xml')
data = mujoco.MjData(model)

# Pose objetivo para las 7 articulaciones del brazo (radianes)
pose_objetivo = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7]

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 15:
        t = time.time() - start

        data.ctrl[:7] = pose_objetivo
        # Gripper: abierto los primeros 5s, cerrado despues (rango 0-255)
        data.ctrl[7] = 0 if t < 5 else 255

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)

print('OK: prueba de control terminada')
