"""
Fase 1 - Script 1: Cargar el modelo del Franka Panda y visualizarlo.

Objetivo: confirmar que MuJoCo puede cargar el modelo desde mujoco_menagerie
y que el renderizado funciona en el entorno (WSL2 + WSLg).
"""
import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path('mujoco_menagerie/franka_emika_panda/scene.xml')
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 10:
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

print('OK: el visor se cerro sin errores')
