"""
Fase 1 - Script 3: Lectura del estado cartesiano del efector final.

Objetivo: combinar control (data.ctrl) con lectura de estado (data.xpos)
para obtener la posicion 3D de la mano del robot en el mundo.
Esto es la base para la funcion de recompensa de la Fase 2
(distancia efector-objeto).
"""
import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path('mujoco_menagerie/franka_emika_panda/scene.xml')
data = mujoco.MjData(model)

hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')

pose_objetivo = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7]

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    last_print = 0
    while viewer.is_running() and time.time() - start < 15:
        t = time.time() - start

        data.ctrl[:7] = pose_objetivo
        data.ctrl[7] = 0

        mujoco.mj_step(model, data)
        viewer.sync()

        if t - last_print >= 1.0:
            pos = data.xpos[hand_id]
            print(f't={t:.1f}s | posicion hand (x,y,z) = {pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}')
            last_print = t

        time.sleep(0.002)

print('OK: prueba de lectura de estado terminada')
