"""
Fase 2 - Script de debug: verificar si el cubo se mueve de su
posicion inicial cuando el brazo se desplaza hacia el, para
diagnosticar por que la recompensa no baja como se esperaba.
"""
import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path('fase2_definir_tarea/escena_pick_and_place.xml')
data = mujoco.MjData(model)

hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cube')

pose_cerca_cubo = [0.0, 0.8, 0.0, -2.2, 0.0, 2.6, 0.7]

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    last_print = 0
    while viewer.is_running() and time.time() - start < 8:
        t = time.time() - start

        data.ctrl[:7] = pose_cerca_cubo
        data.ctrl[7] = 0

        mujoco.mj_step(model, data)
        viewer.sync()

        if t - last_print >= 0.3:
            hand_pos = data.xpos[hand_id]
            cube_pos = data.xpos[cube_id]
            print(f't={t:.2f}s | hand=({hand_pos[0]:.3f},{hand_pos[1]:.3f},{hand_pos[2]:.3f}) '
                  f'| cube=({cube_pos[0]:.3f},{cube_pos[1]:.3f},{cube_pos[2]:.3f})')
            last_print = t

print('OK: debug terminado')
