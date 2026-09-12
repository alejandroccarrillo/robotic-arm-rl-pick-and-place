"""
Fase 2 - Script 1: Verificar la escena de pick-and-place.

Objetivo: cargar la escena con el Panda + cubo + target, dejar que la
fisica se asiente (el cubo debe caer y quedarse quieto sobre el suelo,
no atravesarlo ni salir disparado) y confirmar las posiciones iniciales.
"""
import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path('fase2_definir_tarea/escena_pick_and_place.xml')
data = mujoco.MjData(model)

cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'target')

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    last_print = 0
    while viewer.is_running() and time.time() - start < 10:
        t = time.time() - start

        mujoco.mj_step(model, data)
        viewer.sync()

        if t - last_print >= 1.0:
            cube_pos = data.xpos[cube_id]
            target_pos = data.site_xpos[target_id]
            print(f't={t:.1f}s | cubo=({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}) '
                  f'| target=({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})')
            last_print = t

print('OK: verificacion de escena terminada')
