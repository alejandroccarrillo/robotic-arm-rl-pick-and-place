"""
Fase 2 - Script 2: Probar la funcion de recompensa con la escena real.

Mueve el brazo hacia una pose verificada cerca del cubo (encontrada con
03_buscar_pose_acercamiento.py) y muestra la evolucion de la recompensa.
Primero con el gripper abierto (no deberia activarse el bonus de
grasping aunque este cerca), luego cerrado (si deberia activarse).
"""
import mujoco
import mujoco.viewer
import time
from recompensa import calcular_recompensa

model = mujoco.MjModel.from_xml_path('fase2_definir_tarea/escena_pick_and_place.xml')
data = mujoco.MjData(model)

hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'target')

# Pose verificada con mj_forward: deja la mano a ~0.049m del cubo
pose_cerca_cubo = [0.0, 0.8, 0.0, -2.2, 0.0, 2.6, 0.7]

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    last_print = 0
    while viewer.is_running() and time.time() - start < 12:
        t = time.time() - start

        data.ctrl[:7] = pose_cerca_cubo
        # Gripper abierto los primeros 6s, cerrado despues
        data.ctrl[7] = 0 if t < 6 else 255

        mujoco.mj_step(model, data)
        viewer.sync()

        if t - last_print >= 0.5:
            hand_pos = data.xpos[hand_id]
            cube_pos = data.xpos[cube_id]
            target_pos = data.site_xpos[target_id]

            reward, terminated, info = calcular_recompensa(
                hand_pos, cube_pos, target_pos, data.ctrl[7]
            )

            print(f't={t:.1f}s | reward={reward:.3f} | '
                  f'dist_hand_cube={info["dist_hand_cube"]:.3f} | '
                  f'grasping={info["grasping"]} | '
                  f'terminated={terminated}')
            last_print = t

print('OK: prueba de recompensa terminada')
