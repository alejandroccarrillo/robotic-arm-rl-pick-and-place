"""
Fase 4 - Script 7: Diagnostico definitivo de agarre real vs cubo
volcado/rodando. Revisa la orientacion del cubo (deberia mantenerse
practicamente sin rotar si esta agarrado limpiamente por caras
paralelas) y la posicion de ambos dedos respecto al cubo (deberian
estar a ambos lados si hay agarre real).
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fase3_entorno_gym'))

from stable_baselines3 import PPO
from panda_pick_place_env import PandaPickPlaceEnv

MODEL_PATH = "./modelos/best_v3/best_model.zip"

env = PandaPickPlaceEnv()
model = PPO.load(MODEL_PATH)

left_finger_id = None
right_finger_id = None
import mujoco
left_finger_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'left_finger')
right_finger_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'right_finger')

obs, info = env.reset()
terminated = truncated = False
step = 0
quat_inicial = env.data.qpos[12:16].copy()

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1

    cube_z = env.data.xpos[env.cube_id][2]
    quat_actual = env.data.qpos[12:16]
    # Diferencia angular aproximada respecto a la orientacion inicial (sin rotar)
    desviacion_quat = np.linalg.norm(quat_actual - quat_inicial)

    left_pos = env.data.xpos[left_finger_id]
    right_pos = env.data.xpos[right_finger_id]
    dist_dedos = np.linalg.norm(left_pos - right_pos)
    cube_pos = env.data.xpos[env.cube_id]
    dist_left_cube = np.linalg.norm(left_pos - cube_pos)
    dist_right_cube = np.linalg.norm(right_pos - cube_pos)

    if step % 10 == 0 or terminated:
        print(f"step={step} | cube_z={cube_z:.4f} | desv_quat={desviacion_quat:.3f} | "
              f"dist_dedos={dist_dedos:.4f} | dist_L_cubo={dist_left_cube:.4f} | dist_R_cubo={dist_right_cube:.4f}")

print()
print(f"Si desv_quat es alta (>0.2) -> el cubo roto/volco, no fue agarrado limpiamente")
print(f"Si dist_dedos es muy pequeña (<0.01) -> los dedos estan juntos, no hay nada agarrado entre ellos")

env.close()
