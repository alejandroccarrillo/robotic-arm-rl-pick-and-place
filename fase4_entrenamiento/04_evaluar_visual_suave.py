"""
Fase 4 - Script 4: Evaluacion visual con sincronizacion del viewer en
cada sub-paso de fisica (no solo al final de cada step() de Gymnasium),
para diagnosticar si el movimiento erratico observado es real o un
artefacto de renderizado por el frame_skip.
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fase3_entorno_gym'))

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from panda_pick_place_env import PandaPickPlaceEnv, FRAME_SKIP

MODEL_PATH = "./modelos/best_v4/best_model.zip"

env = PandaPickPlaceEnv()
model = PPO.load(MODEL_PATH)

obs, info = env.reset()

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    terminated = truncated = False
    step = 0

    while viewer.is_running() and not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)

        # Replicamos la logica interna de step() pero sincronizando
        # el viewer en CADA sub-paso de fisica, no solo al final
        action_clipped = np.clip(action, -1.0, 1.0)
        delta_brazo = action_clipped[:7] * 0.05
        delta_gripper = action_clipped[7] * 25.0

        env.current_ctrl[:7] = np.clip(
            env.current_ctrl[:7] + delta_brazo,
            env.model.actuator_ctrlrange[:7, 0],
            env.model.actuator_ctrlrange[:7, 1],
        )
        env.current_ctrl[7] = np.clip(env.current_ctrl[7] + delta_gripper, 0, 255)
        env.data.ctrl[:] = env.current_ctrl

        for _ in range(FRAME_SKIP):
            mujoco.mj_step(env.model, env.data)
            viewer.sync()
            time.sleep(0.01)

        env.step_count += 1
        obs = env._get_obs()

        hand_pos = env.data.xpos[env.hand_id]
        cube_pos = env.data.xpos[env.cube_id]
        target_pos = env.data.site_xpos[env.target_id]
        from recompensa import calcular_recompensa
        reward, terminated, info = calcular_recompensa(hand_pos, cube_pos, target_pos, env.current_ctrl[7])
        truncated = env.step_count >= 300

        step += 1

print(f"Terminado en step {step}: exito={info.get('exito')}, fallo={info.get('fallo')}")
env.close()
