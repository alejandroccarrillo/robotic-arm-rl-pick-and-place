"""
Fase 4 - Script 5: Diagnostico de si el "exito" es un agarre real o
el cubo siendo empujado sin ser agarrado (reward hacking).

Registra en cada step si la condicion de "grasping" (mano cerca +
gripper cerrado) estuvo activa en algun momento antes de alcanzar
el exito.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fase3_entorno_gym'))

from stable_baselines3 import PPO
from panda_pick_place_env import PandaPickPlaceEnv

MODEL_PATH = "./modelos/best_v2/best_model.zip"

env = PandaPickPlaceEnv()
model = PPO.load(MODEL_PATH)

obs, info = env.reset()
terminated = truncated = False
step = 0
hubo_grasping_alguna_vez = False

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1

    if info['grasping']:
        hubo_grasping_alguna_vez = True

    if step % 20 == 0 or terminated:
        print(f"step={step} | dist_hand_cube={info['dist_hand_cube']:.3f} | "
              f"grasping_ahora={info['grasping']} | gripper_ctrl={env.current_ctrl[7]:.0f} | "
              f"dist_cube_target={info['dist_cube_target']:.3f}")

print()
print(f"RESULTADO: exito={info.get('exito')} | hubo_grasping_en_algun_momento={hubo_grasping_alguna_vez}")
if info.get('exito') and not hubo_grasping_alguna_vez:
    print("CONFIRMADO: el cubo llego al target SIN haber sido agarrado nunca -> reward hacking")

env.close()
