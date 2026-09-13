"""
Fase 4 - Script 12: Diagnostico completo de best_v7 - agarre sostenido
confirmado, pero el episodio no termino en el diagnostico anterior.
Aqui anadimos dist_cube_target y resultado final para entender si el
agente se acerco al target sujetando el cubo o se quedo inmovil.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fase3_entorno_gym'))

from stable_baselines3 import PPO
from panda_pick_place_env import PandaPickPlaceEnv

MODEL_PATH = "./modelos/best_v8/best_model.zip"

env = PandaPickPlaceEnv()
model = PPO.load(MODEL_PATH)

obs, info = env.reset()
terminated = truncated = False
step = 0

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1

    if step % 20 == 0 or terminated or truncated:
        cube_z = env.data.xpos[env.cube_id][2]
        print(f"step={step} | dist_cube_target={info['dist_cube_target']:.3f} | "
              f"dist_hand_cube={info['dist_hand_cube']:.3f} | cube_z={cube_z:.4f} | "
              f"grasping={info['grasping']}")

print()
print(f"FINAL: exito={info.get('exito')} | fallo={info.get('fallo')} | truncated={truncated} | steps={step}")
env.close()
