"""
Fase 4 - Script 6: Diagnostico de si el cubo es levantado de verdad
(agarre real) o arrastrado por el suelo con el gripper cerrado.

Un agarre real deberia mostrar cube_z subiendo claramente por encima
de 0.02 (altura de reposo) en algun momento del episodio.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fase3_entorno_gym'))

from stable_baselines3 import PPO
from panda_pick_place_env import PandaPickPlaceEnv

MODEL_PATH = "./modelos/best_v3/best_model.zip"
ALTURA_REPOSO = 0.02
UMBRAL_LEVANTADO = 0.03  # 1cm por encima de reposo ya cuenta como "levantado"

env = PandaPickPlaceEnv()
model = PPO.load(MODEL_PATH)

obs, info = env.reset()
terminated = truncated = False
step = 0
max_altura_cubo = ALTURA_REPOSO

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1

    cube_z = env.data.xpos[env.cube_id][2]
    max_altura_cubo = max(max_altura_cubo, cube_z)

    if step % 10 == 0 or terminated:
        print(f"step={step} | cube_z={cube_z:.4f} | grasping={info['grasping']} | "
              f"dist_cube_target={info['dist_cube_target']:.3f}")

print()
print(f"Altura maxima alcanzada por el cubo: {max_altura_cubo:.4f} (reposo=0.02)")
if max_altura_cubo > UMBRAL_LEVANTADO:
    print("El cubo SI fue levantado del suelo -> agarre real")
else:
    print("El cubo NUNCA se elevo significativamente -> fue arrastrado/empujado, no agarrado")

env.close()
