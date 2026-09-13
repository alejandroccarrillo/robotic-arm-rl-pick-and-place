"""
Fase 4 - Script 8: Verificar la deteccion de contacto real
dedo-cubo usando data.contact de MuJoCo, antes de integrarla
en la funcion de recompensa.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fase3_entorno_gym'))

import mujoco
from stable_baselines3 import PPO
from panda_pick_place_env import PandaPickPlaceEnv

# Geoms de colision real (contype=1), identificados en Fase 4
LEFT_FINGER_COLLISION_GEOMS = set(range(68, 74))   # 68-73
RIGHT_FINGER_COLLISION_GEOMS = set(range(76, 82))  # 76-81
CUBE_GEOM = 82


def hay_contacto_dedo_cubo(data, finger_geoms, cube_geom):
    """Comprueba si algun contacto activo en data.contact involucra
    un geom del dedo dado y el geom del cubo."""
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        if (g1 in finger_geoms and g2 == cube_geom) or (g2 in finger_geoms and g1 == cube_geom):
            return True
    return False


MODEL_PATH = "./modelos/best_v7/best_model.zip"
env = PandaPickPlaceEnv()
model = PPO.load(MODEL_PATH)

obs, info = env.reset()
terminated = truncated = False
step = 0

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1

    contacto_izq = hay_contacto_dedo_cubo(env.data, LEFT_FINGER_COLLISION_GEOMS, CUBE_GEOM)
    contacto_der = hay_contacto_dedo_cubo(env.data, RIGHT_FINGER_COLLISION_GEOMS, CUBE_GEOM)
    agarre_real = contacto_izq and contacto_der

    if step % 10 == 0 or terminated:
        print(f"step={step} | contacto_izq={contacto_izq} | contacto_der={contacto_der} | "
              f"agarre_real={agarre_real} | ncon_total={env.data.ncon}")

env.close()
