"""
Fase 3 - Script 2: Validacion oficial con la utilidad check_env de
Stable-Baselines3. Comprueba conformidad estricta con la API de
Gymnasium antes de conectar el entorno con un algoritmo real (Fase 4).
"""
from stable_baselines3.common.env_checker import check_env
from panda_pick_place_env import PandaPickPlaceEnv

env = PandaPickPlaceEnv()
check_env(env, warn=True)
print("OK: check_env de Stable-Baselines3 no encontro problemas criticos")
