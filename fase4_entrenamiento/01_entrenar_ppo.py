"""
Fase 4 - Script 1: Primer entrenamiento con PPO.

Objetivo de este experimento: confirmar que la curva de recompensa en
TensorBoard sube con el entrenamiento (senal de que entorno + recompensa
+ algoritmo funcionan juntos de extremo a extremo). No se busca todavia
convergencia real de la tarea.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fase3_entorno_gym'))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from panda_pick_place_env import PandaPickPlaceEnv

N_ENVS = 8
TOTAL_TIMESTEPS = 100_000
LOG_DIR = "./tensorboard_logs/"
MODEL_SAVE_PATH = "./modelos/ppo_panda_v1"

if __name__ == "__main__":
    # SubprocVecEnv corre cada entorno en un proceso separado, aprovechando
    # los nucleos de CPU disponibles en paralelo
    env = make_vec_env(
        PandaPickPlaceEnv,
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv,
    )

    model = PPO(
        "MlpPolicy",       # red neuronal simple (fully connected), adecuada
                           # para observaciones vectoriales de baja dimension
                           # como la nuestra (no imagenes -> no hace falta CNN)
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    print(f"Entrenando PPO durante {TOTAL_TIMESTEPS} timesteps con {N_ENVS} entornos en paralelo...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Modelo guardado en {MODEL_SAVE_PATH}.zip")

    env.close()
