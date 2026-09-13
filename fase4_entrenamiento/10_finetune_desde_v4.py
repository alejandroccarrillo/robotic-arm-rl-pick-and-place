"""
Fase 4 - Script 10: Fine-tuning conservador desde best_v4.

v5 (continuacion a tasa de aprendizaje normal) empeoro respecto a v4 -
la politica perdio el agarre bilateral que ya habia encontrado, un caso
tipico de inestabilidad de PPO. Reintentamos con learning_rate mas bajo
para refinar sin arriesgarnos a perder el comportamiento ya logrado.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fase3_entorno_gym'))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from panda_pick_place_env import PandaPickPlaceEnv

N_ENVS = 8
TIMESTEPS_ADICIONALES = 300_000
LOG_DIR = "./tensorboard_logs/"
MODELO_BASE = "./modelos/best_v4/best_model.zip"  # partimos de v4, NO de v5
MODEL_SAVE_PATH = "./modelos/ppo_panda_v6"
BEST_MODEL_DIR = "./modelos/best_v6/"

if __name__ == "__main__":
    env = make_vec_env(PandaPickPlaceEnv, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    eval_env = make_vec_env(PandaPickPlaceEnv, n_envs=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=max(10_000 // N_ENVS, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    model = PPO.load(MODELO_BASE, env=env, device="cpu")
    model.learning_rate = 5e-5  # 6x mas bajo que el default (3e-4), para fine-tuning conservador

    print(f"Fine-tuning conservador desde {MODELO_BASE}, lr=5e-5, {TIMESTEPS_ADICIONALES} steps...")
    model.learn(
        total_timesteps=TIMESTEPS_ADICIONALES,
        callback=eval_callback,
        progress_bar=True,
        reset_num_timesteps=False,
    )

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Modelo final guardado en {MODEL_SAVE_PATH}.zip")
    print(f"Mejor modelo guardado en {BEST_MODEL_DIR}best_model.zip")

    env.close()
    eval_env.close()
