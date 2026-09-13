"""
Fase 4 - Script 11: Entrenamiento largo desde best_v4 con
BONUS_GRASPING subido (0.5 -> 3.0) para incentivar SOSTENER el
agarre, no solo lograrlo brevemente. Learning rate normal (no
conservador como en v6) para permitir mas exploracion real.
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
TIMESTEPS_ADICIONALES = 1_500_000
LOG_DIR = "./tensorboard_logs/"
MODELO_BASE = "./modelos/best_v7/best_model.zip"
MODEL_SAVE_PATH = "./modelos/ppo_panda_v8"
BEST_MODEL_DIR = "./modelos/best_v8/"

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

    # learning_rate por defecto (3e-4): recompensa cambio (bonus grasping
    # subido), asi que interesa permitir mas adaptacion, no menos
    model = PPO.load(MODELO_BASE, env=env, device="cpu")

    print(f"Entrenando {TIMESTEPS_ADICIONALES} steps desde {MODELO_BASE}, BONUS_GRASPING=3.0...")
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
