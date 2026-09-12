"""
Fase 4 - Script 2: Entrenamiento PPO en CPU, mas largo, con callback
de evaluacion periodica que guarda el mejor modelo y registra metricas
de exito/fallo explicitas en TensorBoard.
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
TOTAL_TIMESTEPS = 500_000
LOG_DIR = "./tensorboard_logs/"
MODEL_SAVE_PATH = "./modelos/ppo_panda_v2"
BEST_MODEL_DIR = "./modelos/best_v2/"

if __name__ == "__main__":
    env = make_vec_env(PandaPickPlaceEnv, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    # Entorno separado para evaluacion, no mezclado con el entrenamiento
    eval_env = make_vec_env(PandaPickPlaceEnv, n_envs=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=max(10_000 // N_ENVS, 1),  # cada ~10k steps totales
        n_eval_episodes=10,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="cpu",  # MlpPolicy rinde peor en GPU, confirmado en el experimento anterior
    )

    print(f"Entrenando PPO durante {TOTAL_TIMESTEPS} timesteps en CPU con {N_ENVS} entornos...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, progress_bar=True)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Modelo final guardado en {MODEL_SAVE_PATH}.zip")
    print(f"Mejor modelo guardado en {BEST_MODEL_DIR}best_model.zip")

    env.close()
    eval_env.close()
