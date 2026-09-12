"""
Fase 4 - Script 3: Evaluacion visual del mejor modelo entrenado.

Carga best_model.zip y corre unos episodios con el viewer abierto,
en modo deterministico, para confirmar visualmente si el agente
esta resolviendo la tarea de verdad (acercarse, agarrar, llevar al
target) o si ha encontrado algun comportamiento degenerado que
maximiza reward sin resolver la tarea real.
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fase3_entorno_gym'))

from stable_baselines3 import PPO
from panda_pick_place_env import PandaPickPlaceEnv

MODEL_PATH = "./modelos/best_v2/best_model.zip"
N_EPISODIOS = 3

env = PandaPickPlaceEnv(render_mode="human")
model = PPO.load(MODEL_PATH)

for ep in range(N_EPISODIOS):
    obs, info = env.reset()
    env.render()
    terminated = truncated = False
    ep_reward = 0
    step = 0

    print(f"\n--- Episodio {ep+1} ---")

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        step += 1
        env.render()
        time.sleep(0.02)

        if step % 50 == 0:
            print(f"  step={step} | dist_cube_target={info['dist_cube_target']:.3f} | "
                  f"dist_hand_cube={info['dist_hand_cube']:.3f}")

    resultado = "EXITO" if info.get('exito') else ("FALLO" if info.get('fallo') else "TRUNCADO")
    print(f"  Episodio {ep+1} terminado: {resultado} | reward_total={ep_reward:.1f} | steps={step}")

env.close()
