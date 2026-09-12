"""
Fase 3 - Script 1: Validar que PandaPickPlaceEnv cumple el contrato
basico de Gymnasium (formas correctas, no explota con acciones random).
"""
import numpy as np
from panda_pick_place_env import PandaPickPlaceEnv

env = PandaPickPlaceEnv()

print("observation_space:", env.observation_space.shape)
print("action_space:", env.action_space.shape)
print()

obs, info = env.reset()
print(f"Reset OK. obs.shape={obs.shape} (esperado: {env.observation_space.shape})")
assert obs.shape == env.observation_space.shape, "Shape de observacion no coincide!"

print()
print("Ejecutando 50 steps con acciones aleatorias...")
for i in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if i % 10 == 0:
        print(f"  step={i} | reward={reward:.3f} | terminated={terminated} | "
              f"truncated={truncated} | dist_cube_target={info['dist_cube_target']:.3f}")

    if terminated or truncated:
        print(f"  Episodio termino en step {i} (terminated={terminated}, truncated={truncated})")
        obs, info = env.reset()

print()
print("OK: validacion basica completada sin errores")
env.close()
