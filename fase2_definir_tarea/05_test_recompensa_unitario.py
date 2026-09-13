"""
Fase 2 - Script 5: Test unitario de la funcion de recompensa,
con posiciones sinteticas (sin fisica, sin robot).

NOTA: gripper_ctrl=255 es ABIERTO y valores bajos (~0-20) son CERRADO
en este modelo, segun confirmamos empiricamente en Fase 1. No asumir
la relacion contraria.
"""
import numpy as np
from recompensa import calcular_recompensa

target_pos = np.array([0.3, 0.3, 0.02])

casos = [
    {
        'nombre': 'Lejos de todo, gripper abierto',
        'hand_pos': np.array([0.0, 0.0, 0.5]),
        'cube_pos': np.array([0.5, 0.0, 0.02]),
        'gripper_ctrl': 255,
    },
    {
        'nombre': 'Mano cerca del cubo, gripper abierto (no deberia dar grasping)',
        'hand_pos': np.array([0.51, 0.0, 0.03]),
        'cube_pos': np.array([0.5, 0.0, 0.02]),
        'gripper_ctrl': 255,
    },
    {
        'nombre': 'Mano cerca del cubo, gripper cerrado (SI deberia dar grasping)',
        'hand_pos': np.array([0.51, 0.0, 0.03]),
        'cube_pos': np.array([0.5, 0.0, 0.02]),
        'gripper_ctrl': 20,
    },
    {
        'nombre': 'Cubo en el target (exito)',
        'hand_pos': np.array([0.3, 0.3, 0.03]),
        'cube_pos': np.array([0.3, 0.3, 0.02]),
        'gripper_ctrl': 20,
    },
    {
        'nombre': 'Cubo fuera del area de trabajo (fallo)',
        'hand_pos': np.array([0.0, 0.0, 0.5]),
        'cube_pos': np.array([1.5, 1.5, 0.02]),
        'gripper_ctrl': 255,
    },
]

for caso in casos:
    reward, terminated, info = calcular_recompensa(
        caso['hand_pos'], caso['cube_pos'], target_pos, caso['gripper_ctrl']
    )
    print(f"--- {caso['nombre']} ---")
    print(f"  reward={reward:.3f} | terminated={terminated} | {info}")
    print()
