"""
Fase 2 - Script 5: Test unitario de la funcion de recompensa,
con posiciones sinteticas (sin fisica, sin robot).

Objetivo: validar que calcular_recompensa() se comporta como se
espera en casos concretos, aislado del problema de motion planning
(que es responsabilidad del agente de RL, no de este test).
"""
import numpy as np
from recompensa import calcular_recompensa

target_pos = np.array([0.3, 0.3, 0.02])

casos = [
    {
        'nombre': 'Lejos de todo, gripper abierto',
        'hand_pos': np.array([0.0, 0.0, 0.5]),
        'cube_pos': np.array([0.5, 0.0, 0.02]),
        'gripper_ctrl': 0,
    },
    {
        'nombre': 'Mano cerca del cubo, gripper abierto (no deberia dar grasping)',
        'hand_pos': np.array([0.51, 0.0, 0.03]),
        'cube_pos': np.array([0.5, 0.0, 0.02]),
        'gripper_ctrl': 0,
    },
    {
        'nombre': 'Mano cerca del cubo, gripper cerrado (SI deberia dar grasping)',
        'hand_pos': np.array([0.51, 0.0, 0.03]),
        'cube_pos': np.array([0.5, 0.0, 0.02]),
        'gripper_ctrl': 200,
    },
    {
        'nombre': 'Cubo en el target (exito)',
        'hand_pos': np.array([0.3, 0.3, 0.03]),
        'cube_pos': np.array([0.3, 0.3, 0.02]),
        'gripper_ctrl': 200,
    },
    {
        'nombre': 'Cubo fuera del area de trabajo (fallo)',
        'hand_pos': np.array([0.0, 0.0, 0.5]),
        'cube_pos': np.array([1.5, 1.5, 0.02]),
        'gripper_ctrl': 0,
    },
]

for caso in casos:
    reward, terminated, info = calcular_recompensa(
        caso['hand_pos'], caso['cube_pos'], target_pos, caso['gripper_ctrl']
    )
    print(f"--- {caso['nombre']} ---")
    print(f"  reward={reward:.3f} | terminated={terminated} | {info}")
    print()
