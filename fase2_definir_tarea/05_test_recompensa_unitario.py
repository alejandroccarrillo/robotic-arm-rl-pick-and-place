"""
Fase 2 - Script 5: Test unitario de calcular_recompensa(), actualizado
en Fase 4 a la firma con agarre_real (bool) y hubo_agarre_alguna_vez (bool)
en vez de gripper_ctrl.
"""
import numpy as np
from recompensa import calcular_recompensa

target_pos = np.array([0.3, 0.3, 0.02])

casos = [
    {
        'nombre': 'Lejos de todo, sin contacto',
        'hand_pos': np.array([0.0, 0.0, 0.5]),
        'cube_pos': np.array([0.5, 0.0, 0.02]),
        'agarre_real': False,
        'hubo_agarre_alguna_vez': False,
    },
    {
        'nombre': 'Contacto ahora mismo (deberia dar bonus grasping)',
        'hand_pos': np.array([0.51, 0.0, 0.03]),
        'cube_pos': np.array([0.5, 0.0, 0.02]),
        'agarre_real': True,
        'hubo_agarre_alguna_vez': True,
    },
    {
        'nombre': 'Cubo en target pero SIN haber agarrado nunca (NO deberia dar exito)',
        'hand_pos': np.array([0.3, 0.3, 0.03]),
        'cube_pos': np.array([0.3, 0.3, 0.02]),
        'agarre_real': False,
        'hubo_agarre_alguna_vez': False,
    },
    {
        'nombre': 'Cubo en target Y hubo agarre antes (SI deberia dar exito)',
        'hand_pos': np.array([0.3, 0.3, 0.03]),
        'cube_pos': np.array([0.3, 0.3, 0.02]),
        'agarre_real': False,
        'hubo_agarre_alguna_vez': True,
    },
    {
        'nombre': 'Cubo fuera del area de trabajo (fallo)',
        'hand_pos': np.array([0.0, 0.0, 0.5]),
        'cube_pos': np.array([1.5, 1.5, 0.02]),
        'agarre_real': False,
        'hubo_agarre_alguna_vez': False,
    },
]

for caso in casos:
    reward, terminated, info = calcular_recompensa(
        caso['hand_pos'], caso['cube_pos'], target_pos,
        caso['agarre_real'], caso['hubo_agarre_alguna_vez']
    )
    print(f"--- {caso['nombre']} ---")
    print(f"  reward={reward:.3f} | terminated={terminated} | {info}")
    print()
