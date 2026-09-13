"""
Fase 2: Funcion de recompensa para la tarea de pick-and-place.
Actualizada en Fase 4 tras detectar reward hacking (dos veces): el
"grasping" ya NO se infiere de proxies geometricos (proximidad +
estado del gripper, o altura del cubo), sino de contacto fisico real
entre AMBOS dedos y el cubo simultaneamente, calculado en el entorno
via data.contact de MuJoCo.

Ademas, el EXITO exige que haya habido agarre real en ALGUN momento
del episodio, no solo que el cubo termine cerca del target - esto
cierra la puerta a que el agente empuje/arrastre/voltee el cubo hasta
el target sin haberlo agarrado nunca.

Componentes:
  1. -distancia(hand, cubo): incentiva acercar el efector final al cubo
  2. bonus_grasping: recompensa extra en cada step con agarre_real=True
  3. -peso * distancia(cubo, target): incentiva llevar el cubo al destino
  4. bonus_exito: SOLO si dist_cube_target < umbral Y hubo_agarre_alguna_vez=True
  5. penalizacion_fallo: si el cubo sale del area de trabajo alcanzable
"""
import numpy as np

UMBRAL_EXITO = 0.05
LIMITE_AREA_TRABAJO = 1.0

PESO_DISTANCIA_TARGET = 2.0
BONUS_GRASPING = 0.3   # bajado de 3.0 en v7: acumulado sobre 300 steps competia con BONUS_EXITO, incentivando "granjear" el agarre en vez de completar la tarea
BONUS_EXITO = 10.0
PENALIZACION_FALLO = -10.0


def calcular_recompensa(hand_pos, cube_pos, target_pos, agarre_real, hubo_agarre_alguna_vez):
    """
    Args:
        hand_pos, cube_pos, target_pos: np.array (3,)
        agarre_real: bool, contacto fisico bilateral EN ESTE STEP
        hubo_agarre_alguna_vez: bool, True si agarre_real fue True en
            algun step anterior del episodio actual (se acumula en el
            entorno, no aqui, porque esta funcion no mantiene estado)

    Returns:
        reward: float
        terminated: bool
        info: dict
    """
    dist_hand_cube = np.linalg.norm(hand_pos - cube_pos)
    dist_cube_target = np.linalg.norm(cube_pos - target_pos)
    dist_cube_origen = np.linalg.norm(cube_pos[:2])

    reward = -dist_hand_cube
    reward -= PESO_DISTANCIA_TARGET * dist_cube_target

    if agarre_real:
        reward += BONUS_GRASPING

    terminated = False
    cerca_del_target = dist_cube_target < UMBRAL_EXITO
    exito = cerca_del_target and hubo_agarre_alguna_vez
    if exito:
        reward += BONUS_EXITO
        terminated = True

    fallo = dist_cube_origen > LIMITE_AREA_TRABAJO
    if fallo:
        reward += PENALIZACION_FALLO
        terminated = True

    info = {
        'dist_hand_cube': dist_hand_cube,
        'dist_cube_target': dist_cube_target,
        'grasping': agarre_real,
        'exito': exito,
        'fallo': fallo,
    }
    return reward, terminated, info
