"""
Fase 2: Funcion de recompensa para la tarea de pick-and-place.

Diseno: recompensa densa (no dispersa) para dar señal de gradiente
desde el primer step, en vez de esperar a que el agente choque con
el cubo por pura casualidad.

Componentes:
  1. -distancia(hand, cubo): incentiva acercar el efector final al cubo
  2. bonus_grasping: recompensa extra si el gripper esta cerrado Y cerca del cubo
  3. -peso * distancia(cubo, target): incentiva llevar el cubo al destino
  4. bonus_exito: recompensa grande si el cubo llega al target (fin de episodio)
  5. penalizacion_fallo: si el cubo sale del area de trabajo alcanzable (fin de episodio)
"""
import numpy as np

# Umbrales y pesos (ajustables durante el entrenamiento en Fase 4 si hace falta)
UMBRAL_GRASPING = 0.05      # metros: "cerca del cubo" para contar como agarre
UMBRAL_EXITO = 0.05         # metros: "cubo en el target"
GRIPPER_CERRADO_MIN = 100   # valor de ctrl[7] (rango 0-255) a partir del cual se considera "cerrado"
LIMITE_AREA_TRABAJO = 1.0   # metros: distancia maxima del cubo al origen antes de dar por perdido el episodio

PESO_DISTANCIA_TARGET = 2.0  # esta parte de la tarea pesa mas que solo acercarse
BONUS_GRASPING = 0.5
BONUS_EXITO = 10.0
PENALIZACION_FALLO = -10.0


def calcular_recompensa(hand_pos, cube_pos, target_pos, gripper_ctrl):
    """
    Calcula la recompensa del step actual y si el episodio debe terminar.

    Args:
        hand_pos: np.array (3,) posicion cartesiana del efector final
        cube_pos: np.array (3,) posicion cartesiana del cubo
        target_pos: np.array (3,) posicion cartesiana del target
        gripper_ctrl: valor actual de data.ctrl[7] (0-255)

    Returns:
        reward: float
        terminated: bool (True si el episodio debe acabar, exito o fallo)
        info: dict con detalles para debug
    """
    dist_hand_cube = np.linalg.norm(hand_pos - cube_pos)
    dist_cube_target = np.linalg.norm(cube_pos - target_pos)
    dist_cube_origen = np.linalg.norm(cube_pos[:2])  # solo x,y: distancia horizontal

    reward = -dist_hand_cube
    reward -= PESO_DISTANCIA_TARGET * dist_cube_target

    gripper_cerrado = gripper_ctrl > GRIPPER_CERRADO_MIN
    grasping = dist_hand_cube < UMBRAL_GRASPING and gripper_cerrado
    if grasping:
        reward += BONUS_GRASPING

    terminated = False
    exito = dist_cube_target < UMBRAL_EXITO
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
        'grasping': grasping,
        'exito': exito,
        'fallo': fallo,
    }
    return reward, terminated, info
