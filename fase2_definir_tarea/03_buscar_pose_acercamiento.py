"""
Fase 2 - Script 3: Buscar una pose que acerque la mano al cubo,
usando cinematica directa instantanea (mj_forward) en vez de
simular la convergencia de los actuadores.

mj_forward recalcula todas las posiciones (xpos, etc.) a partir de
qpos inmediatamente, sin necesidad de step() ni de que los actuadores
"lleguen" a su objetivo. Es la herramienta correcta para explorar
poses candidatas rapido.
"""
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('fase2_definir_tarea/escena_pick_and_place.xml')
data = mujoco.MjData(model)

hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
cube_pos_inicial = np.array([0.5, 0.0, 0.02])  # posicion conocida del cubo en reposo

# Candidatas: variamos sobre todo joint2 (hombro) y joint4 (codo),
# que son los que mas influyen en alcance (x) y altura (z)
candidatas = [
    [0.0, 0.3, 0.0, -1.8, 0.0, 2.0, 0.7],
    [0.0, 0.6, 0.0, -1.8, 0.0, 2.2, 0.7],
    [0.0, 0.8, 0.0, -2.2, 0.0, 2.6, 0.7],
    [0.0, 1.0, 0.0, -2.4, 0.0, 3.0, 0.7],
    [0.0, 0.9, 0.0, -2.6, 0.0, 3.2, 0.7],
    [0.0, 1.2, 0.0, -2.8, 0.0, 3.4, 0.7],
]

mejor_pose = None
mejor_dist = float('inf')

for pose in candidatas:
    data.qpos[:7] = pose
    mujoco.mj_forward(model, data)  # recalcula xpos sin simular dinamica

    hand_pos = data.xpos[hand_id]
    dist = np.linalg.norm(hand_pos - cube_pos_inicial)

    print(f'pose={pose} -> hand=({hand_pos[0]:.3f}, {hand_pos[1]:.3f}, {hand_pos[2]:.3f}) '
          f'dist_al_cubo={dist:.3f}')

    if dist < mejor_dist:
        mejor_dist = dist
        mejor_pose = pose

print()
print(f'Mejor pose encontrada: {mejor_pose} (distancia={mejor_dist:.3f})')
