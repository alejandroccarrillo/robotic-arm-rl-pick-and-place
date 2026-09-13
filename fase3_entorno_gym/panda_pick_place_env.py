"""
Fase 3: Entorno Gymnasium para la tarea de pick-and-place con el Panda.

Envuelve la escena MuJoCo (fase2) y la funcion de recompensa (fase2)
en la interfaz estandar de Gymnasium: reset() y step().

Actualizado en Fase 4: deteccion de agarre real via contactos fisicos
de MuJoCo (data.contact) en vez de proxies geometricos explotables.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fase2_definir_tarea'))
from recompensa import calcular_recompensa

# Geoms de colision real de los dedos (contype=1), identificados en Fase 4
# via inspeccion de model.geom_contype. Los geoms con contype=0 son solo
# visuales y nunca generan contactos, se excluyen deliberadamente.
LEFT_FINGER_COLLISION_GEOMS = frozenset(range(68, 74))   # 68-73
RIGHT_FINGER_COLLISION_GEOMS = frozenset(range(76, 82))  # 76-81
CUBE_GEOM_ID = 82

HOME_QPOS_BRAZO = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
HOME_QPOS_DEDOS = np.array([0.04, 0.04])
HOME_CTRL = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255], dtype=np.float64)

CUBE_POS_INICIAL = np.array([0.5, 0.0, 0.02])
CUBE_QUAT_INICIAL = np.array([1.0, 0.0, 0.0, 0.0])

MAX_DELTA_BRAZO = 0.05
MAX_DELTA_GRIPPER = 25.0

FRAME_SKIP = 5
MAX_STEPS = 300


class PandaPickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human", None]}

    def __init__(self, render_mode=None):
        super().__init__()

        xml_path = os.path.join(
            os.path.dirname(__file__), '..', 'fase2_definir_tarea', 'escena_pick_and_place.xml'
        )
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
        self.cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
        self.target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'target')

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        obs_dim = 7 + 1 + 3 + 3 + 3 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.current_ctrl = HOME_CTRL.copy()
        self.render_mode = render_mode
        self.viewer = None
        self.step_count = 0
        self.hubo_agarre = False

    def _hay_contacto(self, finger_geoms, cube_geom):
        """Recorre data.contact (contactos activos en el ultimo mj_step)
        y comprueba si alguno involucra un geom del set de dedo dado y
        el geom del cubo, en cualquier orden (geom1/geom2)."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 in finger_geoms and c.geom2 == cube_geom) or \
               (c.geom2 in finger_geoms and c.geom1 == cube_geom):
                return True
        return False

    def _get_obs(self):
        hand_pos = self.data.xpos[self.hand_id].copy()
        cube_pos = self.data.xpos[self.cube_id].copy()
        target_pos = self.data.site_xpos[self.target_id].copy()

        obs = np.concatenate([
            self.data.qpos[0:7],
            [self.current_ctrl[7]],
            hand_pos,
            cube_pos,
            target_pos,
            hand_pos - cube_pos,
            cube_pos - target_pos,
        ]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[0:7] = HOME_QPOS_BRAZO
        self.data.qpos[7:9] = HOME_QPOS_DEDOS
        self.data.qpos[9:12] = CUBE_POS_INICIAL
        self.data.qpos[12:16] = CUBE_QUAT_INICIAL
        self.data.qvel[:] = 0.0

        self.current_ctrl = HOME_CTRL.copy()
        self.data.ctrl[:] = self.current_ctrl

        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.hubo_agarre = False

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        delta_brazo = action[:7] * MAX_DELTA_BRAZO
        delta_gripper = action[7] * MAX_DELTA_GRIPPER

        self.current_ctrl[:7] = np.clip(
            self.current_ctrl[:7] + delta_brazo,
            self.model.actuator_ctrlrange[:7, 0],
            self.model.actuator_ctrlrange[:7, 1],
        )
        self.current_ctrl[7] = np.clip(self.current_ctrl[7] + delta_gripper, 0, 255)

        self.data.ctrl[:] = self.current_ctrl

        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        contacto_izq = self._hay_contacto(LEFT_FINGER_COLLISION_GEOMS, CUBE_GEOM_ID)
        contacto_der = self._hay_contacto(RIGHT_FINGER_COLLISION_GEOMS, CUBE_GEOM_ID)
        agarre_real = contacto_izq and contacto_der

        if agarre_real:
            self.hubo_agarre = True

        hand_pos = self.data.xpos[self.hand_id]
        cube_pos = self.data.xpos[self.cube_id]
        target_pos = self.data.site_xpos[self.target_id]

        reward, terminated, info = calcular_recompensa(
            hand_pos, cube_pos, target_pos, agarre_real, self.hubo_agarre
        )

        truncated = self.step_count >= MAX_STEPS

        obs = self._get_obs()

        if self.render_mode == "human" and self.viewer is not None:
            self.viewer.sync()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
