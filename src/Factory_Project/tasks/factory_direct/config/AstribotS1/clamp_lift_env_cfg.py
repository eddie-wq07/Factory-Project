"""Configuration of the clamp-and-lift skill task for reinforcement learning."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass, replace

from Factory_Project.assets.astribot_s1 import ASTRIBOT_S1_CFG

CARTON_SIZE = (0.5, 0.5, 0.25)
"""Depth, width, height [m]: the Q4 carton of the sorting mission."""
SHELF_TOP = 0.623
"""Height of the lowest sorting-rack deck [m]."""
DOCK = 0.41
"""Distance from the chassis centre to the carton's front face [m], as the teacher docks."""
ARM_ACTIONS = 14
OBSERVATION_DIM = 14 + 14 + 6 + 3 + 3 + 3 + ARM_ACTIONS


@configclass
class ClampLiftSceneCfg(InteractiveSceneCfg):
    """A docked S1, one shelf deck and one carton per environment."""

    robot: ArticulationCfg = replace(
        ASTRIBOT_S1_CFG,
        prim_path="/World/envs/env_.*/Robot",
        spawn=replace(
            ASTRIBOT_S1_CFG.spawn,
            # The skill starts docked: the chassis is held where the navigation stack parked it.
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True, sleep_threshold=0.0, stabilization_threshold=0.0
            ),
        ),
    )
    shelf: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Shelf",
        spawn=sim_utils.CuboidCfg(
            size=(0.62, 1.4, 0.03),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.4),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.22)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(DOCK - 0.03 + 0.31, 0.0, SHELF_TOP - 0.015)),
    )
    carton: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Carton",
        spawn=sim_utils.CuboidCfg(
            size=CARTON_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.7, dynamic_friction=0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.45, 0.28)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(DOCK + CARTON_SIZE[0] / 2, 0.0, SHELF_TOP + CARTON_SIZE[2] / 2 + 0.002)
        ),
    )


@configclass
class ClampLiftEnvCfg(DirectRLEnvCfg):
    """Clamp a shelved carton between both palms and lift it clear of the deck.

    This is the first of the short manipulation skills in docs/policy-arch.md, posed for PPO: no
    demonstrations, a dense shaped reward, and the scripted teacher's pre-grasp as the start state.
    """

    decimation: int = 6
    episode_length_s: float = 6.0
    action_space: int = ARM_ACTIONS
    observation_space: int = OBSERVATION_DIM
    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(
        device="cpu",
        dt=1.0 / 120.0,
        render_interval=6,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7, restitution=0.0),
    )
    scene: ClampLiftSceneCfg = ClampLiftSceneCfg(num_envs=16, env_spacing=3.0, replicate_physics=True)

    action_scale: float = 0.03
    """Largest change of an arm joint target per control step [rad]."""
    carton_xy_noise: float = 0.02
    """Carton start offset on the shelf [m]."""
    carton_yaw_noise: float = 0.05
    """Carton start twist [rad]."""
    joint_noise: float = 0.03
    """Arm start offset from the teacher's pre-grasp [rad]."""
    lift_goal: float = 0.04
    """Lift that counts as clear of the deck [m]."""

    reach_weight: float = 1.0
    squeeze_weight: float = 1.0
    lift_weight: float = 6.0
    success_bonus: float = 4.0
    tilt_weight: float = 1.0
    push_weight: float = 4.0
    action_rate_weight: float = 0.01
