# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the full six-item sorting mission in the warehouse."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass, replace
from isaaclab_physx.physics import PhysxCfg

from Factory_Project.assets.astribot_s1 import ASTRIBOT_S1_CFG
from Factory_Project.sorting.manifest import load_manifest

ROBOT_PRIM_PATH = "/World/Robot"
SITE_PRIM_PATH = "/World/Site"
"""Where the saved warehouse scene is referenced; its ``/World/...`` paths live beneath this prim."""

_MANIFEST = load_manifest()

ACTION_DIM = 25
"""Base twist (3) + torso (4) + head (2) + left arm (7) + right arm (7) + grippers (2)."""
OBSERVATION_DIM = 78
"""See :meth:`FactoryEnv._get_observations` for the layout."""


@configclass
class FactorySceneCfg(InteractiveSceneCfg):
    """One S1 at the persisted spawn.

    The warehouse, sorting rack and drop zones are referenced by the environment itself (see
    :meth:`FactoryEnv._setup_scene`). Isaac Lab's file spawner would mirror every remote warehouse
    asset and re-validate each copy against the server on every launch, which takes many minutes;
    a plain USD reference lets Kit's resolver use its own cache.
    """

    robot: ArticulationCfg = replace(
        ASTRIBOT_S1_CFG,
        # A global path: this task has one robot in one warehouse, and RTX does not draw the robot
        # when it sits inside the cloned per-environment namespace next to the referenced site.
        prim_path=ROBOT_PRIM_PATH,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=_MANIFEST.spawn_position,
            rot=(0.0, 0.0, math.sin(_MANIFEST.spawn_yaw / 2), math.cos(_MANIFEST.spawn_yaw / 2)),
            joint_pos={".*": 0.0},
        ),
    )


@configclass
class FactoryEnvCfg(DirectRLEnvCfg):
    """One robot sorts six items from the 'To be sorted.' rack to their addressed drop zones."""

    # 120 Hz physics, 20 Hz control: the action rate proposed in docs/policy-arch.md.
    decimation: int = 6
    episode_length_s: float = 1800.0
    action_space: int = ACTION_DIM
    observation_space: int = OBSERVATION_DIM
    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(
        device="cpu",
        dt=1.0 / 120.0,
        render_interval=6,
        physics=PhysxCfg(
            solver_type=1,
            bounce_threshold_velocity=0.2,
            enable_stabilization=False,
            friction_correlation_distance=0.00625,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7, restitution=0.0),
    )
    scene: FactorySceneCfg = FactorySceneCfg(num_envs=1, env_spacing=1.0, replicate_physics=False)

    # -- base ---------------------------------------------------------------------------------------
    base_speed_limit: tuple[float, float, float] = (1.0, 0.6, 1.2)
    """Largest commanded forward [m/s], lateral [m/s] and yaw [rad/s] chassis speed."""
    base_velocity_gain: tuple[float, float, float] = (700.0, 700.0, 60.0)
    """Proportional gains of the chassis velocity loop [N per m/s, N per m/s, N·m per rad/s]."""
    base_velocity_integral_time: float = 0.25
    """Integral time of the chassis velocity loop [s]. The integral builds the push that rolls the
    wheels out of the small dips between roller contacts, where a slow command would otherwise stall."""
    base_force_limit: tuple[float, float, float] = (400.0, 400.0, 90.0)
    """Largest commanded net chassis force [N, N] and yaw moment [N·m]."""
    wheel_radius: float = 0.0802
    """Rolling radius of the omni wheels [m]."""
    wheel_offset: float = 0.3024
    """Distance from the chassis centre to each wheel centre [m]."""
    wheel_torque_limit: float = 30.0
    """Hub torque rating [N·m]."""

    # -- task checks --------------------------------------------------------------------------------
    settle_linear_speed: float = 0.05
    """An item counts as at rest below this speed [m/s]."""
    settle_angular_speed: float = 0.3
    """An item counts as at rest below this angular speed [rad/s]."""
    release_clearance: float = 0.05
    """Both palms must be at least this far outside the item's sides to count as released [m]."""
    drop_tilt: float = 0.7
    """An item tipped beyond this angle is counted as dropped [rad]."""
    floor_height: float = 0.06
    """An item whose underside is below this height off its supports has fallen to the floor [m]."""
    terminate_on_drop: bool = True

    # -- reset variation -----------------------------------------------------------------------------
    reset_item_xy_noise: float = 0.0
    """Each rack item starts up to this far left or right of its slot centre [m]."""
    reset_item_yaw_noise: float = 0.0
    """Each rack item starts twisted by up to this angle about the vertical [rad]."""

    # -- rewards ------------------------------------------------------------------------------------
    delivery_reward: float = 1.0
    drop_penalty: float = -1.0

    # -- cameras ------------------------------------------------------------------------------------
    enable_cameras: bool = False
    """Attach the head and wrist RGB cameras and add their images to the observations."""
    camera_resolution: tuple[int, int] = (160, 120)
    """Width and height of each onboard camera image [px]."""
    viewer_cameras: dict[str, tuple[int, int]] = {}
    """Free cameras for recordings and inspection: name -> (width, height) [px]. See :meth:`FactoryEnv.view`."""
