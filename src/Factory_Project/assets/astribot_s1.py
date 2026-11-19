# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab articulation configuration for the mobile, dual-gripper Astribot S1."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from Factory_Project.sorting.kinematics import ARM_DAMPING, ARM_STIFFNESS

from . import ASTRIBOT_S1_USD

ASTRIBOT_S1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ASTRIBOT_S1_USD),
        activate_contact_sensors=False,
        # Never let the solver put the robot to sleep or "stabilise" slow motion away: either one
        # swallows chassis commands below about 0.06 m/s, and docking then stalls 1-2 cm short.
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(sleep_threshold=0.0, stabilization_threshold=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.09523), joint_pos={".*": 0.0}),
    actuators={
        # Wheel hubs take torque commands from the chassis velocity controller in the task; a little
        # damping keeps an unloaded wheel from spinning up. The 64 rollers stay passive.
        "wheels": ImplicitActuatorCfg(joint_names_expr=["wheel_.*_Joint"], stiffness=0.0, damping=0.3),
        "torso": ImplicitActuatorCfg(joint_names_expr=["astribot_torso_joint_.*"], stiffness=None, damping=None),
        "head": ImplicitActuatorCfg(joint_names_expr=["astribot_head_joint_.*"], stiffness=None, damping=None),
        # Position gains stiff enough to hold a two-handed carton clamp. These are simulation gains
        # [N·m/rad, N·m·s/rad]; effort limits stay at the URDF ratings.
        "arms": ImplicitActuatorCfg(
            joint_names_expr=["astribot_arm_.*_joint_.*"],
            stiffness={f".*_joint_{i + 1}": gain for i, gain in enumerate(ARM_STIFFNESS)},
            damping={f".*_joint_{i + 1}": gain for i, gain in enumerate(ARM_DAMPING)},
        ),
        "grippers": ImplicitActuatorCfg(
            joint_names_expr=["astribot_gripper_.*_joint_L1"], stiffness=None, damping=None
        ),
    },
)
"""S1 with the gains used by the sorting task. Set ``prim_path`` before use."""
