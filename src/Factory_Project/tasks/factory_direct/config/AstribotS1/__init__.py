# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="FactoryProject-Factory-ASTRIBOTS1-Direct",
    entry_point=f"{__name__}.env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FactoryEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:DistillationRunnerCfg",
    },
)

gym.register(
    id="FactoryProject-ClampLift-ASTRIBOTS1-Direct",
    entry_point=f"{__name__}.clamp_lift_env:ClampLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.clamp_lift_env_cfg:ClampLiftEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_clamp_lift_ppo_cfg:ClampLiftPPORunnerCfg",
    },
)
