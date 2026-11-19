# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the project's task registrations."""

import gymnasium as gym

import Factory_Project.tasks  # noqa: F401


def test_task_registrations():
    """The project tasks must expose valid environment and agent entry points."""
    expected = {
        "FactoryProject-Factory-ASTRIBOTS1": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": "Factory_Project.tasks.factory.config.AstribotS1.env_cfg:FactoryEnvCfg",
            "default_agent": "rsl_rl",
        },
        "FactoryProject-Factory-ASTRIBOTS1-Direct": {
            "entry_point": "Factory_Project.tasks.factory_direct.config.AstribotS1.env:FactoryEnv",
            "env_cfg_entry_point": "Factory_Project.tasks.factory_direct.config.AstribotS1.env_cfg:FactoryEnvCfg",
            "default_agent": "rsl_rl",
        },
        "FactoryProject-ClampLift-ASTRIBOTS1-Direct": {
            "entry_point": "Factory_Project.tasks.factory_direct.config.AstribotS1.clamp_lift_env:ClampLiftEnv",
            "env_cfg_entry_point": "Factory_Project.tasks.factory_direct.config.AstribotS1.clamp_lift_env_cfg:ClampLiftEnvCfg",
            "default_agent": "rsl_rl",
        },
        "FactoryProject-Factory-Marl-ASTRIBOTS1-Direct": {
            "entry_point": "Factory_Project.tasks.factory_marl_direct.config.AstribotS1.env:FactoryMarlEnv",
            "env_cfg_entry_point": "Factory_Project.tasks.factory_marl_direct.config.AstribotS1.env_cfg:FactoryMarlEnvCfg",
        },
    }

    for task_id, expected_values in expected.items():
        spec = gym.spec(task_id)
        assert spec.entry_point == expected_values["entry_point"]
        assert spec.kwargs["env_cfg_entry_point"] == expected_values["env_cfg_entry_point"]
        if "default_agent" in expected_values:
            assert spec.kwargs["default_agent"] == expected_values["default_agent"]
