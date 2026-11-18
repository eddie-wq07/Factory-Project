# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Paths and configurations for project-owned assets."""

from pathlib import Path

FACTORY_PROJECT_ASSETS_DIR = Path(__file__).resolve().parent / "data"
"""Path to project-owned asset data."""

ASTRIBOT_S1_USD = FACTORY_PROJECT_ASSETS_DIR / "astribot_s1" / "astribot_s1.usd"
"""Loadable USD for the mobile, dual-gripper Astribot S1."""

FACTORY_SCENE_USD = FACTORY_PROJECT_ASSETS_DIR / "environments" / "factory.usda"
"""Persistent warehouse, robot spawn, boundary and inspection camera."""
