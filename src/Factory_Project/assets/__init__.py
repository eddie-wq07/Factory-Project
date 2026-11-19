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

SORTING_TASK_USD = FACTORY_PROJECT_ASSETS_DIR / "environments" / "sorting_task.usda"
"""Warehouse, sorting rack and drop zones without the robot or a physics scene, for Isaac Lab to reference."""

SORTING_TASK_MANIFEST = FACTORY_PROJECT_ASSETS_DIR / "environments" / "sorting_task.json"
"""Items, drop zones and stock of the sorting task."""

NAV_MAP = FACTORY_PROJECT_ASSETS_DIR / "environments" / "nav" / "occupancy.npz"
"""Occupancy grid of the static warehouse."""
