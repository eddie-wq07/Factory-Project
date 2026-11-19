"""Simulator-free tests of the sorting task's rules, planning and learning code."""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from Factory_Project.learning.dataset import SKILLS, chunk_windows, condition
from Factory_Project.learning.policies import build_policy
from Factory_Project.sorting.geometry import Pose
from Factory_Project.sorting.manifest import load_manifest, obstacle_rectangles
from Factory_Project.sorting.navigation import OccupancyMap, RoutePlanner, path_length
from Factory_Project.sorting.stacking import can_stack, check_placement, plan_placement, stack_heights


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


@pytest.fixture(scope="module")
def poses(manifest):
    """Every object where the saved scene puts it."""

    def yaw(name):
        return manifest.destination(name.split("_")[0]).face_yaw if "_" in name else -math.pi / 2

    return {name: Pose.from_yaw(unit.center, yaw(name)) for name, unit in manifest.objects.items()}


def test_mission_is_six_items_in_alphabetical_order(manifest):
    assert manifest.order == ("A2", "F8", "H4", "K3", "O8", "Q4")
    assert list(manifest.order) == sorted(manifest.order)


def test_rack_cartons_fit_the_two_handed_clamp(manifest):
    for address in manifest.order:
        item = manifest.item(address)
        if item.kind == "carton":
            assert item.size[1] <= 0.5 + 1e-3, f"{address} is too wide to clamp"
        assert item.mass <= 3.5


def test_every_item_has_a_legal_placement(manifest, poses):
    heights = stack_heights(manifest, poses)
    for address in manifest.order:
        item = manifest.item(address)
        placement = plan_placement(manifest, address, poses)
        if item.kind == "bottle":
            assert placement.in_bin, "loose bottles go into a tote"
            continue
        assert placement.tier < manifest.max_stack
        top = placement.pose.position[2] + item.size[2] / 2
        assert top <= manifest.max_stack_height + 0.01
        assert check_placement(placement, placement.pose).placed
    assert max(max(columns) for columns in heights.values()) <= manifest.max_stack


def test_started_column_is_topped_up_first(manifest, poses):
    # A2's pallet has a one-high and a two-high column of 0.5 m cartons: a third would pass 1.5 m.
    placement = plan_placement(manifest, "A2", poses)
    assert placement.tier == 1


def test_stacking_rules(manifest):
    carton, crate, bottle = manifest.item("A2"), manifest.item("F8"), manifest.item("O8")
    assert can_stack(carton, None)[0]
    assert can_stack(carton, carton)[0]
    assert not can_stack(carton, crate)[0], "like stacks on like"
    assert not can_stack(bottle, carton)[0], "bottles never stack"
    assert not can_stack(replace(carton, mass=carton.mass + 2.0), carton)[0], "heavier never goes on lighter"
    wider = replace(carton, size=(carton.size[0], carton.size[1] + 0.2, carton.size[2]))
    assert not can_stack(wider, carton)[0], "no overhang"
    assert not can_stack(manifest.item("H4"), crate)[0], "crates stack only on their own family"


def test_misplaced_item_is_rejected(manifest, poses):
    placement = plan_placement(manifest, "Q4", poses)
    shifted = Pose.from_yaw(placement.pose.position + np.array([0.12, 0.0, 0.0]), placement.pose.yaw)
    assert not check_placement(placement, shifted).placed


def test_route_from_spawn_to_every_drop_zone(manifest):
    occupancy = OccupancyMap.load(rectangles=obstacle_rectangles(manifest))
    planner = RoutePlanner(occupancy, radius=0.38)
    rack_front = np.array([manifest.rack.position[0], manifest.rack.front_beam_y - 1.0])
    assert path_length(planner.plan(manifest.spawn_position[:2], rack_front)) > 1.0
    for address in manifest.order:
        zone = manifest.destination(address)
        stand = np.array(zone.edge) + np.array(zone.normal) * 1.0
        route = planner.plan(rack_front, stand)
        assert np.linalg.norm(route[-1] - stand) < 0.5


def test_chunk_windows_and_policies():
    steps, horizon = 50, 8
    episode = dict(
        obs=np.random.randn(steps, 78).astype(np.float32),
        actions=np.random.randn(steps, 25).astype(np.float32),
        skill=np.array([SKILLS.index("pick")] * 30 + [SKILLS.index("retreat")] * 20),
    )
    inputs, chunks, masks = chunk_windows([episode], horizon)
    assert inputs.shape == (30, 78 + len(SKILLS)) and chunks.shape == (30, horizon, 25)
    assert masks[0].sum() == horizon and np.allclose(chunks[0], episode["actions"][:horizon])
    assert np.allclose(condition(episode["obs"], episode["skill"])[0, 78:], np.eye(len(SKILLS))[1])
    x = torch.as_tensor(inputs[:4])
    assert build_policy("mlp", inputs.shape[1], 22, horizon)(x).shape == (4, 1, 22)
    assert build_policy("act", inputs.shape[1], 22, horizon)(x).shape == (4, horizon, 22)
