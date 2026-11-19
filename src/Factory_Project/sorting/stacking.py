"""Putaway stacking rules for the drop pallets, as a picker would apply them on the floor.

* Stock stands in columns flush with the pallet's aisle edge, labelled face toward the aisle.
* A column never exceeds :attr:`TaskManifest.max_stack` units or :attr:`TaskManifest.max_stack_height`.
* Like goes on like: cartons on cartons, a crate only on a crate of its own family.
* Nothing overhangs the unit beneath it, and nothing heavier goes on something lighter.
* Loose bottles go inside the open tote, never on top of a stack.
* A started column is topped up before an empty one is opened; a full one is skipped.

All functions are simulator-free. Live object poses come in as box-frame :class:`Pose` values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Pose, wrap_angle
from .manifest import BoxObject, Column, Destination, TaskManifest

OVERHANG_TOLERANCE = 0.02
"""Largest footprint excess over the supporting unit, per side pair [m]."""
MASS_TOLERANCE = 0.5
"""A unit may be at most this much heavier than the one beneath it [kg]."""
BIN_FLOOR = 0.02
"""Assumed thickness of a tote floor [m]."""
COLUMN_HALF_WIDTH = 0.28
"""Lateral distance from a column's centre line within which a unit belongs to it [m]."""


class NoPlacementError(RuntimeError):
    """No column at the destination can legally take the item."""


@dataclass(frozen=True)
class Placement:
    """Where an item must come to rest."""

    address: str
    column: int
    tier: int
    """Zero-based height index of the item within its column once placed."""
    support: str | None
    """Name of the unit the item rests on, or ``None`` for the pallet deck."""
    pose: Pose
    """Goal box-frame pose in the world."""
    in_bin: bool = False


def can_stack(top: BoxObject, below: BoxObject | None) -> tuple[bool, str]:
    """Whether ``top`` may rest directly on ``below`` (``None`` is the pallet deck)."""
    if top.kind == "bottle":
        return False, "loose bottles go inside a tote"
    if below is None:
        return True, "deck"
    if below.kind != top.kind:
        return False, f"a {top.kind} does not stack on a {below.kind}"
    if top.kind == "crate" and top.family != below.family:
        return False, "crates only stack on their own family"
    if top.size[0] > below.size[0] + OVERHANG_TOLERANCE or top.size[1] > below.size[1] + OVERHANG_TOLERANCE:
        return False, "footprint would overhang the unit below"
    if top.mass > below.mass + MASS_TOLERANCE:
        return False, "heavier than the unit below"
    return True, "ok"


def column_centre(zone: Destination, column: Column, depth: float) -> np.ndarray:
    """Floor position of a unit of the given depth standing flush with the column's front line."""
    return np.array(column.front) - np.array(zone.normal) * depth / 2


def column_members(
    manifest: TaskManifest, zone: Destination, column: Column, poses: dict[str, Pose]
) -> list[tuple[str, BoxObject, Pose]]:
    """Units currently standing in a column, bottom first, judged from live box poses."""
    normal = np.array(zone.normal)
    tangent = np.array([-normal[1], normal[0]])
    members = []
    for name, unit in manifest.objects.items():
        pose = poses.get(name)
        if pose is None:
            continue
        relative = pose.position[:2] - np.array(column.front)
        behind = -float(relative @ normal)
        if abs(float(relative @ tangent)) > COLUMN_HALF_WIDTH or not -0.1 < behind < 1.05:
            continue
        if pose.position[2] - unit.size[2] / 2 < zone.deck_z - 0.03:
            continue  # on the floor or below the deck: not part of the stack
        members.append((name, unit, pose))
    members.sort(key=lambda member: member[2].position[2])
    return members


def stack_heights(manifest: TaskManifest, poses: dict[str, Pose]) -> dict[str, list[int]]:
    """Number of stacked units per column at every destination. A tote's contents do not count."""
    result = {}
    for zone in manifest.destinations:
        counts = []
        for column in zone.columns:
            members = column_members(manifest, zone, column, poses)
            counts.append(sum(1 for _, unit, _ in members if unit.kind != "bottle"))
        result[zone.address] = counts
    return result


def plan_placement(manifest: TaskManifest, address: str, poses: dict[str, Pose]) -> Placement:
    """Choose the column and resting pose for an item from the live state of its destination.

    Args:
        manifest: Task description.
        address: Address of the item to put away.
        poses: Live box-frame world poses of every simulated object, keyed like ``manifest.objects``.

    Raises:
        NoPlacementError: If every column is full or incompatible.
    """
    item = manifest.item(address)
    zone = manifest.destination(address)
    candidates = []
    reasons = []
    for index, column in enumerate(zone.columns):
        members = [m for m in column_members(manifest, zone, column, poses) if m[0] != address]
        stacked = [m for m in members if m[1].kind != "bottle"]
        if column.is_bin:
            if item.kind != "bottle" or not stacked:
                reasons.append(f"column {index}: tote reserved for loose bottles")
                continue
            name, tote, tote_pose = stacked[-1]
            floor = tote_pose.position[2] - tote.size[2] / 2 + BIN_FLOOR
            position = np.array([tote_pose.position[0], tote_pose.position[1], floor + item.size[2] / 2])
            pose = Pose.from_yaw(position, zone.face_yaw)
            candidates.append((0, index, Placement(address, index, len(stacked), name, pose, in_bin=True)))
            continue
        if len(stacked) >= manifest.max_stack:
            reasons.append(f"column {index}: already {len(stacked)} high")
            continue
        below = stacked[-1] if stacked else None
        allowed, why = can_stack(item, below[1] if below else None)
        if not allowed:
            reasons.append(f"column {index}: {why}")
            continue
        base = below[2].position[2] + below[1].size[2] / 2 if below else zone.deck_z
        if base + item.size[2] > manifest.max_stack_height + 0.01:
            reasons.append(f"column {index}: stack would stand {base + item.size[2]:.2f} m high")
            continue
        centre = column_centre(zone, column, item.size[0])
        if below is not None:
            # Sit square on the unit below, front faces flush, rather than on the nominal grid.
            front = below[2].position[:2] + np.array(zone.normal) * below[1].size[0] / 2
            centre = front - np.array(zone.normal) * item.size[0] / 2
        pose = Pose.from_yaw([centre[0], centre[1], base + item.size[2] / 2], zone.face_yaw)
        # Top up a started column before opening an empty one.
        candidates.append(
            (0 if stacked else 1, index, Placement(address, index, len(stacked), below[0] if below else None, pose))
        )
    if not candidates:
        raise NoPlacementError(f"{address}: " + "; ".join(reasons))
    return min(candidates, key=lambda candidate: candidate[:2])[2]


@dataclass(frozen=True)
class PlacementCheck:
    placed: bool
    position_error: float
    height_error: float
    yaw_error: float
    tilt: float


def check_placement(
    placement: Placement,
    pose: Pose,
    *,
    position_tolerance: float = 0.06,
    height_tolerance: float = 0.03,
    yaw_tolerance: float = 0.21,
    tilt_tolerance: float = 0.09,
) -> PlacementCheck:
    """Compare an item's live box pose with its planned resting pose.

    A bottle in a tote only has to be inside the tote and roughly upright or lying flat within it,
    so its tolerances are those of the tote's interior rather than of a stack.
    """
    offset = pose.position - placement.pose.position
    position_error = float(np.linalg.norm(offset[:2]))
    height_error = float(abs(offset[2]))
    yaw_error = abs(wrap_angle(pose.yaw - placement.pose.yaw))
    tilt = pose.tilt
    if placement.in_bin:
        placed = position_error < 0.2 and offset[2] > -0.16 and offset[2] < 0.05
    else:
        placed = (
            position_error < position_tolerance
            and height_error < height_tolerance
            and yaw_error < yaw_tolerance
            and tilt < tilt_tolerance
        )
    return PlacementCheck(placed, position_error, height_error, yaw_error, tilt)
