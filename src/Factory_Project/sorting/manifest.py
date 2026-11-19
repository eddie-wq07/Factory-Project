"""The sorting task manifest: the six rack items, their drop zones and the stock already there.

The manifest is written by ``scripts/build_drop_zones.py`` next to the scene it describes. Everything
here is simulator-free. World coordinates are metres in the warehouse frame (X east, Y north, Z up).

Every handled object has a *box frame* at the centre of its collider's bounding box: +X through the
front face (the labelled face shown to the picker and to the aisle), +Y along the width, +Z up.
``size`` is (depth, width, height) in that frame.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import Pose, rotation_z

DATA_DIR = Path(__file__).resolve().parents[1] / "assets/data/environments"
MANIFEST_PATH = DATA_DIR / "sorting_task.json"


@dataclass(frozen=True)
class BoxObject:
    """A rigid object with a measured collider box."""

    prim: str
    """Prim path inside the saved factory scene (``/World/...``)."""
    kind: str
    """``carton``, ``crate`` or ``bottle``."""
    family: str
    mass: float
    """Mass [kg]."""
    size: tuple[float, float, float]
    """Depth, width and height of the collider box [m]."""
    box_offset: tuple[float, float, float]
    """Box-frame origin in the rigid-body frame [m]."""
    box_yaw: float
    """Yaw of the box frame in the rigid-body frame [rad]."""
    center: tuple[float, float, float]
    """Box centre in the saved scene [m]."""

    @property
    def body_to_box(self) -> Pose:
        return Pose(self.box_offset, rotation_z(self.box_yaw))

    def box_pose(self, body_pose: Pose) -> Pose:
        """Box frame in the world, given the rigid body's world pose."""
        return body_pose * self.body_to_box

    def body_pose(self, box_pose: Pose) -> Pose:
        """Rigid-body world pose that puts the box frame at ``box_pose``."""
        return box_pose * self.body_to_box.inverse()

    @property
    def footprint(self) -> tuple[float, float]:
        return self.size[0], self.size[1]


@dataclass(frozen=True)
class Item(BoxObject):
    """One of the six items waiting on the 'To be sorted.' rack."""

    address: str = ""
    """Destination address, which is also the item's identity (``A2`` ... ``Q4``)."""
    level: int = 0
    """Rack shelf level, 1 = lowest."""
    material: str = ""


@dataclass(frozen=True)
class Column:
    """A stacking position on a drop pallet."""

    offset: float
    front: tuple[float, float]
    """Point on the column's front line, already inset from the pallet edge [m]."""
    is_bin: bool
    """True when the top unit is an open tote that receives loose items."""
    stock: tuple[BoxObject, ...]


@dataclass(frozen=True)
class Destination:
    address: str
    aisle: int
    normal: tuple[float, float]
    """Unit vector from the pallet toward the docked robot."""
    edge: tuple[float, float]
    """Centre of the pallet's aisle-side edge [m]."""
    deck_z: float
    """Height of the pallet deck above the floor [m]."""
    pallets: tuple[dict, ...]
    columns: tuple[Column, ...]

    @property
    def face_yaw(self) -> float:
        """Yaw of a box whose front faces the aisle [rad]."""
        return math.atan2(self.normal[1], self.normal[0])

    @property
    def robot_yaw(self) -> float:
        """Heading of a robot docked square to the pallet [rad]."""
        return math.atan2(-self.normal[1], -self.normal[0])


@dataclass(frozen=True)
class Rack:
    position: tuple[float, float, float]
    front_beam_y: float
    normal: tuple[float, float]
    shelf_tops: tuple[float, ...]

    @property
    def robot_yaw(self) -> float:
        return math.atan2(-self.normal[1], -self.normal[0])


@dataclass(frozen=True)
class TaskManifest:
    max_stack: int
    max_stack_height: float
    front_inset: float
    spawn_position: tuple[float, float, float]
    spawn_yaw: float
    rack: Rack
    items: tuple[Item, ...]
    destinations: tuple[Destination, ...]

    def item(self, address: str) -> Item:
        return next(item for item in self.items if item.address == address)

    def destination(self, address: str) -> Destination:
        return next(zone for zone in self.destinations if zone.address == address)

    @property
    def order(self) -> tuple[str, ...]:
        """Processing order: strictly alphabetical by address."""
        return tuple(sorted(item.address for item in self.items))

    @property
    def objects(self) -> dict[str, BoxObject]:
        """Every simulated object keyed by a stable name (item address or stock id)."""
        result: dict[str, BoxObject] = {item.address: item for item in self.items}
        for zone in self.destinations:
            for index, column in enumerate(zone.columns):
                for tier, unit in enumerate(column.stock):
                    result[f"{zone.address}_c{index}_t{tier}"] = unit
        return result


def _box_fields(data: dict) -> dict:
    return {
        "prim": data["prim"],
        "kind": data["kind"],
        "family": data["family"],
        "mass": float(data["mass"]),
        "size": tuple(data["size"]),
        "box_offset": tuple(data["box_offset"]),
        "box_yaw": float(data["box_yaw"]),
        "center": tuple(data["center"]),
    }


def load_manifest(path: Path | str = MANIFEST_PATH) -> TaskManifest:
    data = json.loads(Path(path).read_text())
    items = tuple(
        Item(**_box_fields(entry), address=entry["address"], level=entry["level"], material=entry["material"])
        for entry in data["items"]
    )
    destinations = tuple(
        Destination(
            address=zone["address"],
            aisle=zone["aisle"],
            normal=tuple(zone["normal"]),
            edge=tuple(zone["edge"]),
            deck_z=zone["deck_z"],
            pallets=tuple(zone["pallets"]),
            columns=tuple(
                Column(
                    offset=column["offset"],
                    front=tuple(column["front"]),
                    is_bin=column["bin"],
                    stock=tuple(BoxObject(**_box_fields(unit)) for unit in column["stock"]),
                )
                for column in zone["columns"]
            ),
        )
        for zone in data["destinations"]
    )
    rack = data["rack"]
    manifest = TaskManifest(
        max_stack=data["max_stack"],
        max_stack_height=data.get("max_stack_height", 1.5),
        front_inset=data["front_inset"],
        spawn_position=tuple(data["spawn"]["position"]),
        spawn_yaw=data["spawn"]["yaw"],
        rack=Rack(tuple(rack["position"]), rack["front_beam_y"], tuple(rack["normal"]), tuple(rack["shelf_tops"])),
        items=items,
        destinations=destinations,
    )
    if {item.address for item in items} != {zone.address for zone in destinations}:
        raise ValueError("Every item needs exactly one destination with the same address")
    return manifest


def obstacle_rectangles(manifest: TaskManifest) -> list[tuple[np.ndarray, np.ndarray]]:
    """Axis-aligned (low, high) floor rectangles of the drop pallets, for the route planner."""
    rectangles = []
    for zone in manifest.destinations:
        for pallet in zone.pallets:
            centre, half = np.array(pallet["center"]), np.array(pallet["half_extents"])
            rectangles.append((centre - half, centre + half))
    return rectangles
