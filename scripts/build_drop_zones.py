"""Persist the six designated drop spots and the task manifest read by the sorting environment.

Run after ``build_sorting_rack.py`` and ``build_nav_map.py`` with the Isaac Sim Python environment::

    ~/env_isaaclab/bin/python scripts/build_drop_zones.py

Each address gets a floor-marked pallet position beside its rack face, already holding stock the way
a live putaway location does. Stock is stacked in columns flush with the pallet's aisle edge, labels
facing the aisle, never more than three high. The robot's item must go onto the first column that can
still take it, so two zones start with a full column that must be skipped. Address O8 lies inside the
coned-off spill area of aisle 5, so its drop spot is the end-cap position outside the cordon, where
loose bottles go into a tote.

Outputs: ``drop_zones.usda`` (sublayer of ``factory.usda``), ``sorting_task.usda`` (the scene without
the robot and physics scene, for Isaac Lab to reference) and ``sorting_task.json`` (the manifest).
"""

import json
import math

import numpy as np
from build_sorting_rack import (
    ENV,
    FRONT_BEAM_Y,
    FRONT_INSET,
    RACK_POSITION,
    SETTLE_GAP,
    SHELF_TOPS,
    body_box,
    make_dynamic,
    place_reference,
    placement_matrix,
)

PALLET = (1.0, 1.2, 0.144)  # depth toward the rack, length along the aisle, height [m]
MAX_STACK = 3
MAX_STACK_HEIGHT = 1.5  # tallest stack above the floor [m]: nothing is stacked above shoulder height
SPAWN = {"position": [4.45, 7.3, 0.09523], "yaw": math.pi}

DESTINATIONS = {
    # normal: unit vector from the pallet toward the docked robot. edge: centre of the pallet's aisle
    # edge. columns: (offset along the edge, number of stock units already stacked there).
    "A2": dict(normal=(1, 0), edge=(-23.55, 14.9), pallets=(0.0,), columns=((-0.35, 1), (0.35, 2))),
    "F8": dict(normal=(-1, 0), edge=(-17.02, 10.9), pallets=(-0.65, 0.65), columns=((-0.65, 3), (0.65, 2))),
    # Shallow totes stand on two stacked pallets, the usual way to bring a low stack up to working height.
    "H4": dict(normal=(1, 0), edge=(-13.83, 22.9), pallets=(0.0,), columns=((-0.33, 2), (0.33, 0)), layers=2),
    "K3": dict(normal=(1, 0), edge=(-8.78, 20.2), pallets=(0.0,), columns=((-0.35, 2), (0.35, 3))),
    "O8": dict(normal=(0, -1), edge=(-0.55, 7.35), pallets=(0.0,), columns=((0.0, 2),), bin_source="F8"),
    "Q4": dict(normal=(1, 0), edge=(1.12, 22.9), pallets=(0.0,), columns=((-0.35, 3), (0.35, 2))),
}


def quad(stage, path, corners, material=None, color=None):
    from pxr import Sdf, UsdGeom, UsdShade

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([tuple(c) for c in corners])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(
        [(0, 0), (1, 0), (1, 1), (0, 1)]
    )
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    if color is not None:
        mesh.CreateDisplayColorAttr([color])
    return mesh


def flat_material(stage, name, color, roughness=0.8):
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, "/World/DropZones/Materials/" + name)
    shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def cube(stage, path, size, translation, yaw_deg, material, collide):
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    prim = UsdGeom.Cube.Define(stage, path)
    prim.CreateSizeAttr(1.0)
    prim.CreateExtentAttr([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
    prim.AddTranslateOp().Set(Gf.Vec3d(*translation))
    prim.AddRotateZOp().Set(yaw_deg)
    prim.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    return prim


def build_pallet(stage, path, centre, yaw_deg, wood, friction, base=0.0):
    """A static two-way pallet: one solid collider with plank visuals on top of it.

    ``base`` is the height its underside stands at [m], for a pallet stacked on another.
    """
    from pxr import UsdGeom, UsdShade

    UsdGeom.Xform.Define(stage, path)
    depth, length, height = PALLET
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))

    def at(dx, dy, z):
        return (centre[0] + c * dx - s * dy, centre[1] + s * dx + c * dy, base + z)

    body = cube(stage, path + "/Collider", (depth, length, height), at(0, 0, height / 2), yaw_deg, wood, True)
    UsdGeom.Imageable(body.GetPrim()).CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(friction, UsdShade.Tokens.weakerThanDescendants, "physics")
    board = 0.022
    for i in range(7):  # top deck boards run along the pallet length
        dx = -depth / 2 + 0.05 + i * (depth - 0.1) / 6
        cube(stage, f"{path}/Top{i}", (0.1, length, board), at(dx, 0, height - board / 2), yaw_deg, wood, False)
    for i, dy in enumerate((-length / 2 + 0.05, 0.0, length / 2 - 0.05)):  # stringers
        cube(
            stage, f"{path}/Stringer{i}", (depth, 0.1, height - 2 * board), at(0, dy, height / 2), yaw_deg, wood, False
        )
    for i, dx in enumerate((-depth / 2 + 0.05, 0.0, depth / 2 - 0.05)):  # bottom boards
        cube(stage, f"{path}/Bottom{i}", (0.1, length, board), at(dx, 0, board / 2), yaw_deg, wood, False)


def main():
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "multi_gpu": False})
    try:
        import omni.usd
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

        items = json.loads((ENV / "pickup/selection.json").read_text())
        by_address = {item["rack_address"]: item for item in items}
        occupancy = np.load(ENV / "nav/occupancy.npz")
        grid, origin, resolution = occupancy["grid"], occupancy["origin"], float(occupancy["resolution"])

        def is_free(low, high):
            i0, j0 = np.floor((np.array(low) - origin) / resolution).astype(int)
            i1, j1 = np.ceil((np.array(high) - origin) / resolution).astype(int)
            return not grid[j0:j1, i0:i1].any()

        original = Sdf.Layer.CreateAnonymous()
        original.ImportFromString((ENV / "factory.usda").read_text())
        ctx = omni.usd.get_context()
        assert ctx.open_stage(str(ENV / "factory.usda"))
        stage = ctx.get_stage()
        root = stage.GetRootLayer()
        root.subLayerPaths = [p for p in root.subLayerPaths if p != "drop_zones.usda"]
        path = ENV / "drop_zones.usda"
        layer = Sdf.Layer.FindOrOpen(str(path)) if path.exists() else Sdf.Layer.CreateNew(str(path))
        layer.Clear()
        root.subLayerPaths.append("drop_zones.usda")
        stage.SetEditTarget(layer)

        UsdGeom.Xform.Define(stage, "/World/DropZones")
        wood = flat_material(stage, "PalletWood", (0.66, 0.5, 0.32))
        tape = flat_material(stage, "FloorTape", (0.95, 0.78, 0.08), 0.5)
        friction = UsdShade.Material.Define(stage, "/World/DropZones/Materials/PalletFriction")
        api = UsdPhysics.MaterialAPI.Apply(friction.GetPrim())
        api.CreateStaticFrictionAttr(0.6)
        api.CreateDynamicFrictionAttr(0.5)
        api.CreateRestitutionAttr(0.0)

        destinations = []
        for address, spec in DESTINATIONS.items():
            item = by_address[address]
            normal = np.array(spec["normal"], dtype=float)
            tangent = np.array([-normal[1], normal[0]])
            edge = np.array(spec["edge"], dtype=float)
            pallet_half = np.abs(normal) * PALLET[0] / 2 + np.abs(tangent) * PALLET[1] / 2
            for _ in range(13):  # slide away from the rack face until every pallet clears overhanging stock
                centres = [edge + tangent * offset - normal * PALLET[0] / 2 for offset in spec["pallets"]]
                if all(is_free(centre - pallet_half, centre + pallet_half) for centre in centres):
                    break
                edge = edge + normal * 0.05
            else:
                raise RuntimeError(f"{address}: no free pallet position near {spec['edge']}")
            face_yaw = math.degrees(math.atan2(normal[1], normal[0]))
            zone = f"/World/DropZones/{address}"
            UsdGeom.Xform.Define(stage, zone)

            pallets = []
            layers = spec.get("layers", 1)
            deck_z = layers * PALLET[2]
            for index, offset in enumerate(spec["pallets"]):
                centre = edge + tangent * offset - normal * PALLET[0] / 2
                for stacked in range(layers):
                    name = f"{zone}/Pallet{index}" + (f"_{stacked}" if stacked else "")
                    build_pallet(stage, name, centre, face_yaw, wood, friction, stacked * PALLET[2])
                half = pallet_half
                pallets.append({"center": [round(v, 4) for v in centre], "half_extents": [round(v, 4) for v in half]})
            span = (min(spec["pallets"]) - PALLET[1] / 2, max(spec["pallets"]) + PALLET[1] / 2)
            stand = (edge + tangent * span[0] + normal * 0.05, edge + tangent * span[1] + normal * 1.3)
            if not is_free(np.minimum(*stand), np.maximum(*stand)):
                raise RuntimeError(f"{address}: no clear standing room in front of the pallet")

            # Floor tape around the pallets and the address painted on the floor in front of them.
            margin, width, z = 0.08, 0.05, 0.002
            a0, a1 = span[0] - margin, span[1] + margin
            d0, d1 = -PALLET[0] - margin, margin
            for i, (ta, tb, da, db) in enumerate(
                ((a0, a1, d0, d0 + width), (a0, a1, d1 - width, d1), (a0, a0 + width, d0, d1), (a1 - width, a1, d0, d1))
            ):
                corners = [edge + tangent * t + normal * d for t, d in ((ta, da), (tb, da), (tb, db), (ta, db))]
                quad(stage, f"{zone}/Tape{i}", [(c[0], c[1], z) for c in corners], tape)
            sticker = UsdShade.Material.Get(stage, "/World/StickerMaterials/" + address)
            if not sticker:
                raise RuntimeError(f"Missing address sticker material {address}")
            size = 0.36
            sign = [
                edge + tangent * t + normal * d
                for t, d in ((size / 2, 0.2), (-size / 2, 0.2), (-size / 2, 0.2 + size), (size / 2, 0.2 + size))
            ]
            quad(stage, f"{zone}/FloorAddress", [(c[0], c[1], z) for c in sign], sticker)

            stock_source = by_address[spec.get("bin_source", address)]
            columns = []
            for column, (offset, count) in enumerate(spec["columns"]):
                front = edge + tangent * offset - normal * FRONT_INSET
                stock = []
                top = deck_z
                for tier in range(count):
                    prim, op, scale = place_reference(
                        stage,
                        f"{zone}/Stock_c{column}_t{tier}",
                        stock_source["path"],
                        0.0,
                        stock_source.get("resize", 1.0),
                    )
                    # Turn the same front face the rack item shows toward the aisle.
                    yaw = stock_source["yaw_deg"] + 90.0 + face_yaw
                    op.Set(placement_matrix(scale, yaw))
                    measured = body_box(stage, prim, yaw, face_yaw)
                    depth, width, height = measured["size"]
                    centre = front - normal * depth / 2
                    target = Gf.Vec3d(centre[0], centre[1], top + SETTLE_GAP + height / 2)
                    low, high = Gf.Vec3d(*measured["world_min"]), Gf.Vec3d(*measured["world_max"])
                    op.Set(placement_matrix(scale, yaw, target - (low + high) / 2))
                    measured = body_box(stage, prim, yaw, face_yaw)
                    is_bin = "bin_source" in spec and tier == count - 1
                    make_dynamic(
                        stage,
                        prim,
                        stock_source["mass"],
                        stock_source["material"],
                        "convexDecomposition" if is_bin else "convexHull",
                    )
                    if is_bin:  # An open tote must stay hollow so the bottle can go inside.
                        for mesh in Usd.PrimRange(prim):
                            if mesh.IsA(UsdGeom.Mesh) and mesh.HasAPI(UsdPhysics.CollisionAPI):
                                hulls = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh)
                                hulls.CreateMaxConvexHullsAttr(32)
                                hulls.CreateHullVertexLimitAttr(32)
                                hulls.CreateVoxelResolutionAttr(500000)
                                hulls.CreateErrorPercentageAttr(2.0)
                                hulls.CreateShrinkWrapAttr(True)
                    top += SETTLE_GAP + height
                    stock.append(
                        {
                            "prim": str(prim.GetPath()),
                            "kind": stock_source["kind"],
                            "family": stock_source["family"],
                            "mass": stock_source["mass"],
                            "center": [round(v, 4) for v in target],
                            **{key: measured[key] for key in ("size", "box_offset", "box_yaw")},
                        }
                    )
                columns.append(
                    {
                        "offset": offset,
                        "front": [round(v, 4) for v in front],
                        "bin": "bin_source" in spec,
                        "stock": stock,
                    }
                )
            destinations.append(
                {
                    "address": address,
                    "aisle": item["aisle"],
                    "normal": list(spec["normal"]),
                    "edge": [round(float(v), 4) for v in edge],
                    "deck_z": deck_z,
                    "pallets": pallets,
                    "columns": columns,
                }
            )

        layer.Save()
        if "drop_zones.usda" not in original.subLayerPaths:
            original.subLayerPaths.append("drop_zones.usda")
        original.Export(str(ENV / "factory.usda"))

        # The Isaac Lab scene: everything except the saved robot and physics scene, which the
        # environment creates itself.
        task = Sdf.Layer.CreateNew(str(ENV / "sorting_task.usda")) if not (ENV / "sorting_task.usda").exists() else None
        task = task or Sdf.Layer.FindOrOpen(str(ENV / "sorting_task.usda"))
        task.Clear()
        task.subLayerPaths.append("factory.usda")
        task_stage = Usd.Stage.Open(task)
        UsdGeom.SetStageUpAxis(task_stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(task_stage, 1.0)
        task_stage.SetDefaultPrim(task_stage.GetPrimAtPath("/World"))
        for hidden in ("/World/Astribot", "/World/Physics"):
            task_stage.OverridePrim(hidden).SetActive(False)
        task.Save()

        manifest = {
            "version": 1,
            "max_stack": MAX_STACK,
            "max_stack_height": MAX_STACK_HEIGHT,
            "front_inset": FRONT_INSET,
            "spawn": SPAWN,
            "rack": {
                "position": list(RACK_POSITION),
                "front_beam_y": FRONT_BEAM_Y,
                "normal": [0, -1],
                "shelf_tops": list(SHELF_TOPS),
            },
            "items": [
                {
                    "address": item["rack_address"],
                    "prim": item["sorting_prim"],
                    "source": item["path"],
                    "family": item["family"],
                    "kind": item["kind"],
                    "material": item["material"],
                    "mass": item["mass"],
                    "level": item["level"],
                    "center": item["sorting_position"],
                    **{key: item[key] for key in ("size", "box_offset", "box_yaw")},
                }
                for item in items
            ],
            "destinations": destinations,
        }
        (ENV / "sorting_task.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stock = sum(len(column["stock"]) for zone in destinations for column in zone["columns"])
        print(f"PERSISTED: {len(destinations)} drop zones, {stock} stock units, manifest and task scene", flush=True)
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
