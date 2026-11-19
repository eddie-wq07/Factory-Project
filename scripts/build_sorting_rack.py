"""Persist the 'To be sorted.' rack with six dynamic, graspable items: one from each aisle.

Run with the Isaac Sim Python environment from the repository root::

    ~/env_isaaclab/bin/python scripts/build_sorting_rack.py

Slotting follows everyday warehouse practice and the S1's measured reach: the heaviest cartons sit on
the lowest shelf, the crates at chest height, and only the 0.4 kg bottle on the 1.82 m shelf, which is
above the robot's shoulders. Every item is pulled to the front beam, label facing the picker. Items are
rigid bodies with convex colliders, a declared mass and a friction material; the rack stays static.
"""

import json
import math
from pathlib import Path

from label_factory import location

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "src/Factory_Project/assets/data/environments"

RACK_POSITION = (1.0, 3.3, 0.0)
FRONT_BEAM_Y = 2.697  # world Y of the front beam's outer face [m]
FRONT_INSET = 0.03  # gap between the beam face and an item's front face [m]
SHELF_TOPS = (0.623, 1.223, 1.823)  # deck heights of the three used levels [m]
SETTLE_GAP = 0.002  # items start this far above the deck and settle under gravity [m]
AISLE_CENTERS = (-22.83, -17.91, -12.95, -7.99, -3.02, 1.93)

FRONT_YAW = {"+x": -90.0, "-x": 90.0, "+y": 180.0, "-y": 0.0}
"""Item yaw [deg] that turns the named local face toward the picker (world -Y)."""

MATERIALS = {
    # static friction, dynamic friction: dry-contact handbook values, a simulation assumption
    "cardboard": (0.70, 0.60),
    "plastic": (0.50, 0.45),
}

ITEMS = [
    # Alphabetical processing order. `x` is the slot centre along the rack; `front` is the item-local
    # face presented to the picker; `mass` is an assumed parcel mass within the S1's two-arm payload.
    dict(source="/World/Factory/Box_4013", level=1, x=0.03, front="+x", mass=3.0, kind="carton", material="cardboard"),
    dict(source="/World/Factory/Crate_15940", level=2, x=0.27, front="-y", mass=1.8, kind="crate", material="plastic"),
    dict(source="/World/Factory/Crate_16579", level=2, x=1.73, front="+x", mass=1.0, kind="crate", material="plastic"),
    # Rack cartons are at most 0.5 m wide: the S1 clamps a carton between its palms, and on a wider
    # one its forearms bear on the near corners before the palms reach the sides. The K3 carton is
    # this stock line's smaller pack size (the 0.7 m master carton scaled to 0.5 m), long side out.
    dict(source="/World/Factory/Box_19347", level=1, x=1.0, front="-y", mass=2.5, kind="carton", material="cardboard", resize=0.714),
    dict(source="/World/Factory/SM_BottlePlasticD_2", level=2, x=1.0, front="-y", mass=0.4, kind="bottle", material="plastic"),
    dict(source="/World/Factory/Box_33430", level=1, x=1.97, front="+x", mass=2.0, kind="carton", material="cardboard"),
]


def uniform_scale(matrix):
    from pxr import Gf

    scales = [Gf.Vec3d(*matrix.GetRow3(i)).GetLength() for i in range(3)]
    if max(scales) - min(scales) > 1e-4:
        raise ValueError(f"Source prim has a non-uniform scale: {scales}")
    return scales[0]


def placement_matrix(scale, yaw_deg, translation=(0, 0, 0)):
    from pxr import Gf

    matrix = Gf.Matrix4d().SetScale(scale) * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw_deg))
    matrix.SetTranslateOnly(Gf.Vec3d(*translation))
    return matrix


def collider_meshes(prim):
    from pxr import Usd, UsdGeom, UsdPhysics

    return [p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI)]


def body_box(stage, prim, yaw_deg, face_yaw_deg):
    """Measure a placed object's collider box and return its frame relative to the rigid body.

    The *box frame* sits at the collider's bounding-box centre with +X through the front face (the one
    presented to the picker), +Y along the width and +Z up. ``size`` is (depth, width, height) [m].

    Args:
        stage: Stage holding the placed prim.
        prim: Root prim of the object, already carrying its final transform.
        yaw_deg: World yaw of the prim [deg].
        face_yaw_deg: World yaw of the direction the front face looks toward [deg].
    """
    from pxr import Gf, Usd, UsdGeom

    meshes = collider_meshes(prim)
    if len(meshes) != 1:
        raise ValueError(f"{prim.GetPath()} needs exactly one collider mesh, found {len(meshes)}")
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(meshes[0])
    box = bounds.ComputeAlignedRange()
    oriented = bounds.GetRange().GetSize()
    scale = uniform_scale(Gf.Matrix4d(bounds.GetMatrix()))
    aligned = box.GetSize()
    if aligned[0] * aligned[1] * aligned[2] > 1.03 * oriented[0] * oriented[1] * oriented[2] * scale**3:
        raise ValueError(f"{prim.GetPath()} is not axis-aligned after placement")
    centre = (box.GetMin() + box.GetMax()) / 2
    root = UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractTranslation()
    face = math.radians(face_yaw_deg)
    depth_axis = (abs(math.cos(face)), abs(math.sin(face)))
    depth = aligned[0] * depth_axis[0] + aligned[1] * depth_axis[1]
    width = aligned[0] * depth_axis[1] + aligned[1] * depth_axis[0]
    # Express the centre offset and the box yaw in the body frame (the prim frame without scale).
    c, s = math.cos(math.radians(-yaw_deg)), math.sin(math.radians(-yaw_deg))
    offset = centre - root
    local = (c * offset[0] - s * offset[1], s * offset[0] + c * offset[1], offset[2])
    return {
        "size": [round(depth, 4), round(width, 4), round(aligned[2], 4)],
        "box_offset": [round(v, 5) for v in local],
        "box_yaw": round(math.remainder(math.radians(face_yaw_deg - yaw_deg), math.tau), 6),
        "world_min": [round(v, 4) for v in box.GetMin()],
        "world_max": [round(v, 4) for v in box.GetMax()],
    }


def physics_material(stage, name):
    """Create (once) and return a rigid-body friction material shared by the sorting objects."""
    from pxr import UsdPhysics, UsdShade

    path = "/World/SortingMaterials/" + name
    material = UsdShade.Material.Get(stage, path)
    if not material:
        material = UsdShade.Material.Define(stage, path)
        api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr(MATERIALS[name][0])
        api.CreateDynamicFrictionAttr(MATERIALS[name][1])
        api.CreateRestitutionAttr(0.0)
    return material


def make_dynamic(stage, prim, mass, material, approximation="convexHull"):
    """Turn a referenced static prop into a simulated rigid body with a convex collider."""
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    body = UsdPhysics.RigidBodyAPI.Apply(prim)
    body.CreateRigidBodyEnabledAttr(True)
    body.CreateKinematicEnabledAttr(False)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    physx = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx.CreateSolverPositionIterationCountAttr(16)
    physx.CreateSolverVelocityIterationCountAttr(2)
    physx.CreateMaxDepenetrationVelocityAttr(1.0)
    physx.CreateLinearDampingAttr(0.05)
    physx.CreateAngularDampingAttr(0.1)
    physx.CreateSleepThresholdAttr(0.002)
    for mesh in collider_meshes(prim):
        UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr(approximation)
        collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh)
        collision.CreateContactOffsetAttr(0.005)
        collision.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(
            physics_material(stage, material), UsdShade.Tokens.weakerThanDescendants, "physics"
        )


RACK_HALF_SPAN = (1.55, 0.65)  # half the rack's length and depth about its position [m]
RACK_POST = (0.10, 0.15)  # upright footprint along the length and the depth [m]
RACK_HEIGHT = 2.5
DECK_TOPS = (*SHELF_TOPS, 2.423)
DECK_THICKNESS = 0.03


def build_rack_colliders(stage, rack):
    """Collide with the rack through a few boxes instead of its 850,000-triangle render mesh.

    The mesh is far denser than contact needs, and PhysX crashed when the robot pressed into it.
    Four decks and four uprights reproduce every surface an item or the robot can touch: the decks
    carry the stock, and the space under the lowest deck stays open for the chassis.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

    for prim in Usd.PrimRange(rack):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    material = UsdShade.Material.Define(stage, "/World/SortingMaterials/painted_steel")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(0.5)
    physics.CreateDynamicFrictionAttr(0.4)
    physics.CreateRestitutionAttr(0.0)
    root = UsdGeom.Xform.Define(stage, str(rack.GetPath()) + "/Colliders")
    half_x, half_y = RACK_HALF_SPAN
    post_x, post_y = RACK_POST
    boxes = {f"Deck{i}": ((0, 0, top - DECK_THICKNESS / 2), (2 * half_x - 2 * post_x, 2 * half_y - 0.094, DECK_THICKNESS)) for i, top in enumerate(DECK_TOPS, start=1)}
    for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
        centre = (sx * (half_x - post_x / 2), sy * (half_y - post_y / 2), RACK_HEIGHT / 2)
        boxes[f"Post{i}"] = (centre, (post_x, post_y, RACK_HEIGHT))
    for name, (centre, size) in boxes.items():
        cube = UsdGeom.Cube.Define(stage, f"{root.GetPath()}/{name}")
        cube.CreateSizeAttr(1.0)
        cube.CreateExtentAttr([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
        cube.AddTranslateOp().Set(Gf.Vec3d(*centre))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")


def place_reference(stage, path, source_path, yaw_deg, resize=1.0):
    """Reference an existing warehouse prop at ``path`` with only its scale, times ``resize``, and a new yaw."""
    from pxr import UsdGeom

    source = stage.GetPrimAtPath(source_path)
    if not source:
        raise RuntimeError(f"Missing source prim {source_path}")
    scale = uniform_scale(UsdGeom.XformCache().GetLocalToWorldTransform(source)) * resize
    xform = UsdGeom.Xform.Define(stage, path)
    prim = xform.GetPrim()
    prim.GetReferences().AddInternalReference(source_path)
    prim.SetInstanceable(False)
    xform.ClearXformOpOrder()
    op = xform.AddTransformOp(opSuffix="sorting")
    op.Set(placement_matrix(scale, yaw_deg))
    return prim, op, scale


def main():
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "multi_gpu": False})
    try:
        import omni.usd
        from PIL import Image, ImageDraw, ImageFont
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

        original = Sdf.Layer.CreateAnonymous()
        original.ImportFromString((ENV / "factory.usda").read_text())
        ctx = omni.usd.get_context()
        assert ctx.open_stage(str(ENV / "factory.usda"))
        stage = ctx.get_stage()
        root = stage.GetRootLayer()
        dependents = ("sorting_rack.usda", "drop_zones.usda")
        root.subLayerPaths = [p for p in root.subLayerPaths if p not in dependents]
        path = ENV / "sorting_rack.usda"
        layer = Sdf.Layer.FindOrOpen(str(path)) if path.exists() else Sdf.Layer.CreateNew(str(path))
        layer.Clear()
        root.subLayerPaths.append("sorting_rack.usda")
        stage.SetEditTarget(layer)
        rack = UsdGeom.Xform.Define(stage, "/World/SortingRack/Rack")
        rack.GetPrim().GetReferences().AddReference("pickup/rack/sm_rack_composition_l10_01.usd")
        # Same location as rack 6 in the reverted arrangement; original aisle 6 is untouched.
        rack.AddTranslateOp().Set(Gf.Vec3d(*RACK_POSITION))
        for variant in rack.GetPrim().GetVariantSets().GetNames():
            rack.GetPrim().GetVariantSet(variant).SetVariantSelection("Disabled")
        for prim in stage.GetPrimAtPath(str(rack.GetPath()) + "/Geometry").GetChildren():
            if prim.GetName() != "sm_rack_l18_obj_00":
                prim.SetActive(False)
        for prim in Usd.PrimRange(rack.GetPrim()):
            if prim.IsInstance():
                prim.SetInstanceable(False)
        for prim in Usd.PrimRange(rack.GetPrim()):
            # The rack is a fixed fixture: keep its collider and drop the rigid-body schema entirely.
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)

        build_rack_colliders(stage, rack.GetPrim())

        xforms = UsdGeom.XformCache()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        entries = []
        for slot, spec in enumerate(ITEMS, start=1):
            source = stage.GetPrimAtPath(spec["source"])
            assert source, spec["source"]
            source_box = cache.ComputeWorldBound(source).ComputeAlignedRange()
            centre = (source_box.GetMin() + source_box.GetMax()) / 2
            references = [
                str(ref.assetPath)
                for layer_spec in source.GetPrimStack()
                for ref in layer_spec.referenceList.GetAddedOrExplicitItems()
            ]
            asset = references[0] if references else ""
            aisle = min(range(6), key=lambda i: abs(centre[0] - AISLE_CENTERS[i])) + 1
            side = int(centre[0] > AISLE_CENTERS[aisle - 1])
            address = location(aisle - 1, side, centre[1])

            yaw = FRONT_YAW[spec["front"]]
            prim, op, scale = place_reference(stage, f"/World/SortingRack/Item{slot}", spec["source"], yaw, spec.get("resize", 1.0))
            xforms.Clear()
            measured = body_box(stage, prim, yaw, -90.0)
            depth, width, height = measured["size"]
            shelf = SHELF_TOPS[spec["level"] - 1]
            target = Gf.Vec3d(spec["x"], FRONT_BEAM_Y + FRONT_INSET + depth / 2, shelf + SETTLE_GAP + height / 2)
            low, high = Gf.Vec3d(*measured["world_min"]), Gf.Vec3d(*measured["world_max"])
            op.Set(placement_matrix(scale, yaw, target - (low + high) / 2))
            xforms.Clear()
            measured = body_box(stage, prim, yaw, -90.0)
            assert height < 0.55 and width < 0.65 and depth < 0.52, (address, measured["size"])
            make_dynamic(stage, prim, spec["mass"], spec["material"])
            prim.SetCustomDataByKey("sourceAisle", aisle)
            prim.SetCustomDataByKey("rackAddress", address)

            entries.append(
                {
                    "name": source.GetName(),
                    "path": spec["source"],
                    "aisle": aisle,
                    "asset": asset,
                    "family": Path(asset).stem,
                    "dimensions": [round(v, 4) for v in source_box.GetSize()],
                    "center": [round(v, 4) for v in centre],
                    "slot": slot,
                    "level": spec["level"],
                    "sorting_prim": str(prim.GetPath()),
                    "sorting_position": [round(v, 4) for v in target],
                    "rack_address": address,
                    "kind": spec["kind"],
                    "material": spec["material"],
                    "mass": spec["mass"],
                    "yaw_deg": yaw,
                    "source_scale": round(scale, 5),
                    "resize": spec.get("resize", 1.0),
                    **{key: measured[key] for key in ("size", "box_offset", "box_yaw")},
                }
            )
            # Three-inch squares fit the narrow front rail, one pair below each item.
            for name, material_name, offset in (("Address", address, 0.0411), ("AprilTag", "AprilTag", -0.0411)):
                material = UsdShade.Material.Get(stage, "/World/StickerMaterials/" + material_name)
                if not material:
                    raise RuntimeError(f"Missing existing sticker material: {material_name}")
                sticker = UsdGeom.Mesh.Define(stage, f"/World/SortingRack/Rack/SlotLabels/Item{slot}/{name}")
                x = spec["x"] - RACK_POSITION[0] + offset
                z = shelf - 0.04
                half = 0.0762 / 2
                sticker.CreatePointsAttr(
                    [
                        (x - half, -0.652, z - half),
                        (x + half, -0.652, z - half),
                        (x + half, -0.652, z + half),
                        (x - half, -0.652, z + half),
                    ]
                )
                sticker.CreateFaceVertexCountsAttr([4])
                sticker.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
                sticker.CreateSubdivisionSchemeAttr("none")
                sticker.CreateDoubleSidedAttr(True)
                UsdGeom.PrimvarsAPI(sticker).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(
                    [(0, 0), (1, 0), (1, 1), (0, 1)]
                )
                UsdShade.MaterialBindingAPI.Apply(sticker.GetPrim()).Bind(material)

        addresses = [entry["rack_address"] for entry in entries]
        assert len(entries) == 6 and {entry["aisle"] for entry in entries} == set(range(1, 7)), addresses
        assert addresses == sorted(addresses), f"Items must be listed in alphabetical address order: {addresses}"

        image = Image.new("RGB", (1024, 200), "white")
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 100)
        ImageDraw.Draw(image).text((512, 100), "To be sorted.", font=font, anchor="mm", fill="black")
        image.save(ENV / "pickup/to-be-sorted.png")
        material = UsdShade.Material.Define(stage, "/World/SortingRack/SignMaterial")
        shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1)
        texture = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("pickup/to-be-sorted.png"))
        uv = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/UV")
        uv.CreateIdAttr("UsdPrimvarReader_float2")
        uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        mesh = UsdGeom.Mesh.Define(stage, "/World/SortingRack/Sign")
        mesh.CreatePointsAttr([(-0.1, 2.635, 2.35), (2.1, 2.635, 2.35), (2.1, 2.635, 2.78), (-0.1, 2.635, 2.78)])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(
            [(0, 0), (1, 0), (1, 1), (0, 1)]
        )
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        camera = UsdGeom.Camera.Define(stage, "/World/SortingRack/Camera")
        camera.CreateFocalLengthAttr(36)
        camera.CreateHorizontalApertureAttr(36)
        camera.CreateVerticalApertureAttr(27)
        camera.AddTransformOp().Set(
            Gf.Matrix4d().SetLookAt(Gf.Vec3d(1, -1.8, 2.6), Gf.Vec3d(1, 3.3, 1.4), Gf.Vec3d(0, 0, 1)).GetInverse()
        )
        layer.Save()
        for name in dependents:
            if (ENV / name).is_file() and name not in original.subLayerPaths:
                original.subLayerPaths.append(name)
        original.Export(str(ENV / "factory.usda"))
        (ENV / "pickup/selection.json").write_text(json.dumps(entries, indent=2) + "\n")
        print("PERSISTED: one rack, six dynamic items:", ", ".join(addresses), flush=True)
        for entry in entries:
            print(f"  {entry['rack_address']} level {entry['level']} {entry['family']} D/W/H {entry['size']}", flush=True)
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
