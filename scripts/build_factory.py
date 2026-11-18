"""Regenerate the persistent factory scene; run with the Isaac Sim Python environment."""

from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics

DATA = Path(__file__).resolve().parents[1] / "src/Factory_Project/assets/data"
WAREHOUSE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"


def build():
    output = DATA / "environments/factory.usda"
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    robot = UsdGeom.Xform.Define(stage, "/World/Astribot")
    robot.GetPrim().GetReferences().AddReference("../astribot_s1/astribot_s1.usd")
    # Face away from the east wall, into the clear aisle beside the first shelf.
    robot.AddTranslateOp().Set(Gf.Vec3d(4.45, 7.3, 0.09523))
    robot.AddRotateZOp().Set(180)
    cache = UsdGeom.BBoxCache(0, ["default", "render"])
    bounds = Gf.Range3d()
    for prim in stage.GetPrimAtPath("/World/Astribot/Geometry").GetChildren():
        if prim.GetName() == "astribot_torso_base" or any(tag in prim.GetName().lower() for tag in ("wheel", "roller")):
            bounds.UnionWith(cache.ComputeWorldBound(prim).ComputeAlignedRange())
    low, high = bounds.GetMin(), bounds.GetMax()
    center = (low + high) / 2
    side = max(high[0] - low[0], high[1] - low[1])
    half = side / 2
    # Minimum enclosing square: inner edges touch the longest chassis/wheel dimension.
    # The slightly narrower dimension has only ~5 mm clearance on each side.
    for name, x, y, angle in [
        ("South", center[0] - half - 0.15, center[1] - half, 0),
        ("East", center[0] + half, center[1] - half - 0.15, 90),
        ("North", center[0] + half + 0.15, center[1] + half, 180),
        ("West", center[0] - half, center[1] + half + 0.15, 270),
    ]:
        strip = UsdGeom.Xform.Define(stage, "/World/SpawnBoundary/" + name)
        strip.GetPrim().GetReferences().AddReference("tape/tape.usd")
        for variant in ["PhysX", "Newton", "MuJoCo"]:
            strip.GetPrim().GetVariantSet(variant).SetVariantSelection("Disabled")
        strip.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.001))
        strip.AddRotateZOp().Set(angle)
        scale = (side + 0.15) / 12
        strip.AddScaleOp().Set(Gf.Vec3f(scale, 1, 1))
        obj = stage.GetPrimAtPath(str(strip.GetPath()) + "/Geometry/sm_tape_safety_12m15cm_f06_obj_00")
        obj.SetInstanceable(False)
        for prim in Usd.PrimRange(strip.GetPrim()):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        mesh = obj.GetChildren()[0]
        uv = mesh.GetAttribute("primvars:st")
        uv.Set([Gf.Vec2f(0.057 + (v[0] - 0.057) * scale, v[1]) for v in uv.Get()])
    boundary = stage.GetPrimAtPath("/World/SpawnBoundary")
    boundary.SetCustomDataByKey("innerSideMeters", side)
    boundary.SetCustomDataByKey(
        "description", "Minimum square enclosing chassis and wheels; decorative non-colliding tape"
    )
    physics = UsdPhysics.Scene.Define(stage, "/World/Physics")
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr(9.81)
    camera = UsdGeom.Camera.Define(stage, "/World/SpawnCamera")
    camera.CreateFocalLengthAttr(32)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000))
    camera.AddTransformOp().Set(
        Gf.Matrix4d().SetLookAt(Gf.Vec3d(0.8, 4.2, 3.1), Gf.Vec3d(4.45, 7.3, 0.8), Gf.Vec3d(0, 0, 1)).GetInverse()
    )
    # Author last: this remote environment resolves through Kit's HTTP resolver.
    stage.DefinePrim("/World/Factory", "Xform").GetReferences().AddReference(WAREHOUSE)
    for layer in ("factory_labels.usda", "sorting_rack.usda"):
        if (DATA / "environments" / layer).is_file():
            stage.GetRootLayer().subLayerPaths.append(layer)
    stage.GetRootLayer().Save()
    print(output, "inner square:", side)


if __name__ == "__main__":
    build()
