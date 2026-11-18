"""Rebuild the mobile S1 with urdf-usd-converter (0.3.2) and OpenUSD.

Run with the Python environment containing Isaac Sim 6.2 or urdf-usd-converter.
Geometry, texture bindings and inertias come from the bundled URDF.
"""

import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import urdf_usd_converter
import usdex.core
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

ASSET = Path(__file__).resolve().parent
URDF = ASSET / "urdf" / "astribot_s1.urdf"
NAME = URDF.stem


def prepare_bodies(stage, robot):
    """Place rigid bodies side by side so a parent transform never moves a simulated child twice."""
    geometry = stage.GetDefaultPrim().GetChild("Geometry")
    masses = {link.get("name"): float(link.find("inertial/mass").get("value")) for link in robot.findall("link")}
    links = {p.GetName(): p for p in Usd.PrimRange(geometry) if p.IsA(UsdGeom.Xform) and p.GetName() in masses}
    if set(links) != set(masses):
        raise ValueError(f"Converter omitted links: {masses.keys() - links.keys()}")

    # Fixed zero-mass mounts stay as ordinary frames attached to their massive ancestor.
    for name, mass in masses.items():
        if mass == 0:
            links[name].RemoveAPI(UsdPhysics.RigidBodyAPI)
            links[name].RemoveAPI(UsdPhysics.MassAPI)

    cache = UsdGeom.XformCache()
    moves = [
        (p.GetPath(), cache.GetLocalToWorldTransform(p))
        for name, p in links.items()
        if masses[name] > 0 and p.GetParent() != geometry
    ]
    for path, transform in sorted(moves, key=lambda item: -item[0].pathElementCount):
        editor = Usd.NamespaceEditor(stage)
        editor.MovePrimAtPath(path, geometry.GetPath().AppendChild(path.name))
        if not editor.ApplyEdits():
            raise ValueError(f"Could not relocate {path}")
        prim = geometry.GetChild(path.name)
        usdex.core.setLocalTransform(prim, transform.ExtractTranslation(), Gf.Quatf(transform.ExtractRotationQuat()))

    return {p.GetName(): p.GetPath() for p in Usd.PrimRange(geometry) if p.IsA(UsdGeom.Xform) and p.GetName() in masses}


def drive_gains(name):
    """Simulation PD gains in N m/rad and N m s/rad, grouped by actuator."""
    if name.startswith("wheel_"):
        return 0, 45
    if "torso" in name:
        return 10000, 1000
    if "gripper" in name:
        return 20, 1
    if "arm" in name:
        return [(100, 10), (100, 10), (100, 10), (500, 50), (500, 50), (200, 20), (200, 20)][int(name[-1]) - 1]
    return 100, 10  # Head pan and tilt.


def write_joints(stage, robot, paths):
    physics = UsdGeom.Scope.Define(stage, stage.GetDefaultPrim().GetPath().AppendChild("Physics")).GetPrim()
    frame = usdex.core.JointFrame(usdex.core.JointFrame.Space.Body1, Gf.Vec3d(0), Gf.Quatd.GetIdentity())
    for source in robot.findall("joint"):
        name, kind = source.get("name"), source.get("type")
        body0 = stage.GetPrimAtPath(paths[source.find("parent").get("link")])
        body1 = stage.GetPrimAtPath(paths[source.find("child").get("link")])
        if not body1.HasAPI(UsdPhysics.RigidBodyAPI):
            if kind != "fixed":
                raise ValueError(f"Missing moving rigid body for {name}")
            continue
        while not body0.HasAPI(UsdPhysics.RigidBodyAPI):
            body0 = body0.GetParent()
        if kind == "fixed":
            usdex.core.definePhysicsFixedJoint(physics, name, body0, body1, frame)
            continue
        limit = source.find("limit")
        axis = Gf.Vec3f(*map(float, source.find("axis").get("xyz").split()))
        bounds = [] if kind == "continuous" else [math.degrees(float(limit.get(key))) for key in ("lower", "upper")]
        prim = usdex.core.definePhysicsRevoluteJoint(physics, name, body0, body1, frame, axis, *bounds).GetPrim()
        prim.ApplyAPI("NewtonJointAPI")
        speed_limit = math.degrees(float(limit.get("velocity")))
        prim.GetAttribute("newton:velocityLimit").Set(speed_limit)
        prim.AddAppliedSchema("PhysxJointAPI")
        prim.CreateAttribute("physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float, custom=False).Set(speed_limit)
        # Reflect motor inertia into the upper-body joints to stabilize PhysX GPU solves.
        if kind == "revolute":
            armature = 0.005 if "gripper" in name else 0.06
            prim.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float, custom=False).Set(armature)
        mimic = source.find("mimic")
        if mimic is not None:
            prim.ApplyAPI("NewtonMimicAPI")
            prim.GetAttribute("newton:mimicCoef0").Set(math.degrees(float(mimic.get("offset", "0"))))
            prim.GetAttribute("newton:mimicCoef1").Set(float(mimic.get("multiplier")))
            prim.CreateRelationship("newton:mimicJoint").SetTargets([physics.GetPath().AppendChild(mimic.get("joint"))])
            continue
        effort = float(limit.get("effort"))
        if effort == 0:
            continue  # Passive roller. A drive here would resist lateral travel.
        kp, kd = drive_gains(name)
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(math.radians(kp))
        drive.CreateDampingAttr(math.radians(kd))
        drive.CreateMaxForceAttr(effort)
        drive.CreateTargetVelocityAttr(0)
        if kp:
            drive.CreateTargetPositionAttr(0)


def configure_cameras(stage, paths):
    for source in json.loads(URDF.with_name("cameras.json").read_text()):
        camera = UsdGeom.Camera.Define(stage, paths[source["link"]].AppendChild(source["name"]))
        w, x, y, z = source["quat_wxyz"]
        orientation = Gf.Quatf(w, x, y, z).GetNormalized()
        if source["name"] == "head_rgbd":
            # The head sensor is mounted roll-inverted. Deliver an upright optical image.
            orientation *= Gf.Quatf(0, 0, 0, 1)
        usdex.core.setLocalTransform(camera.GetPrim(), Gf.Vec3d(*source["pos"]), orientation)
        focal = 24.0
        aperture = 2.0 * focal * math.tan(math.radians(source["fovy_degrees"]) / 2.0)
        camera.CreateFocalLengthAttr(focal)
        camera.CreateVerticalApertureAttr(aperture)
        camera.CreateHorizontalApertureAttr(aperture * (4 / 3))
        camera.CreateClippingRangeAttr((0.01, 100))


def composed_stage(file, sublayer):
    stage = Usd.Stage.CreateNew(str(file))
    stage.GetRootLayer().subLayerPaths = ["./" + sublayer.name]
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/" + NAME))
    return stage


def build(output):
    robot = ET.parse(URDF)
    raw = urdf_usd_converter.Converter(layer_structure=False, scene=False).convert(str(URDF), str(output))
    base = Usd.Stage.Open(raw.path)
    paths = prepare_bodies(base, robot)
    # The generic converter adds a fixed joint to the world. Recreate the URDF
    # joints in a separate editable layer, with a floating base and welded mounts.
    base.RemovePrim(base.GetDefaultPrim().GetChild("Physics").GetPath())
    mesh_file = output / "model.usdc"
    base.GetRootLayer().documentation = "Astribot S1 geometry and link inertias converted from urdf/astribot_s1.urdf."
    base.GetRootLayer().Export(str(mesh_file))
    del base

    file = output / (NAME + ".usda")
    stage = composed_stage(file, mesh_file)
    write_joints(stage, robot, paths)
    configure_cameras(stage, paths)
    root = next(p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI))
    root.AddAppliedSchema("PhysxArticulationAPI")
    root.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool, custom=False).Set(False)
    root.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int, custom=False).Set(32)
    root.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int, custom=False).Set(8)
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.CollisionAPI):
            p.AddAppliedSchema("PhysxCollisionAPI")
            p.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float, custom=False).Set(0.002)
            p.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float, custom=False).Set(0)
    stage.GetRootLayer().Save()
    entry = output / (NAME + ".usd")
    interface = composed_stage(entry, file)
    interface.GetRootLayer().Save()
    layers, textures, unresolved = UsdUtils.ComputeAllDependencies(str(entry))
    files = [Path(layer.realPath) for layer in layers] + [Path(path) for path in textures]
    if unresolved or any(not path.is_relative_to(output) for path in files):
        raise ValueError(f"Nonportable USD dependencies: {unresolved}, {files}")
    return files


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix=".build-", dir=ASSET) as directory:
        output = Path(directory)
        for file in reversed(build(output)):
            dest = ASSET / file.relative_to(output)
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(file, dest)
    print(ASSET / (NAME + ".usd"))
