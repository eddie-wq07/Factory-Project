"""Persist one 'To be sorted.' rack with one unchanged-scale sample from each aisle."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "src/Factory_Project/assets/data/environments"


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
        root.subLayerPaths = [p for p in root.subLayerPaths if p != "sorting_rack.usda"]
        path = ENV / "sorting_rack.usda"
        layer = Sdf.Layer.FindOrOpen(str(path)) if path.exists() else Sdf.Layer.CreateNew(str(path))
        layer.Clear()
        root.subLayerPaths.append("sorting_rack.usda")
        stage.SetEditTarget(layer)
        rack = UsdGeom.Xform.Define(stage, "/World/SortingRack/Rack")
        rack.GetPrim().GetReferences().AddReference("pickup/rack/sm_rack_composition_l10_01.usd")
        # Same location as rack 6 in the reverted arrangement; original aisle 6 is untouched.
        rack.AddTranslateOp().Set(Gf.Vec3d(1, 3.3, 0))
        for variant in rack.GetPrim().GetVariantSets().GetNames():
            rack.GetPrim().GetVariantSet(variant).SetVariantSelection("Disabled")
        for prim in stage.GetPrimAtPath(str(rack.GetPath()) + "/Geometry").GetChildren():
            if prim.GetName() != "sm_rack_l18_obj_00":
                prim.SetActive(False)
        for prim in Usd.PrimRange(rack.GetPrim()):
            if prim.IsInstance():
                prim.SetInstanceable(False)
        for prim in Usd.PrimRange(rack.GetPrim()):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)

        cache = UsdGeom.BBoxCache(0, ["default", "render"])
        xforms = UsdGeom.XformCache()
        entries = json.loads((ENV / "pickup/selection.json").read_text())
        assert len(entries) == 6 and {p["aisle"] for p in entries} == set(range(1, 7))
        for entry in entries:
            source = stage.GetPrimAtPath(entry["path"])
            assert source
            item = UsdGeom.Xform.Define(stage, f"/World/SortingRack/Item{entry['slot']}")
            item.GetPrim().GetReferences().AddInternalReference(entry["path"])
            item.GetPrim().SetInstanceable(False)
            matrix = xforms.GetLocalToWorldTransform(source)
            matrix.SetTranslateOnly(Gf.Vec3d(0))
            if entry["name"].startswith("Box_"):
                rows = (-25.3, -20.4, -15.4, -10.5, -5.5, -0.55, 4.41)
                row = min(range(7), key=lambda i: abs(entry["center"][0] - rows[i]))
                positive = row == 0 or (row != 6 and entry["center"][0] >= rows[row])
                matrix = matrix * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), -90 if positive else 90))
            item.ClearXformOpOrder()
            op = item.AddTransformOp(opSuffix="sorting")
            op.Set(matrix)
            cache.Clear()
            bounds = cache.ComputeWorldBound(item.GetPrim()).ComputeAlignedRange()
            low, high = bounds.GetMin(), bounds.GetMax()
            shelf = (0.6232, 1.2234, 1.8234)[entry["level"] - 1]
            target = Gf.Vec3d(1 + (-0.73 if entry["slot"] % 2 else 0.73), 3.25, shelf + 0.003)
            matrix.SetTranslateOnly(target - Gf.Vec3d((low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]))
            op.Set(matrix)
            assert high[2] - low[2] < 0.519 and high[0] - low[0] < 1.35 and high[1] - low[1] < 1.15
            item.GetPrim().SetCustomDataByKey("sourceAisle", entry["aisle"])
            entry["sorting_prim"] = str(item.GetPath())
            entry["sorting_position"] = list(target)

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
        if "sorting_rack.usda" not in original.subLayerPaths:
            original.subLayerPaths.append("sorting_rack.usda")
        original.Export(str(ENV / "factory.usda"))
        (ENV / "pickup/selection.json").write_text(json.dumps(entries, indent=2) + "\n")
        print("PERSISTED: one rack, six items, one from each aisle; top empty", flush=True)
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
