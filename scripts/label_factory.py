"""Author persistent decorative carton and mixed-rack labels in a separate USD layer.

Run with the project's Isaac Sim Python environment. No physics properties change.
The factory must be loaded through factory.usda, which composes factory_labels.usda.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src/Factory_Project/assets/data/environments"
ROWS = (-25.3, -20.4, -15.4, -10.5, -5.5, -0.55, 4.41)


def location(aisle, side, y):
    """Use the organization map: left face northbound, right face southbound."""
    progress = max(0.0, min(3.99999, (y - 8.9) / 4))
    progress = progress if side == 0 else 8 - progress
    progress = min(progress, 7.99999)
    bay = int(progress) + 1
    letter = chr(65 + aisle * 3 + min(2, int(progress / (8 / 3))))
    return f"{letter}{bay}"


def main():
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "multi_gpu": False})
    try:
        import csv

        import cv2
        import numpy as np
        import omni.usd
        from PIL import Image, ImageDraw, ImageFont
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

        textures = DATA / "labels/textures"
        textures.mkdir(parents=True, exist_ok=True)
        # AprilTag 36h11 / ID 0: deliberately identical everywhere, no detector integration.
        marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 0, 400)
        tag = Image.new("RGB", (512, 512), "white")
        tag.paste(Image.fromarray(marker).convert("RGB"), (56, 56))
        tag.save(textures / "apriltag_36h11_id0.png")
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 235)
        original_root = Sdf.Layer.CreateAnonymous()
        original_root.ImportFromString((DATA / "factory.usda").read_text())
        context = omni.usd.get_context()
        assert context.open_stage(str(DATA / "factory.usda"))
        stage = context.get_stage()
        root_layer = stage.GetRootLayer()
        # Remove old label composition before measuring original geometry on a rebuild.
        root_layer.subLayerPaths = [p for p in root_layer.subLayerPaths if p != "factory_labels.usda"]
        path = DATA / "factory_labels.usda"
        layer = Sdf.Layer.FindOrOpen(str(path)) if path.exists() else Sdf.Layer.CreateNew(str(path))
        layer.Clear()
        root_layer.subLayerPaths.append("factory_labels.usda")
        stage.SetEditTarget(layer)
        cache = UsdGeom.BBoxCache(0, ["default", "render"])
        xforms = UsdGeom.XformCache()
        materials = {}
        report = []

        def material(code):
            if code in materials:
                return materials[code]
            filename = "apriltag_36h11_id0.png" if code == "AprilTag" else code + ".png"
            if code != "AprilTag":
                im = Image.new("RGB", (512, 512), "white")
                draw = ImageDraw.Draw(im)
                draw.text((256, 256), code, font=font, fill="black", anchor="mm")
                im.save(textures / filename)
            m = UsdShade.Material.Define(stage, "/World/StickerMaterials/" + code)
            shader = UsdShade.Shader.Define(stage, str(m.GetPath()) + "/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            tex = UsdShade.Shader.Define(stage, str(m.GetPath()) + "/Texture")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("labels/textures/" + filename))
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            uv = UsdShade.Shader.Define(stage, str(m.GetPath()) + "/UV")
            uv.CreateIdAttr("UsdPrimvarReader_float2")
            uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
            m.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            materials[code] = m
            return m

        def plane(parent, name, center, right, up, size, mat):
            mesh = UsdGeom.Mesh.Define(stage, str(parent.GetPath()) + "/LocationStickers/" + name)
            corners = [center + size / 2 * (a * right + b * up) for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
            inverse = xforms.GetLocalToWorldTransform(parent).GetInverse()
            mesh.CreatePointsAttr([Gf.Vec3f(inverse.Transform(Gf.Vec3d(*p))) for p in corners])
            mesh.CreateFaceVertexCountsAttr([4])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(True)
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(
                [(0, 0), (1, 0), (1, 1), (0, 1)]
            )
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
            mesh.GetPrim().SetCustomDataByKey("decorativeOnly", True)
            return mesh

        def surface(parent, wanted):
            """Find the largest planar mesh patch facing the aisle, in world coordinates."""
            world = xforms.GetLocalToWorldTransform(parent)
            inv = world.GetInverse()
            normal = np.array(inv.TransformDir(Gf.Vec3d(*wanted)))
            axis = int(np.argmax(np.abs(normal)))
            sign = 1 if normal[axis] > 0 else -1
            n = np.eye(3)[axis] * sign
            n = np.array(world.TransformDir(Gf.Vec3d(*n)))
            n /= np.linalg.norm(n)
            groups = {}
            for prim in Usd.PrimRange(parent, Usd.TraverseInstanceProxies()):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                mesh = UsdGeom.Mesh(prim)
                transform = xforms.GetLocalToWorldTransform(prim)
                points = np.array([transform.Transform(Gf.Vec3d(*p)) for p in mesh.GetPointsAttr().Get()])
                indices = list(mesh.GetFaceVertexIndicesAttr().Get())
                offset = 0
                for count in mesh.GetFaceVertexCountsAttr().Get():
                    polygon = points[indices[offset : offset + count]]
                    offset += count
                    if count < 3:
                        continue
                    cross = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
                    area = np.linalg.norm(cross) / 2
                    if area < 1e-10 or np.dot(cross / (2 * area), n) < 0.98:
                        continue
                    actual_normal = cross / (2 * area)
                    d = round(float(np.mean(polygon @ actual_normal)), 4)
                    key = (*np.round(actual_normal, 4), d)
                    entry = groups.setdefault(key, [0.0, [], actual_normal])
                    entry[0] += area
                    entry[1].extend(polygon)
            if not groups:
                raise RuntimeError("No aisle-facing box surface: " + str(parent.GetPath()))
            _, (_, points, n) = max(groups.items(), key=lambda item: item[1][0])
            points = np.array(points)
            up = np.array([0.0, 0.0, 1.0])
            up -= n * np.dot(up, n)
            up /= np.linalg.norm(up)
            right = np.cross(up, n)
            d = float(np.mean(points @ n))
            rmin, rmax = float(min(points @ right)), float(max(points @ right))
            umin, umax = float(min(points @ up)), float(max(points @ up))
            return n, right, up, d, rmin, rmax, umin, umax

        boxes = [
            p
            for p in stage.GetPrimAtPath("/World/Factory").GetChildren()
            if p.GetName().startswith(("Box_", "SM_CardBox"))
        ]
        for index, box in enumerate(boxes):
            b = cache.ComputeWorldBound(box).ComputeAlignedRange()
            c = (b.GetMin() + b.GetMax()) / 2
            row = min(range(7), key=lambda r: abs(c[0] - ROWS[r]))
            if box.GetName().startswith("SM_CardBox"):
                aisle, side = 4, 0 if c[0] < -3.02 else 1
                wanted = (0, -1, 0)  # Floor-clutter labels face the south approach.
            elif row == 0:
                aisle, side, wanted = 0, 0, (1, 0, 0)
            elif row == 6:
                aisle, side, wanted = 5, 1, (-1, 0, 0)
            elif c[0] < ROWS[row]:
                aisle, side, wanted = row - 1, 1, (-1, 0, 0)
            else:
                aisle, side, wanted = row, 0, (1, 0, 0)
            code = location(aisle, side, c[1])
            n, right, up, d, rmin, rmax, umin, umax = surface(box, wanted)
            available_width, available_height = rmax - rmin, umax - umin
            size = 0.1524 if available_width >= 0.3248 and available_height >= 0.1724 else 0.0762
            if available_width < 2 * size + 0.012 or available_height < size + 0.012:
                raise RuntimeError(
                    f"Even paired 3-inch stickers do not fit {box.GetPath()}: {available_width, available_height}"
                )
            box.SetInstanceable(False)
            # The text occupies the top-right corner of the aisle-facing wall; tag immediately left.
            center = n * (d + 0.0015) + right * (rmax - 0.006 - size / 2) + up * (umax - 0.006 - size / 2)
            plane(box, "Address", center, right, up, size, material(code))
            plane(box, "AprilTag", center - right * (size + 0.006), right, up, size, material("AprilTag"))
            report.append([str(box.GetPath()), code, aisle + 1, "carton", size])
            if index % 300 == 0:
                print("LABELED", index, flush=True)
        # R3 is mixed storage, with two aisle-facing sides and four longitudinal bays.
        shelves = [
            p for p in stage.GetPrimAtPath("/World/Factory").GetChildren() if p.GetName().startswith("SM_RackShelf")
        ]
        for aisle, side, wanted in [(1, 1, (-1, 0, 0)), (2, 0, (1, 0, 0))]:
            for bay_from_south in range(4):
                y = 10.9 + bay_from_south * 4
                z = 1.3 if bay_from_south % 2 == 0 else 2.8

                def distance(p, y=y, z=z):
                    b = cache.ComputeWorldBound(p).ComputeAlignedRange()
                    c = (b.GetMin() + b.GetMax()) / 2
                    return abs(c[0] - ROWS[2]) + abs(c[1] - y) + abs(b.GetMin()[2] - z)

                shelf = min(shelves, key=distance)
                b = cache.ComputeWorldBound(shelf).ComputeAlignedRange()
                # Rail top is approximately the shelf height; place the 6-inch plate on its front.
                x = b.GetMin()[0] - 0.002 if wanted[0] < 0 else b.GetMax()[0] + 0.002
                world_center = Gf.Vec3d(x, y, z - 0.075)
                center = np.array(world_center)
                up = np.array([0.0, 0.0, 1.0])
                right = np.array([0.0, wanted[0], 0.0])
                shelf.SetInstanceable(False)
                code = location(aisle, side, y)
                plane(shelf, f"Aisle{aisle + 1}_Address", center + right * 0.0792, right, up, 0.1524, material(code))
                plane(
                    shelf,
                    f"Aisle{aisle + 1}_AprilTag",
                    center - right * 0.0792,
                    right,
                    up,
                    0.1524,
                    material("AprilTag"),
                )
                report.append([str(shelf.GetPath()), code, aisle + 1, "mixed-cargo rail", 0.1524])
        layer.Save()
        if "factory_labels.usda" not in original_root.subLayerPaths:
            original_root.subLayerPaths.append("factory_labels.usda")
        original_root.Export(str(DATA / "factory.usda"))
        with (DATA / "labels/manifest.csv").open("w") as f:
            writer = csv.writer(f)
            writer.writerow(["parent_prim", "address", "aisle", "placement", "size_m"])
            writer.writerows(report)
        print(
            "PERSISTED",
            len(boxes),
            "cartons;",
            len(report) - len(boxes),
            "rail pairs;",
            len(materials),
            "materials",
            flush=True,
        )
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
