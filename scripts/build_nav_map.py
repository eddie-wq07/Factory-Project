"""Rasterise the static warehouse into a 2D occupancy grid for route planning.

Run with the Isaac Sim Python environment from the repository root::

    ~/env_isaaclab/bin/python scripts/build_nav_map.py

Every visible mesh triangle that crosses the robot's height band is projected onto the floor. The
result is ``environments/nav/occupancy.npz`` plus a PNG preview. Movable task objects (the robot, the
rack items, drop-zone stock) are left out; the planner adds drop pallets from the task manifest.
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "src/Factory_Project/assets/data/environments"
RESOLUTION = 0.05  # metres per cell
BAND = (0.04, 1.75)  # obstacle heights that can touch the chassis, torso, arms or a carried carton [m]
SKIP = ("/World/Astribot", "/World/SortingRack/Item", "/World/SpawnBoundary", "/World/DropZones")


def main():
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "multi_gpu": False})
    try:
        import cv2
        import omni.usd
        from pxr import Usd, UsdGeom

        context = omni.usd.get_context()
        assert context.open_stage(str(ENV / "factory.usda"))
        stage = context.get_stage()
        bounds = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(stage.GetPrimAtPath("/World/Factory"))
        low, high = bounds.ComputeAlignedRange().GetMin(), bounds.ComputeAlignedRange().GetMax()
        origin = np.floor(np.array([low[0], low[1]]) - 1.0)
        shape = np.ceil((np.array([high[0], high[1]]) + 1.0 - origin) / RESOLUTION).astype(int)
        grid = np.zeros((shape[1], shape[0]), dtype=np.uint8)  # row = y, column = x

        xforms = UsdGeom.XformCache()
        meshes = triangles = 0
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World"), Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            if any(path.startswith(skip) for skip in SKIP):
                continue
            if not prim.IsA(UsdGeom.Mesh):
                continue
            image = UsdGeom.Imageable(prim)
            if image.ComputeVisibility() == UsdGeom.Tokens.invisible or image.ComputePurpose() == UsdGeom.Tokens.guide:
                continue
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            if not points or not counts:
                continue
            matrix = np.array(xforms.GetLocalToWorldTransform(prim)).reshape(4, 4)
            world = np.asarray(points, dtype=np.float64) @ matrix[:3, :3] + matrix[3, :3]
            if world[:, 2].max() < BAND[0] or world[:, 2].min() > BAND[1]:
                continue
            counts = np.asarray(counts)
            indices = np.asarray(indices)
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            # Fan-triangulate polygons; every sorted warehouse mesh here is made of triangles and quads.
            for corners in range(3, int(counts.max()) + 1):
                faces = starts[counts >= corners]
                if not len(faces):
                    continue
                tri = np.stack([indices[faces], indices[faces + corners - 2], indices[faces + corners - 1]], axis=1)
                heights = world[tri, 2]
                keep = (heights.max(axis=1) >= BAND[0]) & (heights.min(axis=1) <= BAND[1])
                if not keep.any():
                    continue
                cells = np.round((world[tri[keep], :2] - origin) / RESOLUTION).astype(np.int32)
                cv2.fillPoly(grid, list(cells), 1)
                cv2.polylines(grid, list(cells), True, 1)
                triangles += int(keep.sum())
            meshes += 1

        out = ENV / "nav"
        out.mkdir(exist_ok=True)
        np.savez_compressed(out / "occupancy.npz", grid=grid, origin=origin, resolution=RESOLUTION, band=BAND)
        preview = np.where(grid[::-1] > 0, 40, 235).astype(np.uint8)
        cv2.imwrite(str(out / "occupancy.png"), preview)
        print(f"NAV_MAP {grid.shape} cells, origin {origin}, {meshes} meshes, {triangles} triangles", flush=True)
        print(f"occupied fraction {grid.mean():.3f}", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
