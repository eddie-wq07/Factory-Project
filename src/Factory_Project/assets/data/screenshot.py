"""Set up robot inspection views and record Kit viewports as dated PNGs or MP4s."""

import asyncio
import itertools
import math
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

ASSET_DIR = Path(__file__).resolve().parent
ROBOT_USD = ASSET_DIR / "astribot_s1" / "astribot_s1.usd"
RECORDINGS = ASSET_DIR.parents[3] / "recordings"
EXTERIOR_VIEWS = {
    "three-quarter": (1.0, -0.75, 0.23),
    "front": (1.0, 0.0, 0.02),
    "rear": (-1.0, 0.0, 0.06),
    "underside": (0.0, 0.0, -1.0),
}
ONBOARD_VIEWS = {
    "head": ("astribot_head_link_2", "head_rgbd"),
    "left-wrist": ("astribot_arm_left_link_7", "left_wrist_rgbd"),
    "right-wrist": ("astribot_arm_right_link_7", "right_wrist_rgbd"),
}
VIEWS = (*EXTERIOR_VIEWS, *ONBOARD_VIEWS)
RESOLUTION = (1280, 960)


def _reserve_recording(label: str, extension: str) -> Path:
    """Claim a dated capture path without overwriting another session."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", label):
        raise ValueError("Use letters, digits, hyphens or underscores in recording names")
    now = datetime.now(ZoneInfo("America/Vancouver"))
    folder = RECORDINGS / now.strftime("%m-%d-%Y")
    folder.mkdir(parents=True, exist_ok=True)
    stamp = f"{now.hour % 12 or 12}:{now.minute:02}{'am' if now.hour < 12 else 'pm'}"
    for sequence in itertools.count(1):
        suffix = "" if sequence == 1 else f"-{sequence}"
        output = folder / f"{stamp}_{label}{suffix}.{extension}"
        try:
            output.touch(exist_ok=False)
            return output
        except FileExistsError:
            continue


def reserve_screenshot(label: str) -> Path:
    """Reserve a dated PNG in the project recordings folder."""
    return _reserve_recording(label, "png")


def setup_view(stage, asset: Path, view: str):
    """Reference the asset into a temporary studio and frame its visible geometry."""
    import carb.settings
    from omni.kit.viewport.utility import get_active_viewport
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

    carb.settings.get_settings().set("/app/viewport/grid/enabled", False)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    robot = stage.DefinePrim("/World/Robot", "Xform")
    if not robot.GetReferences().AddReference(str(asset.resolve())):
        raise RuntimeError(f"Could not reference {asset}")
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bounds = bbox.ComputeWorldBound(robot).ComputeAlignedRange()
    if bounds.IsEmpty():
        raise RuntimeError(f"No visible geometry composed from {asset}")
    low, high = bounds.GetMin(), bounds.GetMax()
    center = (low + high) * 0.5
    height = high[2] - low[2]
    if view == "underside":
        # Frame the chassis and wheels closely, leaving the floor out of the shot.
        high = Gf.Vec3d(high[0], high[1], low[2] + 0.18 * height)
        center = (low + high) * 0.5
    else:
        floor = UsdGeom.Plane.Define(stage, "/World/Studio/Floor")
        floor.CreateAxisAttr("Z")
        floor.CreateWidthAttr(height * 500)
        floor.CreateLengthAttr(height * 500)
        floor.AddTranslateOp().Set(Gf.Vec3d(center[0], center[1], low[2] - 0.001))
        floor.CreateDisplayColorAttr([(0.13, 0.16, 0.20)])
        UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
        UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")

    dome = UsdLux.DomeLight.Define(stage, "/World/Studio/Fill")
    dome.CreateIntensityAttr(550)
    dome.CreateColorAttr((0.85, 0.91, 1.0))
    sun = UsdLux.DistantLight.Define(stage, "/World/Studio/Key")
    sun.CreateIntensityAttr(1600)
    sun.CreateAngleAttr(6.0)
    sun.AddRotateXYZOp().Set(Gf.Vec3f(145 if view == "underside" else 25, -40, -20))

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("Screenshot capture needs an Isaac Sim Kit viewport")
    viewport.resolution = RESOLUTION
    if view in ONBOARD_VIEWS:
        link, name = ONBOARD_VIEWS[view]
        camera = UsdGeom.Camera.Get(stage, f"{robot.GetPath()}/Geometry/{link}/{name}")
        if not camera:
            raise RuntimeError(f"The robot has no camera named {name} on {link}")
        # Fixed world-space inspection targets reveal self-occlusion and image orientation.
        # The tall red post sits in front of the robot. Green and blue blocks mark its left and right.
        for tag, position, size, color in (
            ("Front", (1.3, 0.1, low[2] + 0.8), (0.12, 0.12, 1.6), (0.8, 0.12, 0.05)),
            ("Left", (0.0, 0.62, low[2] + 0.06), (0.15, 0.12, 0.12), (0.07, 0.55, 0.16)),
            ("Right", (0.0, -0.62, low[2] + 0.06), (0.15, 0.12, 0.12), (0.05, 0.25, 0.85)),
        ):
            marker = UsdGeom.Cube.Define(stage, f"/World/Studio/{tag}")
            marker.CreateSizeAttr(1)
            marker.AddTranslateOp().Set(Gf.Vec3d(*position))
            marker.AddScaleOp().Set(Gf.Vec3f(*size))
            marker.CreateDisplayColorAttr([color])
        viewport.camera_path = camera.GetPath()
        return viewport

    camera = UsdGeom.Camera.Define(stage, "/World/Studio/Camera")
    camera.CreateHorizontalApertureAttr(36)
    camera.CreateVerticalApertureAttr(27)
    camera.CreateFocalLengthAttr(50)
    camera.CreateClippingRangeAttr((0.005, 2000))
    direction = Gf.Vec3d(*EXTERIOR_VIEWS[view]).GetNormalized()
    up = Gf.Vec3d(1, 0, 0) if view == "underside" else Gf.Vec3d(0, 0, 1)
    right = Gf.Cross(up, direction).GetNormalized()
    screen_up = Gf.Cross(direction, right)
    offsets = (Gf.Vec3d(*point) - center for point in itertools.product(*zip(low, high, strict=True)))
    distance = max(
        Gf.Dot(offset, direction) + max(abs(Gf.Dot(offset, right)) / 0.36, abs(Gf.Dot(offset, screen_up)) / 0.27) * 1.12
        for offset in offsets
    )
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError(f"Cannot frame invalid robot bounds {bounds}")
    matrix = Gf.Matrix4d().SetLookAt(center + distance * direction, center, up).GetInverse()
    camera.AddTransformOp().Set(matrix)
    viewport.camera_path = camera.GetPath()
    return viewport


def _capture_png(update, viewport, output: Path) -> None:
    """Write the active Kit viewport at its current resolution."""
    import omni.kit.renderer_capture
    from omni.kit.viewport.utility import capture_viewport_to_file, next_viewport_frame_async

    async def capture_pose():
        # A capture scheduled immediately after a physics step can receive the previous pose.
        await next_viewport_frame_async(viewport, 4)
        await capture_viewport_to_file(viewport, str(output)).wait_for_result()

    pending = asyncio.ensure_future(capture_pose())
    deadline = time.monotonic() + 60
    while not pending.done():
        if time.monotonic() >= deadline:
            pending.cancel()
            raise TimeoutError("Viewport capture timed out")
        update()
    pending.result()
    omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
    with output.open("rb") as image:
        if image.read(8) != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError(f"Kit did not write a PNG at {output}")


def capture_viewport(update, viewport, label: str) -> Path:
    """Wait for the Kit renderer and save the active view without advancing physics."""
    for _ in range(60):
        update()
    output = reserve_screenshot(label)
    try:
        _capture_png(update, viewport, output)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return output


@dataclass
class ViewportVideo:
    """One constant-rate video. Each capture samples the current simulation pose."""

    path: Path
    _capture: Callable[[Path], None]
    _frames_dir: Path
    frame_count: int = 0

    def capture_frame(self) -> None:
        self._capture(self._frames_dir / f"{self.frame_count:06d}.png")
        self.frame_count += 1


@contextmanager
def record_viewport_video(update, viewport, label: str, fps: int = 30) -> Iterator[ViewportVideo]:
    """Encode one frame per capture_frame() call, independent of wall-clock speed.

    Supply a render-only update function. The caller controls physics steps between
    captures. The completed H.264 MP4 becomes available at video.path on exit.
    """
    if not isinstance(fps, int) or fps < 1:
        raise ValueError("fps must be a positive integer")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        from imageio_ffmpeg import get_ffmpeg_exe

        ffmpeg = get_ffmpeg_exe()

    output = _reserve_recording(label, "mp4")
    try:
        with TemporaryDirectory(prefix=".capture-", dir=output.parent) as staging:
            frames = Path(staging)
            video = ViewportVideo(output, lambda path: _capture_png(update, viewport, path), frames)
            yield video
            if not video.frame_count:
                raise ValueError("Cannot encode an empty video")
            result = subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-framerate",
                    str(fps),
                    "-i",
                    str(frames / "%06d.png"),
                    "-frames:v",
                    str(video.frame_count),
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "18",
                    "-movflags",
                    "+faststart",
                    str(frames / "encoded.mp4"),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode:
                raise RuntimeError(f"FFmpeg could not encode the recording: {result.stderr.strip()}")
            (frames / "encoded.mp4").replace(output)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
