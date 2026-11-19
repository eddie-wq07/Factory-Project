"""Record a short cinematic cut of the sorting mission: orbiting shots of the picks and drop-offs.

    ~/env_isaaclab/bin/python scripts/record_cinematic.py

The scripted teacher runs the standard mission (seed 0, which is repeatable). During each shot's time
window a free camera orbits the action and frames are saved; nothing is rendered in between. Each shot
is written to recordings/sorting-mission/cinematic_clips/, then all are joined into
recordings/sorting-mission/cinematic.mp4.
"""

import sim_bootstrap  # noqa: F401  (must come before NumPy and Isaac Sim)

import argparse
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--every", type=int, default=6, help="Save one frame every N control steps inside a shot.")
parser.add_argument("--fps", type=int, default=30)
add_launcher_args(parser)
parser.set_defaults(device="cpu")
args = parser.parse_args()
args.enable_cameras = True

import gymnasium as gym
import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import torch

import Factory_Project.tasks  # noqa: F401
from Factory_Project.sorting.teacher import Teacher
from Factory_Project.tasks.factory_direct.config.AstribotS1.env_cfg import FactoryEnvCfg

OUT = Path(__file__).resolve().parents[1] / "recordings" / "sorting-mission"
CAMERA, RESOLUTION = "cinema", (1920, 1080)


@dataclass
class Shot:
    """One orbiting shot. Angles are measured from the direction the robot stands in, seen from the load."""

    name: str
    address: str
    at_rack: bool
    start: float
    """Mission time at which the shot starts [s]."""
    length: float
    """Mission time the shot covers [s]."""
    angles: tuple[float, float]
    """Camera bearing at the start and end of the shot [deg]; 0 looks over the robot's shoulder at the load."""
    radius: tuple[float, float]
    height: tuple[float, float]


# Times follow the repeatable seed-0 mission: items are verified in place at 151, 300, 453, 604, 732 and 860 s.
SHOTS = [
    Shot("1_pick_A2", "A2", True, 16.0, 20.0, (70.0, 15.0), (4.2, 3.0), (2.4, 1.5)),
    Shot("2_drop_A2", "A2", False, 124.0, 20.0, (-65.0, -10.0), (3.4, 2.8), (1.3, 1.7)),
    Shot("3_drop_F8", "F8", False, 276.0, 20.0, (45.0, 85.0), (3.3, 3.0), (1.9, 1.2)),
    Shot("4_drop_H4", "H4", False, 430.0, 20.0, (-80.0, -30.0), (2.8, 3.2), (1.0, 1.6)),
    Shot("5_drop_K3", "K3", False, 580.0, 20.0, (85.0, 50.0), (3.4, 2.9), (2.2, 1.5)),
    Shot("6_bottle_O8", "O8", False, 710.0, 20.0, (-40.0, -82.0), (3.0, 2.7), (1.6, 1.1)),
    Shot("7_drop_Q4", "Q4", False, 839.0, 20.0, (-75.0, -42.0), (2.9, 3.9), (1.3, 2.3)),
]


def ease(x: float) -> float:
    return x * x * (3.0 - 2.0 * x)


def frame_camera(env, shot: Shot, progress: float):
    """Camera eye and target for a shot at ``progress`` in [0, 1]."""
    manifest = env.manifest
    if shot.at_rack:
        normal = np.array(manifest.rack.normal, dtype=float)
        load = np.array(manifest.item(shot.address).center[:2])
    else:
        normal = np.array(manifest.destination(shot.address).normal, dtype=float)
        load = env.placement(shot.address).pose.position[:2]
    base, _ = env.base_pose()
    focus = np.array([*(0.55 * load + 0.45 * np.asarray(base[:2])), 0.95])
    s = ease(progress)
    bearing = math.atan2(normal[1], normal[0]) + math.radians(shot.angles[0] + (shot.angles[1] - shot.angles[0]) * s)
    radius = shot.radius[0] + (shot.radius[1] - shot.radius[0]) * s
    height = shot.height[0] + (shot.height[1] - shot.height[0]) * s
    eye = np.array([focus[0] + radius * math.cos(bearing), focus[1] + radius * math.sin(bearing), height])
    return eye, focus


def main() -> int:
    cfg = FactoryEnvCfg()
    cfg.sim.device = args.device
    cfg.seed = 0
    cfg.viewer_cameras = {CAMERA: RESOLUTION}
    clips_dir = OUT / "cinematic_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    dt = cfg.sim.dt * cfg.decimation
    clips = []
    with launch_simulation(cfg, args):
        env = gym.make("FactoryProject-Factory-ASTRIBOTS1-Direct", cfg=cfg).unwrapped
        env.reset(seed=0)
        teacher = Teacher(env, seed=0, verbose=False)
        generator = teacher.run()
        pending = list(SHOTS)
        shot, writer, focus = None, None, None
        step = 0
        try:
            action = next(generator)
            while pending or shot is not None:
                _, _, done, time_out, _ = env.step(torch.as_tensor(action, dtype=torch.float32).unsqueeze(0))
                step += 1
                now = step * dt
                if shot is None and pending and now >= pending[0].start:
                    shot = pending.pop(0)
                    path = clips_dir / f"{shot.name}.mp4"
                    writer = imageio.get_writer(path, fps=args.fps, codec="libx264", quality=8, macro_block_size=None)
                    clips.append(path)
                    focus = None
                    print(f"shot {shot.name} at {now:.0f} s", flush=True)
                if shot is not None:
                    progress = (now - shot.start) / shot.length
                    if progress >= 1.0 or bool(done[0]) or bool(time_out[0]):
                        writer.close()
                        shot = writer = None
                    elif step % args.every == 0:
                        eye, target = frame_camera(env, shot, progress)
                        # Follow the action smoothly rather than snapping when the chassis moves.
                        focus = target if focus is None else focus + 0.25 * (target - focus)
                        env.view(CAMERA, eye, focus)  # first pass settles the renderer on the new viewpoint
                        writer.append_data(env.view(CAMERA, eye, focus))
                if bool(done[0]) or bool(time_out[0]):
                    break
                action = generator.send(None)
        except StopIteration:
            pass
        if writer is not None:
            writer.close()
        print("mission status", env.status, flush=True)
        listing = clips_dir / "clips.txt"
        listing.write_text("".join(f"file '{clip.name}'\n" for clip in clips))
        final = OUT / "cinematic.mp4"
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing)]
        command += ["-c:v", "libx264", "-crf", "20", "-preset", "slow", "-pix_fmt", "yuv420p", "-an", str(final)]
        subprocess.run(command, check=True)
        print(f"CINEMATIC {final} from {len(clips)} clips", flush=True)
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
