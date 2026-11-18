# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open an empty Isaac Sim editor to browse assets for ``vision.md``.

Run with the Python environment that has Isaac Sim installed::

    ~/env_isaaclab/bin/python scripts/preview.py

Use Window > Browsers > Content > Isaac Sim > Environments to browse scenes.
The timeline starts stopped; the script only pumps the editor's update loop.
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true", help="Run without a desktop window (for smoke checks).")
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Exit after this many updates; 0 keeps the editor open."
    )
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error("--max-frames must be zero or positive")
    if sys.platform.startswith("linux") and not args.headless and not os.environ.get("DISPLAY"):
        parser.error("No desktop display found. Run from a terminal in your desktop session, or use --headless.")

    # Kit must start before importing any Omniverse or USD modules.
    from isaacsim import SimulationApp # type: ignore

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "create_new_stage": True,
            "sync_loads": False,
            "multi_gpu": False,
            "window_width": 1600,
            "window_height": 1000,
            "display_options": 3286,
            "extra_args": ["--enable", "isaacsim.gui.content_browser"],
        }
    )
    try:
        import carb.settings # type: ignore
        import omni.timeline # type: ignore
        import omni.usd # type: ignore
        from omni.kit.window.content_browser import get_content_window # type: ignore
        from pxr import UsdGeom, UsdLux # type: ignore

        omni.timeline.get_timeline_interface().stop()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        # Neutral illumination for assets dropped into the otherwise empty scene.
        UsdLux.DomeLight.Define(stage, "/World/PreviewLight").CreateIntensityAttr(1000.0)

        asset_root = carb.settings.get_settings().get_as_string("/persistent/isaac/asset_root/default").rstrip("/")
        environments = f"{asset_root}/Isaac/Environments"
        if not args.headless:
            browser = get_content_window()
            if browser.window is None:
                browser.show_window(None, True)
            else:
                browser.window.set_visible(True)
            browser.navigate_to(environments)

        print("\n[preview] Ready: empty stage with a preview light; physics is stopped.", flush=True)
        print(f"[preview] Environment assets: {environments}", flush=True)
        print("[preview] Browse Content > Isaac Sim > Environments (or Props for tables and boxes).", flush=True)
        print("[preview] Drag a USD into the Stage under World; select it and press F in the viewport.", flush=True)
        print(
            "[preview] Record the asset's USD URL in vision.md, or save a composed scene with File > Save As.\n",
            flush=True,
        )

        frame = 0
        while simulation_app.is_running() and (args.max_frames == 0 or frame < args.max_frames):
            simulation_app.update()
            frame += 1
    except KeyboardInterrupt:
        pass
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
