"""Open the persisted warehouse spawn; --screenshot NAME saves a dated PNG."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from Factory_Project.assets.data.screenshot import ASSET_DIR, capture_viewport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot")
    parser.add_argument("--diagnostic", action="store_true", help="Route inspection captures to recordings/misc/")
    args = parser.parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": bool(args.screenshot), "width": 1600, "height": 1000, "multi_gpu": False})
    try:
        import omni.usd
        from omni.kit.viewport.utility import get_active_viewport

        context = omni.usd.get_context()
        if not context.open_stage(str(ASSET_DIR / "environments/factory.usda")):
            raise RuntimeError("Failed to open factory scene")
        viewport = get_active_viewport()
        viewport.resolution = (1600, 1000)
        viewport.camera_path = "/World/SpawnCamera"
        if args.screenshot:
            print(
                "SCREENSHOT_SAVED",
                capture_viewport(app.update, viewport, args.screenshot, diagnostic=args.diagnostic),
                flush=True,
            )
        else:
            while app.is_running():
                app.update()
    finally:
        app.close()


if __name__ == "__main__":
    main()
