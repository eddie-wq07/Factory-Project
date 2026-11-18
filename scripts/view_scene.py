"""View the project robot in Isaac Sim, or save a headless viewport screenshot."""

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from Factory_Project.assets.data.screenshot import RESOLUTION, ROBOT_USD, VIEWS, capture_viewport, setup_view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=ROBOT_USD, help="Robot USD to reference")
    parser.add_argument("--view", choices=VIEWS, default="three-quarter")
    parser.add_argument("--screenshot", metavar="NAME", help="Save a headless PNG and exit")
    parser.add_argument("--diagnostic", action="store_true", help="Save under recordings/misc/")
    args = parser.parse_args()
    if not args.asset.is_file():
        parser.error(f"Robot USD is missing: {args.asset}")

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": args.screenshot is not None,
            "create_new_stage": True,
            "width": RESOLUTION[0],
            "height": RESOLUTION[1],
            "sync_loads": True,
            "multi_gpu": False,
        }
    )
    status = 0
    try:
        import omni.usd

        viewport = setup_view(omni.usd.get_context().get_stage(), args.asset, args.view)
        if args.screenshot is None:
            while app.is_running():
                app.update()
        else:
            output = capture_viewport(app.update, viewport, args.screenshot, diagnostic=args.diagnostic)
            print(f"SCREENSHOT_SAVED {output}", flush=True)
    except KeyboardInterrupt:
        status = 130
    except Exception:
        traceback.print_exc()
        status = 1
    finally:
        app.close(exit_code=status)
    return status


if __name__ == "__main__":
    sys.exit(main())
