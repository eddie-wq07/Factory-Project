# Authoring and viewing scenes

## Preview environment assets

Open an empty Isaac Sim editor using the existing Isaac Sim Python environment:

```bash
cd ~/Desktop/Simulation/Projects/S1-Project
~/env_isaaclab/bin/python scripts/preview.py
```

The scene starts with only `/World` and a neutral preview light. Physics starts stopped, so you can browse and
arrange assets before running a task. This uses Isaac Sim directly and does not require the unfinished task configs.

1. In **Window > Browsers > Content**, expand **Isaac Sim > Environments**. The script opens this folder for you.
2. Start with **Simple_Warehouse** (`warehouse.usd` or `warehouse_multiple_shelves.usd`) for the factory aisle,
   or **Simple_Room** (`simple_room.usd`) for a smaller scene. Browse **Props** for individual objects.
3. Drag a USD onto **World** in the Stage panel to place it at the origin. Select it, then press **F** with the
   viewport focused to frame it. Use **File > New** to clear the scene between candidates.
4. Record each chosen asset's full USD URL in `vision.md`, along with its role (background, pickup table,
   drop-off station, obstacle). If you arrange several assets, use **File > Save As** to save a project scene
   under `src/Factory_Project/assets/data/environments/`; that scene can later be loaded by Isaac Lab.

Assets stream from NVIDIA when first opened, so larger scenes can take time to download. The editor must be
launched from a graphical desktop terminal. For a bounded check without a window:

```bash
~/env_isaaclab/bin/python scripts/preview.py --headless --max-frames 10
```

See NVIDIA's [Content Browser guide](https://docs.isaacsim.omniverse.nvidia.com/latest/utilities/content_browser.html)
and [environment catalog](https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_environments.html).

## Persistent factory starting point

```bash
~/env_isaaclab/bin/python scripts/view_factory.py
~/env_isaaclab/bin/python scripts/view_factory.py --screenshot test-boundary
```

The saved scene is `src/Factory_Project/assets/data/environments/factory.usda`. It places S1 at `(4.45, 7.30, 0.09523)` metres, heading 180°, beside the east wall and shelf end. Reopening this USD restores the same pose and the robot asset's neutral joint configuration. The RL task templates do not yet implement episode resets.

Four references to the supplied NVIDIA SimReady tape form a square with a 0.596852 m inner side and 0.15 m tape width. Its inner edges match the full chassis-and-wheel bounding square; the narrower axis has about 4.6 mm clearance per side. Tape is decorative, with physics disabled, and raised 1 mm above the floor. `scripts/build_factory.py` regenerates the scene from measured robot bounds. Robot and tape dependencies are local; the warehouse still uses NVIDIA's hosted USD and requires network access on an uncached machine.

Requested captures go in `recordings/MM-DD-YYYY/`. Add `--diagnostic` to either screenshot viewer for unprompted inspection captures under `recordings/misc/MM-DD-YYYY/`. Earlier diagnostic screenshots were moved there; the requested render test and wheel video remain in the day folder. Capture filenames use Vancouver local time.

[Boundary screenshot](../recordings/11-18-2026/12:25pm_test-boundary.png). A two-second idle physics check measured approximately 15 mm of base settling/drift; this is a fixed initial pose, not an active parking controller.
