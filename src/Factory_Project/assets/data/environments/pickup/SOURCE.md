# Single sorting rack

NVIDIA SimReady rack, CC BY 4.0; see `rack/LICENSE.txt`:
https://huggingface.co/datasets/nvidia/simready-assets/blob/653d91e182011f357cf1644013cc94ddf194d349/rack_composition_l10/simready_usd/sm_rack_composition_l10_01.usd

The original rack and its dependencies are local under `rack/`. The `sorting_rack.usda` layer deactivates the supplied boxes and retains one unscaled rack at (1.0, 3.3, 0) m: the former aisle-6 rack position from the reverted arrangement. It does not replace the original warehouse aisle or forklift.

Six items are internally referenced from the existing factory, one from each source aisle. `selection.json` records the original prims and new placements. Two items occupy each of three lower shelves; the top is empty. Objects retain source scale, materials and carton stickers. Racks and items remain static; no grasp or physics-control logic was added. The duplicated warehouse objects retain the factory's existing hosted-asset dependency.

The title texture reads exactly "To be sorted.". All additions compose through the main factory USD and can be rebuilt with `scripts/build_sorting_rack.py`. The base factory builder preserves the sorting layer. Only one sorting rack is created.

Each slot has an address sticker and matching duplicated decorative AprilTag on the front rack rail beneath its item. All rail stickers are 3-inch squares to fit the narrow rail. Addresses follow the original source locations and are recorded as `rack_address` in `selection.json`; existing object stickers remain unchanged.
