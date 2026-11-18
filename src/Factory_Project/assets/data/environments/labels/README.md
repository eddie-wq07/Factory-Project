# Decorative factory location stickers

`../factory_labels.usda` is a persistent sublayer of `../factory.usda`. It attaches paired mesh stickers to 1,841 individual carton prims and eight mixed-cargo rack-face subsections. The separate label layer contains no physics schemas. Original warehouse files are not modified.

Labels follow `top-down-annotated-aisle-organization.png`: each aisle has four bays up its left face (1–4), followed by four bays down its right face (5–8). Each letter occupies exactly 8/3 bay-equivalents. A carton is assigned from its center position; shared racks use the face toward the nearest aisle. A split bay can contain two letter prefixes. Floor-clutter cartons use their aisle-5 position and face south for visibility.

White address labels occupy the top-right of the largest planar aisle-facing carton surface; an identical-size tag sits immediately left. Dimensions are 0.1524 m (6 in) where the pair fits and 0.0762 m (3 in) otherwise. Text uses the same typeface and relative size on both formats. Meshes are children of their original objects, offset 1.5 mm from the selected surface. Instances are expanded only where attachment needs child overrides.

For the mixed-cargo R3 rack, there is one address/tag pair per bay per aisle-facing side. South-to-north bays alternate between the 1.3 m and 2.8 m middle rails, avoiding both the ground and top rails. These are section labels for mixed/grouped cargo; the grouped cargo meshes are not split into new individual objects.

Every tag duplicates AprilTag 36h11 ID 0, generated with OpenCV's predefined dictionary and a white margin. These are decorative: no detection, unique IDs, localization, collision or friction integration was added.

`manifest.csv` records parent paths, assigned addresses, aisles, placements and sizes. Rebuild with `~/env_isaaclab/bin/python scripts/label_factory.py` from the repository root. Texture PNGs are local, and the scene does not need the generator, font or OpenCV to display them. `scripts/build_factory.py` retains this sublayer when regenerating the base scene.
