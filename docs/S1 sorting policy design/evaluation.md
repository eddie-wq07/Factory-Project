# S1 sorting readiness and evaluation

## What must exist before policy research is meaningful?

### Takeaway
The saved factory is a visual staging prototype, not yet an executable sorting benchmark. First make six dynamic pickup objects, physically reachable destinations, repeatable resets and measurable success; keep the rest of the warehouse static.

### Cited Findings
- Factory diagnosis reports 1,841 cardboard objects with triangle-mesh colliders and zero rigid bodies under `/World/Factory`; collision alone is not movable-object physics. The final staging section explicitly calls the new rack static, without grasp/reset behavior. — [Repo diagnosis](../../docs/factory-layout.md)
- The rack builder asserts exactly six entries spanning aisles 1–6, internally references original source objects and writes `sourceAisle`, `sorting_prim`, `sorting_position`, and `rack_address`. It disables rigid bodies on the rack and does not enable item dynamics. — [Builder](../../scripts/build_sorting_rack.py)
- Labels use deliberately identical AprilTag36h11 ID0, with no detector integration. Address text and decorative tags do not constitute unique machine-readable identities. — [Labels](../../scripts/label_factory.py)
- Manager-based configuration has missing robot/action definitions and `NotImplementedError`; direct environment lifecycle methods are unimplemented. PPO configuration availability is not task readiness. — [Manager configuration](../../src/Factory_Project/tasks/factory/config/AstribotS1/env_cfg.py), [Direct environment](../../src/Factory_Project/tasks/factory_direct/config/AstribotS1/env.py)

### Inferences
- Proposed task contract: ONE rack, SIX items, one from each aisle. Sort in address order A2 → F8 → H4 → K3 → O8 → Q4. Treat that as processing order, separately from destination occupancy and maximum stack height. Older vision wording must not expand this to 36 objects.
- Add explicit inventory records `{object_id, source_slot, destination_address, destination_pose, allowed_supports}`. A source object's original position is not automatically an empty destination; source copies remain in the warehouse. Check approach corridors, arm reach, gripper fit, payload and shelf headroom for every chosen item.
- Start with one graspable item and accessible destination. Do not assume every current shape is one-arm graspable; large cartons, open bins and the upper shelf require feasibility checks. Designated drop sites must have capacity, support polygons and clearance.
- Promote only selected copies to dynamic bodies, with suitable dynamic collision geometry, mass, inertia and friction. Validate settle, lift, carry and release physically. Keep stickers attached to their object transforms.
- Shelf-slot labels stay on rack rails after pickup; associate slot label with item, then maintain tracked carry identity. Gate OCR/address recognition on confidence, reread from a changed view on ambiguity, and stop after a bounded retry budget. Unique tag IDs are an optional future perception change, not already available functionality. Confirm delivery by tracked item inside the permitted destination support region, released and settled—not merely arriving near the correct sign. Under strict alphabetical order, a blocked next item must trigger recovery or failure rather than silent skipping.
- One item per destination does not exercise stack≤3; add repeated-destination trials and reject a fourth item. Bins/bottles require explicit stacking eligibility rather than a universal box rule.

### Gaps
No measured six-item task success, demo dataset, executable reset/reward task, robot-specific data-generation compatibility, dynamic asset validation or destination reachability was found in inspected files. Current facts do not establish eventual real-robot success.

## How should imitation and recovery data be evaluated?

### Takeaway
Use rollout performance to select policies, then collect corrections at the states where the learned policy fails. Synthetic data helps coverage after the task and success checks exist; it does not remove those prerequisites.

### Cited Findings
- Robomimic compared six offline algorithms on five simulated and three real manipulation tasks. History-dependent BC models were strong on their human datasets; validation imitation loss did not reliably select the strongest policy. Observation choices and data coverage mattered. These are benchmark findings, not a predicted S1 ranking. — [Robomimic study](https://robomimic.github.io/study/)
- DAgger addresses sequential distribution shift: actions influence future observations, violating ordinary independent-data assumptions. It iteratively trains on data from the learner-induced distribution. — [Ross et al., 2011](https://proceedings.mlr.press/v15/ross11a.html)
- Isaac Lab Mimic transforms object-relative demonstration segments, checks generated rollouts for success and retains successful ones. It requires task-space actions (or conversion), subtask annotations and reference objects. Its Franka example is not a ready S1 integration. — [Isaac Lab Mimic documentation](https://isaac-sim.github.io/IsaacLab/main/source/features/imitation-learning/teleop_imitation.html)

### Inferences
- Collect clean skill demonstrations first; run the learned controller; use an expert/controller to correct learner failures and retrain on aggregated data. Calling this DAgger requires expert labels on learner-visited states, not merely adding arbitrary videos.
- Start with a proposed pilot of 50–100 successful trajectories per manipulation skill, covering all intended objects and poses, then measure learning curves at 25/50/100/200 demonstrations. These are planning budgets, not literature-backed guarantees of sufficiency.
- Keep train/validation/test separated at trajectory and scenario level. Adjacent frames from one rollout cannot be a useful held-out test. Generated variants of a seed demonstration should be grouped to avoid overly easy test leakage.
- Match datasets, action spaces and observation budgets when comparing BC-RNN, ACT or diffusion; choose checkpoints on validation rollouts, then freeze before final tests. State-privileged and camera-only policies need separate results.

### Gaps
The minimum data and compute for S1 cannot be responsibly predicted before pilot learning curves. Mimic compatibility depends on the installed Isaac Lab version and custom S1 environment implementation; the linked main documentation is current, not a verified local API contract.

## What counts as completing the long-horizon task?

### Takeaway
Measure complete six-item episodes and failures at skill boundaries, not only reward or attractive video. Separate known-object performance from unseen-object/layout generalization.

### Cited Findings
- The repo task calls for visual pickup, navigation to designated aisle drop spots, alphabetical sorting and a maximum of three stacked boxes. — [Vision](../../docs/vision.md)
- Successful imitation checkpoint selection needs actual policy evaluation rather than supervised validation loss alone. — [Robomimic study](https://robomimic.github.io/study/)

### Inferences
- Proposed gates: (1) deterministic reset and dynamic-object sanity; (2) grasp/lift/hold and stable release at fixed dock; (3) unloaded docking then loaded transit without drop; (4) one complete item; (5) all six in order; (6) recovery and distribution shifts. Return to the failing skill when integration fails.
- Primary endpoint: fraction of episodes with all six correct placements, correct processing order, stable released objects, no prohibited collisions/drops and stack cap respected, within a declared time/retry budget. Report partial objects completed separately.
- Also log conditional per-skill success, first failure reason, wrong-address events, collisions, drops, placement stability, time, distance, interventions and recovery success. Bound retries; unlimited retries obscure reliability.
- Proposed evaluation budget: 100 held-out episodes per training seed, at least three independent training seeds for learned comparisons. Report each seed and aggregate with uncertainty; use Wilson binomial intervals for episode success, and avoid pretending repeated seeds remove scenario correlation. This is an initial budget, not a statistical power guarantee.
- Toy illustration only: if five stages per item independently succeeded with probability 0.95, a six-item run would succeed with probability 0.95^30≈21.5%; at 0.99 per stage it is 0.99^30≈74.0%. Real errors are correlated, so measure full episodes instead of multiplying observed skill rates.
- Held-out test families: same objects/new poses; new distractor or obstacle layouts; lighting/camera noise; grasp/dock perturbations; unseen object variants. Training on all six objects and testing the same six supports only known-object claims.
- Minimal ablations: all-scripted baseline; scripted navigation + learned manipulation; learned docking + scripted manipulation where feasible; combined system; combined with recovery disabled. Compare methods on exactly the same seeded scenarios. Later compare IL-only vs IL plus RL refinement only for the skill shown to bottleneck performance.
- Keep a fixed persistent spawn for the initial product task; broaden starts deliberately for robustness experiments, not silently. Add repeated-destination episodes for stacking even though the six-item nominal task uses different addresses.

### Gaps
Acceptance thresholds, timeouts, tolerances and collision definitions need to be chosen before evaluation. Suggested counts and gates above are experimental design recommendations, not achieved results.
