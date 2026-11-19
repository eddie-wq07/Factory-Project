# Manipulation policy recommendation for S1 sorting

## Which learning approach and architecture?

### Takeaway
Use a deterministic task executive plus short manipulation skills. First learned candidate: behavior cloning with ACT for docked pick/place; retain a state-based BC-MLP as the cheap baseline. Use RL selectively after finding a measurable skill failure, rather than training the entire six-item mission from scratch. This is a project recommendation, not a demonstrated S1 result.

### Cited Findings
- ACT predicts sequences rather than single actions, combines camera observations and joint state with a transformer, and trains using a conditional VAE; its test-time latent is fixed to zero. Published demonstration efficiency is task-specific, not a guarantee for S1. [ACT authors](https://tonyzhaozh.github.io/aloha/)
- Diffusion Policy generates action sequences by iterative denoising, supports multimodal behavior, and executes with receding-horizon control. Its reported results concern its benchmark tasks, not this warehouse. [Diffusion Policy authors](https://diffusion-policy.cs.columbia.edu/)
- PPO learns from environment interactions using repeated minibatch optimization of a policy-gradient objective. [PPO paper](https://arxiv.org/abs/1707.06347)
- SAC is an off-policy actor-critic algorithm with a maximum-entropy objective; the paper motivates it partly by sample complexity of model-free RL. [SAC paper](https://arxiv.org/abs/1801.01290)
- Residual RL combines a conventional controller and a learned correction. [Residual RL paper](https://arxiv.org/abs/1812.03201)
- OpenVLA is a pretrained 7B vision-language-action model supporting adaptation to new setups; it still needs embodiment/task adaptation. Original Franka examples are fine-tuned, whereas reported zero-shot robots are specific existing embodiments. [OpenVLA authors](https://openvla.github.io/)

### Inferences
- Separate four concepts in the final document: **learning method** (IL learns from expert behavior; RL learns from rewards), **algorithm** (BC/PPO/SAC), **model family** (MLP/transformer/diffusion), and **system hierarchy** (executive calls skills). ACT and Diffusion Policy are named methods combining architectural and training choices, so they are not purely network layers. PPO is not a model architecture.
- BC-MLP: fastest state-based baseline to reveal whether observations/actions and data are sensible; less expressive baseline for temporally diverse visual behavior.
- ACT: first visual manipulation experiment, using action chunks to reduce step-by-step drift. Recommend because a bounded six-object setup benefits from focused expert data and straightforward chunk inference, not because literature proves ACT beats diffusion here.
- Diffusion Policy: strongest comparison if ACT fails on genuinely multimodal grasps/placements; benchmark inference latency on available GPU before selecting it. Do not claim it is always slower than all transformers; diffusion implementation and denoising steps matter.
- PPO: familiar fallback for isolated, resettable state-based reach/grasp/lift/place skills with dense intermediate rewards and large parallel simulation; no reason yet to learn object ordering or warehouse route logic through RL.
- SAC: candidate when experience reuse matters, but integrating/reproducing a second trainer is extra work; do not choose only because it is theoretically more sample-efficient.
- VLA fine-tuning: defer until language generalization or unfamiliar objects become a measured requirement. Long mission duration alone does not require a giant VLA.

### Gaps
- No S1 comparison exists in the reviewed sources. Compute, real demonstration quality, reachable grasps, action latency, and target success threshold remain unmeasured.

## What should the manipulation interface and data plan be?

### Takeaway
First prove grasp and placement physically possible with a planner/IK controller; use that controller or teleoperation to collect short, verified trajectories. Do not require a human to demonstrate the entire six-item warehouse route for every training episode.

### Cited Findings
- ACT uses multiple image viewpoints and robot joint observations to predict action sequences. [ACT authors](https://tonyzhaozh.github.io/aloha/)
- Diffusion Policy explicitly predicts sequences for receding-horizon control. [Diffusion Policy authors](https://diffusion-policy.cs.columbia.edu/)
- Imitation policies change the observations they later encounter; this violates independent-example assumptions and motivates collecting expert corrections on policy-induced states. [DAgger paper](https://proceedings.mlr.press/v15/ross11a.html)

### Inferences
- **Proposed observation:** head/wrist RGB, arm and gripper state, target/skill identifier, and optionally short observation history. First state-only baseline may use true relative object/goal pose; report that as privileged-state evaluation. A visual actor must eventually use estimated pose or images, not secretly retain simulator object poses.
- **Proposed action:** six-dimensional end-effector delta pose plus gripper command, passed through existing IK/joint-limit/velocity-limited control. Keep base stationary during manipulation initially. Joint-position targets are a legitimate alternative and closer to original ACT; choose one convention and use it consistently for expert records, training, and deployment.
- **Proposed horizon experiment:** predict about 0.5–1 s, execute only 0.1–0.2 s, then observe/replan; at 20 Hz this means 10–20 predicted steps and 2–4 executed steps. These are starting settings, not measured S1 rates. Validate controller latency first, shorten chunks near contact, and compare against temporal ensembling rather than blindly blending conflicting maneuvers.
- Train pick/lift and place/release separately at first; condition each on object/goal. A torso setting or grasp template can remain explicit until whole-body coordination is actually necessary. The executive owns alphabetical order, destination, occupancy, stack count, timeout, and recovery budget.
- **Pilot data budget, not a promise:** begin with 25 successful trajectories per skill/object condition to validate pipeline; measure learning curves at 25/50/100, expanding to 200 only where held-out failures justify it. For six objects and two skills, 50 per condition would be 600 short trajectories; shared policies may reuse data across similar objects. Scripted demonstrations count only if physics-valid and success-verified.
- Split by entire trajectories and held-out reset seeds, never random adjacent frames. Vary item pose, docking error, lighting, camera noise, and appropriate mass/friction. Add recovery demonstrations from real policy failures. Evaluate each item separately, not only aggregate averages.
- If no demos: begin scripted reach/grasp/place and use it as teacher; if this cannot solve a shape, fix grasp/reach/contact feasibility first. An isolated PPO curriculum is a reasonable alternative, not a reason to jump to sparse end-to-end RL.
- From project context supplied by coordinator: pickup objects currently static and markers are decorative duplicates. Dynamics/reset/success checks and target perception are prerequisites, not policies already present.

### Gaps
- Correct physical size and grip affordance differ across cardboard, crates, and bottle. A single parallel gripper may not support every chosen item or shelf placement. Stack count <=3 is an upper bound, not permission to stack unstable geometries.

## How should RL refinement and beginner terminology be framed?

### Takeaway
Treat RL refinement as an optional experiment with an explicit integration design. A pretrained ACT or diffusion model cannot simply be loaded into the default PPO actor and assumed compatible.

### Cited Findings
- DPPO specifically reformulates diffusion denoising and environment dynamics into a two-layer decision process and introduces training/schedule choices for fine-tuning. This provides evidence that diffusion RL is possible, but requires a dedicated approach. [DPPO authors](https://diffusion-ppo.github.io/)
- Residual RL leaves a conventional control component and learns the remaining correction. [Residual RL paper](https://arxiv.org/abs/1812.03201)
- Prince describes a policy network as mapping observations to actions; RL seeks actions with high reward, with delayed credit assignment and exploration/exploitation as central difficulties. Verified with bundled lookup.py, PDF pages 25–26 (printed 11–12), section 1.3. Supervised mappings verified PDF pages 15–16 (printed 1–2), section 1.1. [Author textbook site](https://udlbook.github.io/udlbook/)

### Inferences
- Practical refinement order: add corrective data first; if failure remains contact/dynamics-specific, test a bounded residual MLP around a frozen nominal controller. Clip residuals, retain base controller constraints, and compare IL-only versus IL-plus-residual on exactly the same held-out seeds.
- Alternative: deliberately implement a compatible stochastic actor and BC-initialize then PPO fine-tune, or use a diffusion-specific method such as DPPO. Neither is a one-line conversion from ACT. Preserve original checkpoints and include forgetting/regression evaluation.
- Suggested glossary: **policy** = observation-to-action rule; **IL** = learn what an expert does; **BC** = supervised prediction of expert actions; **RL** = improve from rewarded trials; **PPO/SAC** = RL training algorithms; **MLP** = ordinary layered neural network; **ACT** = demonstration-trained transformer that proposes an action chunk; **Diffusion Policy** = generate a coherent chunk through repeated denoising; **hierarchy** = a supervisor chooses which short skill should run; **receding horizon** = execute only the start of a plan, observe again, replan.
- Illustrate why long-horizon reliability needs verification/recovery: if six transfers independently succeed with probability 0.95, all-six success is 0.95^6 ≈0.735. This is arithmetic under an independence simplification, not a performance prediction; correlated failures can differ.

### Gaps
- No evidence establishes an RL algorithm or pretrained VLA checkpoint as best for this exact S1 scene. Recommendation should explicitly remain falsifiable through skill-level and full-mission evaluation.
