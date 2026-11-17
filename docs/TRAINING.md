# Fine-tuning on the contest data

> **Status.** This is a recipe, not a verified pipeline. The sandbox that produced
> this repository has no GPU, no dataset and no access to `huggingface.co`, so none
> of the commands below were executed here. They are written against the public
> GreenVLA code (`greenvla/GreenVLA`, main branch) and the dataset documentation.
> Validate each step on the GPU host before it costs a submission slot.

## What the data is

| Item | Reference |
| --- | --- |
| Demonstrations | `SberRoboticsCenter/GreenChallengeData` (LeRobot v2.1) |
| Scene assets | `SberRoboticsCenter/GreenChallengeAssets` |
| Baseline weights | `SberRoboticsCenter/GreenChallengeModel` |

The dataset is expected to contain three cameras at 448×448, a 51-dimensional
`observation.state` in the order documented in `docs/ARCHITECTURE.md`, and a
51-dimensional action plus an `episode_progress` scalar. Verify, do not assume:

```bash
huggingface-cli download SberRoboticsCenter/GreenChallengeData --repo-type dataset --local-dir data
python training/validate_dataset.py data
```

The validator prints the feature schema, camera names, dimensions, fps and whether
a `subtask` field exists. That last answer decides the language strategy.

## Language: subtask or full instruction

The packaged checkpoint was trained with one instruction per subtask
(`S0S1_PROMPT_FROM_SUBTASK=1`), and the simulator supplies that atomic instruction
as `obs["subtask"]`. There are two consistent choices, and mixing them is worse than
either:

1. **Train with subtasks** (matches the baseline). Needs per-frame or per-segment
   subtask labels. If the dataset has a `subtask` column, use it. If not, derive it:
   segment each episode at the goal-completion timestamps if present, otherwise
   annotate the segment boundaries offline (a VLM or a human, once) and store a
   `subtask` column in the dataset. Set `prompt_from_subtask: true` in the data
   config and keep `GC_PROMPT_FROM_SUBTASK=1` at inference.
2. **Train with the full task instruction.** Set `prompt_from_task: true`,
   `prompt_from_subtask: false`, and at inference set `GC_PROMPT_FROM_SUBTASK=0`.
   This is simpler but asks the model to infer progress from a sentence that is
   identical for every subtask; it gives up the strongest signal the benchmark
   provides.

The baseline's 4–5 subtasks per scenario make option 1 worth the annotation effort.

## Data config

The checkpoint is trained under exactly one data contract, and the contract carries
three things that must match at inference: the embodiment name in the prompt, the
`asset_id` that selects `norm_stats/<asset_id>/norm_stats.json`, and the action
sampling step. The public repository ships `lerobot/conf/scripted_014814.yaml` for
the released checkpoint; copy it and change only what the new dataset requires:

```yaml
# lerobot/conf/green_finetuned.yaml
_target_: lerobot.common.datasets.dtwin_stationary_runtime.DtwinStationaryRuntimeConfig
name: green_finetuned_s0s1_step1_state_action_subtasks
asset_id: green_finetuned
repo_id: <your dataset repo id or local name>
root_dir: /data/greenchallenge
action_sequence_keys: [action]
action_horizon: 50
action_offset: 0
action_sample_step: 1
control_mode: joint
smooth_actions: true
has_velocities: true
map_to_closed_kinematic: true
input_kinematic_space: closed
model_kinematic_space: closed
return_subtasks: true
return_subtasks_mode: strict
prompt_from_subtask: true
exclude_negative_subtasks: false
base_config:
  _target_: lerobot.common.datasets.data_config.DataConfig
  local_files_only: true
  prompt_from_task: false
```

The stationary runtime masks input base velocities and zeroes predicted base twist
(`26:32` in model space). That matches the benchmark (`ARM_FF=0`, a standing robot);
do not copy a locomotion contract for these tasks.

## Normalisation statistics

The checkpoint reads `norm_stats/<asset_id>/norm_stats.json`. Compute it from the
fine-tuning dataset with the repository's statistics script, and keep the file next
to the checkpoint, not in the dataset:

```bash
python lerobot/scripts/compute_dataset_stats.py \
  --config-path=../conf --config-name=green_finetuned \
  --output-dir /ckpt/norm_stats/green_finetuned
```

The exact flag names are the script's; check `--help` on the checkout you use. A
mismatch between the `asset_id` and the directory name loads and returns plausible
but wrong actions, so this is a step to verify by file existence, not by hope.

## Training

The public repository's fine-tuning entrypoint is
`lerobot/scripts/train_with_validation.py`, configured through Hydra. Start from the
shipped fine-tuning config, point it at `green_finetuned`, and keep the baseline's
architecture (Qwen3.5-0.8B frozen backbone, 64 cognition tokens, flow matching) so
the resulting checkpoint loads in the contest server unchanged:

```bash
accelerate launch --multi_gpu --num_processes=8 \
  lerobot/scripts/train_with_validation.py \
  --config-path=../conf --config-name=finetune_greenvla_bridge \
  dataset=green_finetuned \
  policy.path=SberRoboticsCenter/GreenChallengeModel \
  training.batch_size=16 training.num_workers=8
```

Check the Hydra overrides against the checkout's `conf/finetune_greenvla_bridge.yaml`;
the names above are the structure, not a promise. Two settings are not free
choices: `action_horizon=50` and `action_sample_step=1`. Changing either changes the
timing the feed assumes, and the feed's 0.080 s period was tuned for 50 rows at
30 fps.

## Robustness to the out-of-distribution evaluation

The benchmark's scenes, lighting, object placements and textures are not the
demonstration distribution, and the whole-body controller is not deterministic. Two
practical measures:

* **Photometric augmentation** (brightness, contrast, saturation, slight blur, mild
  colour jitter) applied per camera at train time. Do not rotate: the policy already
  rotates the head 90° clockwise and the wrists 180°. Do not crop aggressively; the
  grasp point is often at the edge of the wrist image.
* **Closed-loop replanning is the real robustness mechanism.** With
  `PROPRIO_CMD=all` the model sees its own last command, not the measured joints, so
  it cannot learn to correct from proprioception alone. The camera images are the
  only channel that shows what actually happened. Do not train the model to trust the
  echoed state; train it to use vision, and keep the request rate at 80 ms so the
  loop closes often.

An ensemble of two checkpoints selected by the current instruction or by a
vision-derived scene descriptor is explicitly allowed. It is also the only allowed
way to specialise per task; switching by a timer, episode index or scenario name is
not.

## Packaging a fine-tuned checkpoint

Checkpoint layout inside the image:

```text
/ckpt/
  pretrained_model/config.json
  pretrained_model/model.safetensors
  norm_stats/green_finetuned/norm_stats.json
```

Use `docker/Dockerfile.finetuned`, which copies `model/checkpoint/` over `/ckpt` and
`model/conf/*.yaml` into `/app/lerobot/conf/`, then set
`S0S1_DATA_CONFIG=green_finetuned` on the container. Keep `model/` out of Git; the
weights are large and the repository is a workspace, not a model store.

## Evaluation

Every candidate goes through the same protocol as the baseline
(`docs/EVALUATION.md`): alternating A/B runs, at least three repetitions, completion
before score. A fine-tune that raises completion but lowers score is still valuable;
one that raises score on a single run is not evidence.
