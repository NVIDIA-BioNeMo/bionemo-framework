---
name: bionemo-phage-design-adapt-execution
description: Use when a phage-design workflow must discover or adapt to local GPU, SSH, Slurm, Lepton, manual, or unfamiliar execution infrastructure and produce durable launch, monitoring, resume, or handoff commands.
metadata:
  author: NVIDIA <bionemofeedback@nvidia.com>
---

# Adapt Phage Execution

Work inside the recipe and result roots selected by the controller.

Discover enough of the current environment to run the approved scientific stages: repository commands and help, installed environment, storage, CPUs, GPU count/model/memory/occupancy, scheduler or cloud facilities, network constraints, and existing jobs. Follow the shared [command and asset guidance](../bionemo-phage-design/references/command-and-asset-guidance.md). Avoid printing credentials.

An agent sandbox or container can hide host GPUs. A failed `nvidia-smi`, missing `/dev/nvidia*`, or false CUDA-availability result inside that boundary does not prove the host lacks GPUs or drivers. With permission, repeat the probe in a host-visible context before changing drivers, switching to CPU, or reducing the intended topology. If that is unavailable, say what remains unknown and provide the host-side probe.

Choose among local, SSH, Slurm, Lepton, or manual execution from what is actually available. Follow the concise [infrastructure guidance](references/infrastructure-guidance.md), local site policy, and existing repository or user-provided scripts; do not assume a particular mechanism. Build the recipe with `./.ci_build.sh` before sourcing `.ci_test_env.sh`.

Use the [compute guidance](references/compute-guidance.md) and a bounded full-shape smoke test to size training, RL, generation, and computational filters. Preserve whole-genome context and the intended effective batch. Respond to memory pressure from measurements rather than silently changing the scientific task.

Use an available durable job facility for work that may outlive the session. The worker and its status/log queries must survive the chat or agent process; shell backgrounding inside the current session is not enough. Make the run reattachable by a stable job identifier, observe startup and later progress, and verify terminal success or failure before handing results downstream. Durability is not an unconditional relaunch through the configured step ceiling: a segmented SFT or RL supervisor must persist and honor the stage operator's scientific continuation decision before starting its next segment. Record the command, resolved settings, paths, job identifier, status/log location, checkpoints, decision state, and any resume action in `RUNLOG.md`. W&B or other remote telemetry is optional, and local logs remain sufficient.
