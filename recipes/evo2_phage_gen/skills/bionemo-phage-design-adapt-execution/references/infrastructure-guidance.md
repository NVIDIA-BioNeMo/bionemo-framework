# Infrastructure guidance

Use the execution path actually available to the user: local GPU, SSH, Slurm, Lepton, or a manual handoff. Inspect local policy and current command help, and adapt existing scripts when possible.

On a GPU workstation where the agent starts outside Docker, use a GPU-enabled container instead of assuming the host can build the recipe directly. The build scripts are tested against the NVIDIA PyTorch container environment. Build from the recipe `Dockerfile` or use the repository `.devcontainer/` as a convenient base, expose the workstation GPUs through the NVIDIA container runtime, and bind-mount the checkout and selected data, cache, and result locations. Build the code inside the mounted checkout with `./.ci_build.sh`, then source `.ci_test_env.sh` for subsequent commands so the running code and editable checkout remain aligned.

A long-running worker must survive the chat or agent process. Launch it through a durable facility, retain a stable job identifier, and make status and logs queryable after reconnecting. Observe startup and meaningful progress, then continue monitoring until a terminal success or failure is known. Choose check intervals from observed progress timing and the next useful decision point rather than continuously polling a stable job. A submitted or backgrounded command is not completion.

Record resolved settings, paths, job identifiers, logs, checkpoints, and resume actions in the project runlog. Local logs are sufficient; remote telemetry is optional. Do not expose credentials.
