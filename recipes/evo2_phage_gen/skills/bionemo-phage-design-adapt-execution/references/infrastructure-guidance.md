# Infrastructure guidance

Use the execution path actually available to the user: local GPU, SSH, Slurm, Lepton, or a manual handoff. Inspect local policy and current command help, and adapt existing scripts when possible.

A long-running worker must survive the chat or agent process. Launch it through a durable facility, retain a stable job identifier, and make status and logs queryable after reconnecting. Observe startup and meaningful progress, then continue monitoring until a terminal success or failure is known. Choose check intervals from observed progress timing and the next useful decision point rather than continuously polling a stable job. A submitted or backgrounded command is not completion.

Record resolved settings, paths, job identifiers, logs, checkpoints, and resume actions in the project runlog. Local logs are sufficient; remote telemetry is optional. Do not expose credentials.
