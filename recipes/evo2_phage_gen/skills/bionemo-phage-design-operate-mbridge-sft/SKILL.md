---
name: bionemo-phage-design-operate-mbridge-sft
description: Use when launching, monitoring, stopping, resuming, or relaunching Evo 2 phage SFT with Megatron Bridge, or when selecting its best validation-loss checkpoint across local, SSH, scheduler, or cloud execution.
---

# Operate Megatron Bridge Phage SFT

Work inside the recipe and result roots selected by the controller. Use the execution skill to resolve current commands and infrastructure.

Start from the approved base or reused checkpoint and the explicit leakage-controlled split. Follow the concise [training guidance](references/training-guidance.md). Size the run with a bounded full-context smoke test rather than reducing the scientific sequence length.

The smoke test should use real train and validation records and show finite loss, parameter updates, checkpoint writing, and restartability. Fix data, masking, or runtime problems before the full run.

Train to the evidence-based ceiling from SFT preparation. Do not choose a token run merely because it is cheaper or stop at a publication step count without looking at the planned exposure and validation curve. Monitor training/validation loss, throughput, memory, and failures. Resume a compatible interrupted run from its latest usable checkpoint; start a new attempt when data or model semantics change.

Select the checkpoint by validation loss and curve stability, not training loss or the final step. Use the held-out test set once to characterize the selected checkpoint, never to select it. Record the command, resolved settings, environment, data inputs, checkpoints, validation curve, selected step and rationale, test result, and interruptions in `RUNLOG.md` and a concise stage summary.
