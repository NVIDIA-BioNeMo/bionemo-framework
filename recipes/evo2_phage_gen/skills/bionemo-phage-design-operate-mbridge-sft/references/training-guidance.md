# SFT training guidance

Resolve the current training command and record the base model/version, explicit train/validation/test inputs, tokenizer and serialization, context length, precision, optimizer, effective batch, random seed, validation cadence, and checkpoint cadence.

Use a bounded real-data smoke to confirm finite loss, parameter updates, checkpoint writing, and resume. During the full run, monitor comparable training and validation loss, throughput, memory, and failures through a facility that survives the chat process.

Set the training ceiling and validation cadence in optimizer updates and examples or tokens seen, not epoch count alone. A larger global batch means fewer updates per epoch; if the latest comparable validation is still best, continue—often across several epochs—until enough points show a plateau or rebound. Select the lowest credible validation-loss checkpoint with uncertainty and curve stability in mind, then evaluate that selected checkpoint once on the held-out test set. Resume only when model, data, optimizer/scheduler, and serialization semantics remain compatible.
