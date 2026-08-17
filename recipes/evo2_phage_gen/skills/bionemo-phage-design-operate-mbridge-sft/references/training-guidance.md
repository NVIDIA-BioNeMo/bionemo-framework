# SFT training guidance

Resolve the current training command and record the base model/version, explicit train/validation/test inputs, tokenizer and serialization, context length, precision, optimizer, effective batch, random seed, validation cadence, and checkpoint cadence.

Use a bounded real-data smoke to confirm finite loss, parameter updates, checkpoint writing, and resume. During the full run, monitor comparable training and validation loss, throughput, memory, and failures through a facility that survives the chat process.

Train to the evidence-based ceiling unless the validation curve supports an earlier stop. Select the lowest credible validation-loss checkpoint with uncertainty and curve stability in mind, then evaluate that selected checkpoint once on the held-out test set. Resume only when model, data, optimizer/scheduler, and serialization semantics remain compatible.
