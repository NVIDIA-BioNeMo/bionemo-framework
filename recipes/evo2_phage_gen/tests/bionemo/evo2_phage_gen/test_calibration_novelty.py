from __future__ import annotations

import pandas as pd
import pytest

from bionemo.evo2_phage_gen.calibration_novelty import (
    canonical_circular_sequence,
    normalize_prompted_fasta,
    summarize_novelty,
)


def test_canonical_circular_sequence_handles_rotation_and_reverse_complement():
    sequence = "ACGTTT"
    rotated = "TTTACG"
    reverse_complement = "AAACGT"

    assert canonical_circular_sequence(sequence) == canonical_circular_sequence(rotated)
    assert canonical_circular_sequence(sequence) == canonical_circular_sequence(reverse_complement)


def test_normalize_prompted_fasta_strips_control_tokens_and_rejects_non_dna(tmp_path):
    source = tmp_path / "prompted.fna"
    source.write_text(">prompted\n+~ACGT\n>raw\nTGCA\n")
    output = tmp_path / "payload.fna"

    normalize_prompted_fasta(source, output)

    assert output.read_text() == ">prompted\nACGT\n>raw\nTGCA\n"

    invalid = tmp_path / "invalid.fna"
    invalid.write_text(">bad\n+~ACNT\n")
    with pytest.raises(ValueError, match="non-DNA"):
        normalize_prompted_fasta(invalid, tmp_path / "unused.fna")


def test_summarize_novelty_reports_copy_rates():
    metrics = pd.DataFrame(
        {
            "cell": ["prefix0_temp1.0", "prefix0_temp1.0"],
            "exact_target_circular_or_revcomp": [1.0, 0.0],
            "exact_sft_circular_or_revcomp": [1.0, 0.0],
            "target_near_copy_98_9pct": [1.0, 0.0],
            "sft_near_copy_98_9pct": [1.0, 1.0],
            "target_pident": [100.0, 80.0],
            "sft_pident": [100.0, 99.0],
        }
    )

    summary = summarize_novelty(metrics).iloc[0]

    assert summary["exact_target_copy_rate"] == 0.5
    assert summary["sft_near_copy_rate"] == 1.0
