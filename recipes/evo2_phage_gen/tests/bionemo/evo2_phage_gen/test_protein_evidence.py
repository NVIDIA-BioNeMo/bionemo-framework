# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Focused tests for smooth, ORF-gated reference evidence."""

import pandas as pd
import pytest
from Bio import SeqIO

from bionemo.evo2_phage_gen import protein_evidence


SMOOTH_SYNTENY_MATCH = {
    "identity_full_credit": 0.90,
    "reference_coverage_full_credit": 0.95,
    "candidate_coverage_full_credit": 0.95,
    "gamma": 1.5,
    "raw_integrity_min": 0.001,
    "min_credit": 0.01,
}
SMOOTH_TROPISM_MATCH = {
    **SMOOTH_SYNTENY_MATCH,
    "identity_full_credit": 0.95,
    "reference_coverage_full_credit": 0.99,
    "candidate_coverage_full_credit": 0.99,
}


def test_pseudocircular_filter_removes_a_prefix_tail_repeated_by_a_cross_origin_orf(tmp_path):
    """A native-prefix tail must not be counted again when a longer circular ORF contains it."""
    source = tmp_path / "source.fasta"
    source.write_text(">genome\nATGTAAATGAAA\n")
    nucleotide_orfs = tmp_path / "orfs.fasta"
    protein_orfs = tmp_path / "proteins.fasta"
    nucleotide_orfs.write_text(
        ">genome_ORF.1 [0-6](+) type:complete length:6\nATGAAA\n"
        ">genome_ORF.2 [6-18](+) type:complete length:12\nATGAAAATGAAA\n"
        ">genome_ORF.3 [6-12](+) type:complete length:6\nATGAAA\n"
    )
    protein_orfs.write_text(
        ">genome_ORF.1 [0-6](+) type:complete length:6\nMK\n"
        ">genome_ORF.2 [6-18](+) type:complete length:12\nQQMK\n"
        ">genome_ORF.3 [6-12](+) type:complete length:6\nQQ\n"
    )

    protein_evidence.remove_pseudocircular_extension_orfs(source, nucleotide_orfs, protein_orfs)

    assert [record.id for record in SeqIO.parse(protein_orfs, "fasta")] == ["genome_ORF.2", "genome_ORF.3"]


def test_reference_gff_proteins_and_candidate_orf_order_are_loaded_for_smooth_search(tmp_path):
    """The search inputs must preserve reference loci and coordinate-order called ORFs."""
    reference_gff = tmp_path / "reference.gff"
    reference_gff.write_text(
        "##gff-version 3\n"
        "ref\ttest\tCDS\t1\t9\t.\t+\t0\tID=A\n"
        "ref\ttest\tCDS\t10\t18\t.\t+\t0\tID=G\n"
        "##FASTA\n"
        ">ref\n"
        "ATGAAATAAATGCCCTAA\n"
    )
    reference_fasta = tmp_path / "reference-proteins.fasta"
    candidate_orfs = tmp_path / "candidate-orfs.fasta"
    candidate_orfs.write_text(
        ">umi1_ORF.5 [300-450](+) type:complete length:150\nATG\n"
        ">umi1_ORF.2 [10-100](+) type:complete length:90\nATG\n"
    )

    reference_order = protein_evidence.write_reference_protein_fasta(reference_gff, reference_fasta)
    candidate_sequences, candidate_orders = protein_evidence.load_candidate_orf_context(candidate_orfs)

    assert reference_order == ("A", "G")
    assert [(record.id, str(record.seq)) for record in SeqIO.parse(reference_fasta, "fasta")] == [
        ("A", "MK"),
        ("G", "MP"),
    ]
    assert candidate_orders == {"umi1": ("umi1_ORF.2", "umi1_ORF.5")}
    assert set(candidate_sequences) == {"umi1_ORF.2", "umi1_ORF.5"}


def test_smooth_reference_summary_reuses_orf_hits_for_synteny_tropism_and_gene_a_origin():
    """One permissive ORF search must drive all three graded reference objectives."""
    motif = "CAACTTGATATTAATAACACTATAGACCAC"
    a_orf = "G" * 6 + motif + "G" * 30
    hits = pd.DataFrame(
        [
            {
                "query": "A",
                "target": "umi1_ORF.1",
                "evalue": 1e-20,
                "pident": 90.0,
                "alnlen": 95,
                "qlen": 100,
                "tlen": 100,
            },
            {
                "query": "G",
                "target": "umi1_ORF.2",
                "evalue": 1e-20,
                "pident": 95.0,
                "alnlen": 99,
                "qlen": 100,
                "tlen": 100,
            },
        ]
    )

    observed = protein_evidence.summarize_smooth_reference_evidence(
        hits,
        genome_sequences={"umi1": a_orf, "umi2": "G" * len(a_orf)},
        candidate_orf_sequences={"umi1_ORF.1": a_orf, "umi1_ORF.2": "ATG" * 34},
        candidate_orders={"umi1": ("umi1_ORF.1", "umi1_ORF.2"), "umi2": ()},
        reference_order=("A", "G"),
        synteny_match_parameters=SMOOTH_SYNTENY_MATCH,
        tropism_match_parameters=SMOOTH_TROPISM_MATCH,
        synteny_order_weight=0.75,
        synteny_duplicate_penalty_weight=0.75,
        gene_a_reference_locus="A",
        tropism_reference_locus="G",
        gene_a_origin_motif=motif,
        gene_a_origin_offset_nt=6,
        gene_a_origin_offset_tolerance_nt=6,
    ).set_index("id_prompt")

    assert observed.loc["umi1", "reward_external_synteny"] == 1.0
    assert observed.loc["umi1", "reward_external_tropism"] == 1.0
    assert observed.loc["umi1", "reward_gene_a_origin"] == 1.0
    assert observed.loc["umi1", "smooth_reference_matched_loci"] == 2
    assert observed.loc["umi2", "reward_external_synteny"] == 0.0
    assert observed.loc["umi2", "reward_external_tropism"] == 0.0
    assert observed.loc["umi2", "reward_gene_a_origin"] == 0.0


def test_smooth_reference_summary_rejects_invalid_match_settings_without_hits():
    """A no-hit batch must not turn an invalid reward configuration into measured zero."""
    with pytest.raises(ValueError, match="smooth protein-match configuration"):
        protein_evidence.summarize_smooth_reference_evidence(
            pd.DataFrame(columns=["query", "target", "evalue", "pident", "alnlen", "qlen", "tlen"]),
            genome_sequences={"umi1": "A" * 100},
            candidate_orf_sequences={},
            candidate_orders={"umi1": ()},
            reference_order=("A",),
            synteny_match_parameters={**SMOOTH_SYNTENY_MATCH, "identity_full_credit": 0.0},
            tropism_match_parameters=SMOOTH_TROPISM_MATCH,
            synteny_order_weight=0.75,
            synteny_duplicate_penalty_weight=0.75,
            gene_a_reference_locus="A",
            tropism_reference_locus="G",
            gene_a_origin_motif="CAACTTGATATTAATAACACTATAGACCAC",
            gene_a_origin_offset_nt=6,
            gene_a_origin_offset_tolerance_nt=6,
        )


def test_smooth_match_rejects_decoy_scale_evidence_and_grades_real_partial_matches():
    """A shuffled-scale edge must stay zero while a credible fragment starts above zero."""
    kwargs = {
        "identity_full_credit": 0.85,
        "reference_coverage_full_credit": 0.95,
        "candidate_coverage_full_credit": 0.95,
        "gamma": 1.5,
        "raw_integrity_min": 0.001,
        "min_credit": 0.01,
    }

    decoy = protein_evidence.smooth_protein_match_integrity(
        percent_identity=21.8,
        e_value=0.46,
        alignment_length=147,
        reference_length=522,
        candidate_length=427,
        **kwargs,
    )
    partial = protein_evidence.smooth_protein_match_integrity(
        percent_identity=35.0,
        e_value=1e-8,
        alignment_length=70,
        reference_length=100,
        candidate_length=100,
        **kwargs,
    )
    complete = protein_evidence.smooth_protein_match_integrity(
        percent_identity=85.0,
        e_value=1e-20,
        alignment_length=95,
        reference_length=100,
        candidate_length=100,
        **kwargs,
    )

    assert decoy == 0.0
    assert 0.01 < partial < 1.0
    assert complete == 1.0


def test_smooth_match_penalizes_both_truncations_and_fusions():
    """Dropping either reciprocal-coverage side must lower an otherwise identical match."""
    kwargs = {
        "percent_identity": 85.0,
        "e_value": 1e-20,
        "identity_full_credit": 0.85,
        "reference_coverage_full_credit": 0.95,
        "candidate_coverage_full_credit": 0.95,
        "gamma": 1.5,
        "raw_integrity_min": 0.001,
        "min_credit": 0.01,
    }
    complete = protein_evidence.smooth_protein_match_integrity(
        alignment_length=95,
        reference_length=100,
        candidate_length=100,
        **kwargs,
    )
    truncation = protein_evidence.smooth_protein_match_integrity(
        alignment_length=70,
        reference_length=100,
        candidate_length=70,
        **kwargs,
    )
    fusion = protein_evidence.smooth_protein_match_integrity(
        alignment_length=95,
        reference_length=100,
        candidate_length=140,
        **kwargs,
    )

    assert complete == 1.0
    assert 0.0 < truncation < complete
    assert 0.0 < fusion < complete


def test_ordered_partial_matches_outscore_the_same_scrambled_matches():
    """Synteny must add signal beyond reference content for identical edge weights."""
    reference_order = ("A", "B", "C", "D")
    ordered = protein_evidence.score_smooth_reference_architecture(
        {("A", "a"): 0.2, ("B", "b"): 0.2, ("C", "c"): 0.2, ("D", "d"): 0.2},
        reference_order=reference_order,
        candidate_order=("a", "b", "c", "d"),
        order_weight=0.75,
        duplicate_penalty_weight=0.75,
    )
    scrambled = protein_evidence.score_smooth_reference_architecture(
        {("A", "a"): 0.2, ("B", "b"): 0.2, ("C", "c"): 0.2, ("D", "d"): 0.2},
        reference_order=reference_order,
        candidate_order=("a", "c", "b", "d"),
        order_weight=0.75,
        duplicate_penalty_weight=0.75,
    )

    assert ordered.content_score == pytest.approx(scrambled.content_score)
    assert ordered.reward == pytest.approx(0.2)
    assert 0.0 < scrambled.reward < ordered.reward


def test_smooth_architecture_does_not_reward_deletion_or_order_repair_by_duplication():
    """A candidate cannot raise synteny by deleting evidence or adding a second homolog."""
    reference_order = ("A", "B", "C", "D")
    swapped = protein_evidence.score_smooth_reference_architecture(
        {("A", "a"): 1.0, ("B", "b"): 1.0, ("C", "c"): 1.0, ("D", "d"): 1.0},
        reference_order=reference_order,
        candidate_order=("a", "c", "b", "d"),
        order_weight=0.75,
        duplicate_penalty_weight=0.75,
    )
    deleted = protein_evidence.score_smooth_reference_architecture(
        {("A", "a"): 1.0, ("B", "b"): 1.0, ("D", "d"): 1.0},
        reference_order=reference_order,
        candidate_order=("a", "b", "d"),
        order_weight=0.75,
        duplicate_penalty_weight=0.75,
    )
    duplicated = protein_evidence.score_smooth_reference_architecture(
        {
            ("A", "a"): 1.0,
            ("B", "b"): 1.0,
            ("B", "b_ordered"): 1.0,
            ("C", "c"): 1.0,
            ("D", "d"): 1.0,
        },
        reference_order=reference_order,
        candidate_order=("a", "b_ordered", "c", "b", "d"),
        order_weight=0.75,
        duplicate_penalty_weight=0.75,
    )

    assert deleted.reward < swapped.reward
    assert duplicated.reward == pytest.approx(swapped.reward)
    assert duplicated.duplicate_score == pytest.approx(0.25)


def test_smooth_architecture_is_rotation_invariant_and_one_to_one():
    """Circular rotation is neutral and one ORF cannot satisfy two reference loci."""
    rotated = protein_evidence.score_smooth_reference_architecture(
        {("A", "a"): 1.0, ("B", "b"): 1.0, ("C", "c"): 1.0, ("D", "d"): 1.0},
        reference_order=("A", "B", "C", "D"),
        candidate_order=("d", "a", "b", "c"),
        order_weight=0.75,
        duplicate_penalty_weight=0.75,
    )
    ambiguous = protein_evidence.score_smooth_reference_architecture(
        {("A", "shared"): 1.0, ("B", "shared"): 1.0},
        reference_order=("A", "B"),
        candidate_order=("shared",),
        order_weight=0.75,
        duplicate_penalty_weight=0.75,
    )

    assert rotated.reward == 1.0
    assert ambiguous.content_integrity_sum == 1.0


def test_gene_a_origin_requires_the_functional_site_inside_the_assigned_a_orf():
    """An exact site outside its A-ORF context must not earn origin credit."""
    motif = "CAACTTGATATTAATAACACTATAGACCAC"
    expected_offset = 345
    exact_a = "G" * expected_offset + motif + "G" * 200
    misplaced_a = "G" * 100 + motif + "G" * (len(exact_a) - 130)

    exact = protein_evidence.score_gene_a_origin(
        candidate_a_orf_nt=exact_a,
        candidate_genome_nt=exact_a,
        a_match_integrity=1.0,
        motif=motif,
        expected_offset_nt=expected_offset,
        offset_tolerance_nt=30,
    )
    misplaced = protein_evidence.score_gene_a_origin(
        candidate_a_orf_nt=misplaced_a,
        candidate_genome_nt=misplaced_a,
        a_match_integrity=1.0,
        motif=motif,
        expected_offset_nt=expected_offset,
        offset_tolerance_nt=30,
    )

    assert exact.reward == 1.0
    assert exact.exact_functional_site is True
    assert misplaced.reward == 0.0


def test_gene_a_origin_weights_the_nicking_core_and_ignores_nonfunctional_tail_bases():
    """Known nicking-core mutations matter more than bases 29-30, which are dispensable in vitro."""
    motif = "CAACTTGATATTAATAACACTATAGACCAC"
    offset = 345

    def score(site: str):
        candidate = "G" * offset + site + "G" * 200
        return protein_evidence.score_gene_a_origin(
            candidate_a_orf_nt=candidate,
            candidate_genome_nt=candidate,
            a_match_integrity=1.0,
            motif=motif,
            expected_offset_nt=offset,
            offset_tolerance_nt=30,
        )

    critical = score(motif[:3] + "A" + motif[4:])
    binding = score(motif[:14] + "C" + motif[15:])
    tail = score(motif[:28] + "TT")

    assert 0.0 < critical.reward < binding.reward < 1.0
    assert tail.reward == 1.0
    assert tail.exact_functional_site is True


def test_gene_a_origin_penalizes_duplicate_strong_sites():
    """Adding a second strong origin must not improve a candidate's reward."""
    motif = "CAACTTGATATTAATAACACTATAGACCAC"
    offset = 345
    candidate_a = "G" * offset + motif + "G" * 200
    near_exact_functional_site = motif[:14] + "C" + motif[15:]
    duplicated_genome = candidate_a + "G" * 50 + near_exact_functional_site

    result = protein_evidence.score_gene_a_origin(
        candidate_a_orf_nt=candidate_a,
        candidate_genome_nt=duplicated_genome,
        a_match_integrity=1.0,
        motif=motif,
        expected_offset_nt=offset,
        offset_tolerance_nt=30,
    )

    assert result.strong_site_count == 2
    assert result.reward == pytest.approx(0.5)
