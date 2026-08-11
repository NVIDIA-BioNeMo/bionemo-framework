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

import json
from pathlib import Path


REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
ROOT_AGENT_DIR = REPO_ROOT / ".agents"
RECIPE_ROOT = REPO_ROOT / "recipes" / "evo2_phage_gen"
RECIPE_AGENT_DIR = RECIPE_ROOT / ".agents"
EXPECTED_RECIPE_SKILLS = {
    "bionemo-phage-design",
    "bionemo-phage-design-adapt-execution",
    "bionemo-phage-design-calibrate-rl-sampling",
    "bionemo-phage-design-collect-genomes",
    "bionemo-phage-design-generate-and-screen",
    "bionemo-phage-design-implement-rl-objectives",
    "bionemo-phage-design-operate-mbridge-sft",
    "bionemo-phage-design-operate-nemo-rl",
    "bionemo-phage-design-plan-rl-objectives",
    "bionemo-phage-design-prepare-sft",
    "bionemo-phage-design-publish-stage-artifacts",
    "bionemo-phage-design-research-evidence",
}


def _skill_names(agent_dir: Path) -> set[str]:
    return {path.name for path in (agent_dir / "skills").iterdir() if (path / "SKILL.md").is_file()}


def _plugin(agent_dir: Path) -> dict:
    return json.loads((agent_dir / ".codex-plugin" / "plugin.json").read_text())


def test_root_agent_package_contains_only_portable_entry_skill() -> None:
    assert _skill_names(ROOT_AGENT_DIR) == {"bionemo-phage-generation"}
    assert _plugin(ROOT_AGENT_DIR)["name"] == "bionemo-phage-generation"
    assert _plugin(ROOT_AGENT_DIR)["skills"] == "./skills/"


def test_recipe_agent_package_owns_implementation_skills() -> None:
    assert _skill_names(RECIPE_AGENT_DIR) == EXPECTED_RECIPE_SKILLS
    assert _plugin(RECIPE_AGENT_DIR)["name"] == "bionemo-phage-design"
    assert (RECIPE_ROOT / "CLAUDE.md").is_symlink()
    assert (RECIPE_ROOT / "CLAUDE.md").readlink() == Path("AGENTS.md")


def test_portable_skill_requires_complete_checkout_and_absolute_discovery_handoff() -> None:
    portable_skill = (ROOT_AGENT_DIR / "skills" / "bionemo-phage-generation" / "SKILL.md").read_text(encoding="utf-8")
    for marker in (
        "VERSION >= 2.4",
        "bionemo-phage-design/SKILL.md",
        "design-scope-and-viability.md",
        "ema-2025-draft-phage-therapy-quality-guideline.md",
        ".codex-plugin/plugin.json",
        "https://github.com/NVIDIA-BioNeMo/bionemo-recipes",
        "canonical default revision",
        "origin/jstjohn/evo2_phage_gen",
        "separate clean checkout",
        "absolute checkout root",
        "absolute recipe root",
        "original request",
        "Codex",
        "$bionemo-phage-design",
        "--plugin-dir .agents",
        "/evo2-phage-gen:bionemo-phage-design",
        "discoverable",
        "missing skill",
        "plugin's `skills` root",
        "immediate child",
        "Record the discovered names",
        "controller and every discovered sibling",
        "enumeration or loading fails",
        "project-wide RUNLOG",
        "complete whole-genome",
        "EMA-derived therapeutic objectives",
        "auto-enables supported authenticated W&B",
        "dependency-aware, resource-admitted execution",
        "bounded autonomy",
        "durable decision reporting",
    ):
        assert marker in portable_skill

    portable_evals = json.loads(
        (ROOT_AGENT_DIR / "skills" / "bionemo-phage-generation" / "evals" / "evals.json").read_text(encoding="utf-8")
    )
    portable_evals_text = json.dumps(portable_evals)
    for marker in (
        "VERSION == 2.4",
        "no recipe-local controller",
        "aggregation",
        "absolute-root Codex",
        "absolute-root Claude",
        "original request",
        "discovery verification",
        "whole-genome/lifecycle scope",
        "without reward starvation",
    ):
        assert marker in portable_evals_text


def test_recipe_skill_contract_guards_scope_provenance_and_telemetry() -> None:
    skills_root = RECIPE_AGENT_DIR / "skills"
    controller = (skills_root / "bionemo-phage-design" / "SKILL.md").read_text(encoding="utf-8")
    project_contract = (skills_root / "bionemo-phage-design" / "references" / "project-contract.md").read_text(
        encoding="utf-8"
    )
    design_contract = (
        skills_root / "bionemo-phage-design" / "references" / "design-scope-and-viability.md"
    ).read_text(encoding="utf-8")
    ema_guideline = (
        skills_root / "bionemo-phage-design" / "references" / "ema-2025-draft-phage-therapy-quality-guideline.md"
    ).read_text(encoding="utf-8")
    execution_contract = (
        skills_root / "bionemo-phage-design-adapt-execution" / "references" / "execution-contract.md"
    ).read_text(encoding="utf-8")
    objective_skill = (skills_root / "bionemo-phage-design-plan-rl-objectives" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    objective_contract = (
        skills_root / "bionemo-phage-design-plan-rl-objectives" / "references" / "objective-contract.md"
    ).read_text(encoding="utf-8")
    implementation_skill = (skills_root / "bionemo-phage-design-implement-rl-objectives" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    ema_linked_docs = (
        controller,
        (skills_root / "bionemo-phage-design-research-evidence" / "SKILL.md").read_text(encoding="utf-8"),
        objective_skill,
        (skills_root / "bionemo-phage-design-generate-and-screen" / "SKILL.md").read_text(encoding="utf-8"),
        (skills_root / "bionemo-phage-design-research-evidence" / "references" / "search-protocol.md").read_text(
            encoding="utf-8"
        ),
    )

    for marker in (
        "complete whole-genome candidates",
        "provisionally treat adapted work as therapeutic",
        "planning/DESIGN_SPEC.yaml",
        "explicit approval",
        "root `RUNLOG.md`",
        "auto-enable W&B",
        "ema-2025-draft-phage-therapy-quality-guideline.md",
    ):
        assert marker in controller
    for marker in ("root `RUNLOG.md`", "one initialization action", "Stage RUNLOG files do not replace"):
        assert marker in project_contract
    for marker in (
        "mutable_scope: whole-genome",
        "locus-only edit",
        "productive infection rather than adsorption alone",
        "physical epigenetic state",
        "same-taxon pooled",
        "scores remain 1",
        "Apply intended-use therapeutic guardrails",
        "provisionally classify an",
        "do not collapse them into one learned",
        "Product-manufacturing controls",
        "explicitly non-therapeutic project",
        "separate online RL component",
        "current PhiX174 case-study replication",
    ):
        assert marker in design_contract
    for marker in (
        "Intended-use therapeutic objectives",
        "#phage-seed-lots",
        "#genome-characterisation",
        "#host-range",
        "#potency",
        "#transducing-capacity",
        "hard exclusion passable",
    ):
        assert marker in objective_skill
    for marker in (
        "schema_version: 3",
        "therapeutic_quality:",
        "reward_support:",
        "filter 7 disabled",
        "fixed-zero component triggers",
    ):
        assert marker in objective_contract
    for marker in (
        "planning-owned intended-use and safety-applicability block",
        "baseline-SFT generations",
        "Low initial reward support alone",
    ):
        assert marker in implementation_skill
    for document in ema_linked_docs:
        assert "ema-2025-draft-phage-therapy-quality-guideline.md" in document
    assert (
        "[therapeutic suitability and safety-related exclusion criteria]"
        "(references/ema-2025-draft-phage-therapy-quality-guideline.md)" in controller
    )
    assert (
        "[therapeutic suitability and safety-related exclusion criteria]"
        "(../bionemo-phage-design/references/ema-2025-draft-phage-therapy-quality-guideline.md)" in ema_linked_docs[1]
    )
    assert (
        "[therapeutic suitability and safety-related exclusion criteria]"
        "(../bionemo-phage-design/references/ema-2025-draft-phage-therapy-quality-guideline.md)" in objective_skill
    )
    assert (
        "[therapeutic-suitability exclusion rules]"
        "(../bionemo-phage-design/references/ema-2025-draft-phage-therapy-quality-guideline.md)" in ema_linked_docs[3]
    )
    for marker in (
        "EMA/CHMP/BWP/1/2024",
        "e7953ee8e56b55bf147962872e721746e0667815a70428dd88bad428c461db48",
        "The phages used to generate a seed lot should be strictly lytic",
        "lysogeny modules",
        "lysogeny should be demonstrated",
    ):
        assert marker in ema_guideline
    assert "Page 2/15" not in ema_guideline
    for marker in (
        "auto-enable-unless-opted-out",
        "WANDB_API_KEY",
        "api.wandb.ai",
        "A checked-in `wandb_enabled: false` default is",
        "not an opt-out",
        "sandbox-visible GPU failure",
        "host-visible execution context",
        "does not establish a host driver failure",
    ):
        assert marker in execution_contract


def test_controller_uses_dependency_graph_and_bounded_autonomy() -> None:
    skills_root = RECIPE_AGENT_DIR / "skills"
    controller = (skills_root / "bionemo-phage-design" / "SKILL.md").read_text(encoding="utf-8")
    contract = (skills_root / "bionemo-phage-design" / "references" / "project-contract.md").read_text(
        encoding="utf-8"
    )

    for marker in (
        "planning/DEPENDENCY_GRAPH.yaml",
        "dependency-ready and resource-admissible",
        "blocked node blocks only its descendants",
    ):
        assert marker in controller
    for marker in (
        "Every applicable safety, biological-evidence, approval, lineage, and acceptance gate MUST be a satisfied `hard_dependencies` entry",
        "autonomy_envelope",
        "Numeric action IDs preserve traceability",
        "```mermaid",
        "Implement and test RL functions",
        "GPU request: 8",
    ):
        assert marker in contract


def test_execution_adapter_uses_resource_aware_admission() -> None:
    skills_root = RECIPE_AGENT_DIR / "skills"
    adapter = (skills_root / "bionemo-phage-design-adapt-execution" / "SKILL.md").read_text(encoding="utf-8")
    contract = (
        skills_root / "bionemo-phage-design-adapt-execution" / "references" / "execution-contract.md"
    ).read_text(encoding="utf-8")

    assert "resource-aware admission control" in adapter
    for marker in (
        "dependency-ready",
        "resource-admissible",
        "reservations",
        "write-scope conflicts",
        "queued",
    ):
        assert marker in contract


def test_behavioral_evals_cover_scope_runlog_and_wandb_regressions() -> None:
    eval_ids = set()
    eval_text = ""
    for eval_path in sorted((RECIPE_AGENT_DIR / "skills").glob("*/evals/evals.json")):
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        eval_ids.update(case["id"] for case in payload["evals"])
        eval_text += json.dumps(payload)

    assert "bionemo-phage-design-006-whole-genome-host-range-scope" in eval_ids
    assert "bionemo-phage-design-adapt-execution-005-fresh-runtime-and-wandb" in eval_ids
    assert "bionemo-phage-design-007-resource-aware-dag-autonomy" in eval_ids
    assert "bionemo-phage-design-adapt-execution-006-sandbox-gpu-visibility" in eval_ids
    assert "bionemo-phage-design-adapt-execution-007-dag-admission" in eval_ids
    assert "bionemo-phage-design-plan-rl-objectives-006-lifecycle-host-range" in eval_ids
    assert "bionemo-phage-design-plan-rl-objectives-007-replication-therapeutic-boundary" in eval_ids
    assert "bionemo-phage-design-implement-rl-objectives-004-therapeutic-reward-support" in eval_ids
    assert "bionemo-phage-design-operate-nemo-rl-008-wandb-default" in eval_ids
    for marker in (
        "root RUNLOG.md",
        "tail-only",
        "scores above the threshold",
        "WANDB_API_KEY",
        "ema-2025-draft-phage-therapy-quality-guideline.md",
        "sparse or fixed-zero support triggers",
    ):
        assert marker in eval_text


def test_readme_and_historical_evidence_distinguish_rerun_generations() -> None:
    readme = (RECIPE_ROOT / "README.md").read_text(encoding="utf-8")
    historical_evidence = (
        RECIPE_AGENT_DIR / "skills" / "bionemo-phage-design" / "references" / "historical-evidence.md"
    ).read_text(encoding="utf-8")

    for marker in (
        "Prior GDPO run: published Microviridae SFT + step 190",
        "Latest SFT+GDPO run: step 430",
        "Latest step-430 diagnostic",
        "Publication-era screen; not directly comparable",
        "358/1,000 (35.80%)",
        "610/1,000 (61.00%)",
        "22/1,000 (2.20%)",
        "15/110,000 (~0.014%)",
    ):
        assert marker in readme

    for marker in (
        "Later operator-reported SFT+RL rerun",
        "55efb7c2dbe799dfc8b7c67d9517186309c76499",
        "raw run manifests are not present",
        "step `430`",
        "610/1000",
        "22/1000",
    ):
        assert marker in historical_evidence


def test_publication_citation_distinguishes_final_article_from_bundled_preprint() -> None:
    readme = (RECIPE_ROOT / "README.md").read_text(encoding="utf-8")
    controller = (RECIPE_AGENT_DIR / "skills" / "bionemo-phage-design" / "SKILL.md").read_text(encoding="utf-8")

    assert "https://www.science.org/doi/10.1126/science.aec2657" in readme
    assert "https://www.biorxiv.org/content/10.1101/2025.09.12.675911v1.full" in readme
    for marker in (
        "CC BY bioRxiv v1",
        "king-2025-generative-phage-design",
        "final Science publication",
    ):
        assert marker in controller
