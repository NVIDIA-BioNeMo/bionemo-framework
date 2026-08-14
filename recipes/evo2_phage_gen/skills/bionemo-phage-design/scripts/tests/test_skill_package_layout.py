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
import re
from pathlib import Path


def _find_repo_root(path: Path) -> Path:
    for parent in path.resolve().parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError(f"could not locate repository root containing .git above {path}")


REPO_ROOT = _find_repo_root(Path(__file__))
RECIPE_ROOT = REPO_ROOT / "recipes" / "evo2_phage_gen"
RECIPE_AGENT_DIR = RECIPE_ROOT / ".agents"
RECIPE_SKILLS_DIR = RECIPE_ROOT / "skills"
SKILL_ROOT = RECIPE_SKILLS_DIR
REPOSITORY_ROOT = REPO_ROOT
SCRIPT = SKILL_ROOT / "bionemo-phage-design" / "scripts" / "run_skill_evals.py"
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
RETIRED_LEAF_SKILL_NAMES = {
    "adapt-phage-execution",
    "collect-phage-genomes",
    "design-phage-rl-objectives",
    "generate-and-screen-phages",
    "implement-phage-rl-objectives",
    "operate-mbridge-phage-sft",
    "operate-nemo-rl-phage",
    "prepare-phage-sft",
    "research-phage-evidence",
}


def _skill_names(skills_dir: Path) -> set[str]:
    return {path.name for path in skills_dir.iterdir() if (path / "SKILL.md").is_file()}


def _plugin(plugin_root: Path, agent: str) -> dict:
    return json.loads((plugin_root / f".{agent}-plugin" / "plugin.json").read_text())


def _assert_agent_skills_alias(agent_dir: Path, canonical_skills: Path) -> None:
    alias = agent_dir / "skills"
    assert alias.is_symlink()
    assert alias.readlink() == Path("../skills")
    assert alias.resolve() == canonical_skills.resolve()


def _reachable_markdown_files(starts: list[Path], *, root: Path) -> set[Path]:
    linked = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    reachable = {path.resolve() for path in starts}
    pending = list(reachable)
    while pending:
        source = pending.pop()
        for raw_target in linked.findall(source.read_text(encoding="utf-8")):
            target_text = raw_target.split("#", maxsplit=1)[0]
            if not target_text or "://" in target_text or target_text.startswith("mailto:"):
                continue
            target = (source.parent / target_text).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                continue
            if target.is_file() and target.suffix == ".md" and target not in reachable:
                reachable.add(target)
                pending.append(target)
    return reachable


def test_recipe_plugin_uses_catalog_skill_root_and_agent_compatibility_alias() -> None:
    assert _skill_names(RECIPE_SKILLS_DIR) == EXPECTED_RECIPE_SKILLS
    _assert_agent_skills_alias(RECIPE_AGENT_DIR, RECIPE_SKILLS_DIR)
    assert _plugin(RECIPE_ROOT, "codex")["name"] == "bionemo-phage-design"
    assert _plugin(RECIPE_ROOT, "codex")["skills"] == "./skills/"
    assert _plugin(RECIPE_ROOT, "claude")["skills"] == ["./skills/"]
    assert (RECIPE_ROOT / "CLAUDE.md").is_symlink()
    assert (RECIPE_ROOT / "CLAUDE.md").readlink() == Path("AGENTS.md")


def test_every_bundled_markdown_reference_is_reachable_from_a_skill() -> None:
    skills_root = RECIPE_SKILLS_DIR
    starts = list(skills_root.glob("*/SKILL.md"))
    reachable = _reachable_markdown_files(starts, root=skills_root.resolve())
    references = {path.resolve() for path in skills_root.glob("*/references/*.md")}
    assert references <= reachable, sorted(str(path.relative_to(skills_root)) for path in references - reachable)


def test_recipe_skill_contract_guards_scope_provenance_and_telemetry() -> None:
    skills_root = RECIPE_SKILLS_DIR
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
    reward_runtime_contract = (
        skills_root / "bionemo-phage-design-implement-rl-objectives" / "references" / "runtime-contract.md"
    ).read_text(encoding="utf-8")
    resource_policy = (
        skills_root / "bionemo-phage-design-adapt-execution" / "references" / "resource-and-oom-policy.md"
    ).read_text(encoding="utf-8")
    generation_skill = (skills_root / "bionemo-phage-design-generate-and-screen" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    collection_skill = (skills_root / "bionemo-phage-design-collect-genomes" / "SKILL.md").read_text(encoding="utf-8")
    ema_linked_docs = (
        controller,
        (skills_root / "bionemo-phage-design-research-evidence" / "SKILL.md").read_text(encoding="utf-8"),
        objective_skill,
        generation_skill,
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
        "source-neutral",
        "natural-origin attestation",
        "model-generated candidates",
    ):
        assert marker in design_contract
    for marker in ("source-neutral", "natural-origin attestation", "metagenomic", "generated sequence"):
        assert marker in collection_skill
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
        "schema_version: 4",
        "sft_lineage:",
        "checkpoint_stage_binding_evidence:",
        "source_state_sha256:",
        "selection:",
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
    for marker in ("external-tool filtering and scoring", "required operation and database layout", "CPU/GPU outputs"):
        assert marker in generation_skill
    for marker in ("external-tool filtering and scoring", "deterministic row mapping", "accelerated implementation"):
        assert marker in reward_runtime_contract
    for marker in (
        "record workers and per-tool threads",
        "full nested task tree",
        "Do not generalize one tool's result",
        "--batch-size 1",
        "--record-workers",
        "--phrogs-workers",
        "max(batch_workers * threads, batch_workers * phrogs_workers * phrogs_threads)",
        "each shared command",
    ):
        assert marker in resource_policy
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
    skills_root = RECIPE_SKILLS_DIR
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

    for marker in (
        "`schema_version`",
        "`plan_sha256`",
        "`environment_path`",
        "`status`",
        "`approved_at`",
        "`resource_pools`",
        "`capacity_source`",
    ):
        assert marker in contract


def test_controller_does_not_require_a_saved_plan_document() -> None:
    """The durable graph and decisions replace a generated planning/PLAN.md artifact."""
    skill_dir = RECIPE_SKILLS_DIR / "bionemo-phage-design"
    controller = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    contract = (skill_dir / "references" / "project-contract.md").read_text(encoding="utf-8")
    evals = (skill_dir / "evals" / "evals.json").read_text(encoding="utf-8")

    for text in (controller, contract, evals):
        assert "planning/PLAN.md" not in text


def test_execution_adapter_uses_resource_aware_admission() -> None:
    skills_root = RECIPE_SKILLS_DIR
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
    for eval_path in sorted(RECIPE_SKILLS_DIR.glob("*/evals/evals.json")):
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
    assert "bionemo-phage-design-operate-nemo-rl-009-secret-free-command-records" in eval_ids
    assert "bionemo-phage-design-plan-rl-objectives-010-complete-sft-lineage" in eval_ids
    assert "bionemo-phage-design-generate-and-screen-006-amrfinder-nucleotide-evidence" in eval_ids
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
        RECIPE_SKILLS_DIR / "bionemo-phage-design" / "references" / "historical-evidence.md"
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
    controller = (RECIPE_SKILLS_DIR / "bionemo-phage-design" / "SKILL.md").read_text(encoding="utf-8")
    calibration = (RECIPE_SKILLS_DIR / "bionemo-phage-design-calibrate-rl-sampling" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    research = (RECIPE_SKILLS_DIR / "bionemo-phage-design-research-evidence" / "SKILL.md").read_text(encoding="utf-8")

    assert "https://www.science.org/doi/10.1126/science.aec2657" in readme
    assert "https://www.biorxiv.org/content/10.1101/2025.09.12.675911v1.full" in readme
    for marker in (
        "CC BY bioRxiv v1",
        "king-2025-generative-phage-design",
        "final Science publication",
    ):
        assert marker in controller
    for marker in (
        "Before opening any other file in a bundled publication",
        "Begin every response that uses bundled evidence with a source note",
        "source version and license",
        "stable identifier",
        "publication of record",
    ):
        assert marker in research
    assert "../bionemo-phage-design-research-evidence/SKILL.md#use-bundled-publications" in calibration


def test_safeguards_reach_operational_workflows() -> None:
    skills_root = RECIPE_SKILLS_DIR
    controller = (skills_root / "bionemo-phage-design" / "SKILL.md").read_text(encoding="utf-8")
    design_contract = (
        skills_root / "bionemo-phage-design" / "references" / "design-scope-and-viability.md"
    ).read_text(encoding="utf-8")
    objective_skill = (skills_root / "bionemo-phage-design-plan-rl-objectives" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    screen_skill = (skills_root / "bionemo-phage-design-generate-and-screen" / "SKILL.md").read_text(encoding="utf-8")
    reporting_contract = (
        skills_root / "bionemo-phage-design-generate-and-screen" / "references" / "reporting-contract.md"
    ).read_text(encoding="utf-8")

    assert "replication within eukaryotic cells" in controller
    assert "non-replicative eukaryotic entry or host-range work" in design_contract
    assert "prohibited endpoint, not a soft penalty" in objective_skill
    assert "whole-sequence cargo and lysogeny screens" in screen_skill
    assert "Computational QC does not establish biological viability" in reporting_contract

    for marker in (
        "reviewed release descriptor",
        "persistent content-addressed cache",
        "implicit `latest`",
    ):
        assert marker in controller
    for marker in (
        "configs/phage_safety_reference_controls.yaml",
        "NC_015209.1",
        "P01555",
        "P01556",
        "NC_000924.1",
        "AM999887.1",
        "NC_001604.1",
        "NC_001422.1",
        "NC_022054.2",
    ):
        assert marker in screen_skill

    objective_evals = json.loads(
        (skills_root / "bionemo-phage-design-plan-rl-objectives" / "evals" / "evals.json").read_text(encoding="utf-8")
    )
    assert "bionemo-phage-design-plan-rl-objectives-009-eukaryotic-replication-boundary" in {
        case["id"] for case in objective_evals["evals"]
    }


def test_execution_adapter_covers_site_aware_slurm_and_local_runtime_choice() -> None:
    skills_root = RECIPE_SKILLS_DIR
    adapter = (skills_root / "bionemo-phage-design-adapt-execution" / "SKILL.md").read_text(encoding="utf-8")
    contract = (
        skills_root / "bionemo-phage-design-adapt-execution" / "references" / "execution-contract.md"
    ).read_text(encoding="utf-8")
    evals = json.loads(
        (skills_root / "bionemo-phage-design-adapt-execution" / "evals" / "evals.json").read_text(encoding="utf-8")
    )
    eval_ids = {case["id"] for case in evals["evals"]}

    for marker in (
        "allowed_partitions",
        "maximum_walltime",
        "known-good site script",
        "control workstation",
        "head/login-node policy",
        "same job name and user",
        "first automatic continuation",
        "shared-disk transfer job",
        "host-uv",
        "already containerized",
        "Docker",
        "immutable",
        "atomic promotion",
    ):
        assert marker in contract
    for marker in ("maximum walltimes", "tmux", "bounded stage-specific `singleton` chain", "host `uv`", "Docker"):
        assert marker in adapter
    assert "bionemo-phage-design-adapt-execution-008-slurm-site-and-restart-chain" in eval_ids
    assert "bionemo-phage-design-adapt-execution-009-local-runtime-selection" in eval_ids


def test_recipe_skills_use_the_bionemo_phage_design_namespace() -> None:
    actual = {path.name for path in SKILL_ROOT.iterdir() if path.is_dir() and (path / "SKILL.md").is_file()}
    assert EXPECTED_RECIPE_SKILLS == actual
    for skill_name in sorted(actual):
        skill_dir = SKILL_ROOT / skill_name
        frontmatter_name = next(
            line.removeprefix("name:").strip()
            for line in (skill_dir / "SKILL.md").read_text(encoding="utf-8").splitlines()
            if line.startswith("name:")
        )
        assert skill_name == frontmatter_name
        for eval_path in sorted((skill_dir / "evals").glob("*.json")):
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
            assert skill_name == payload["skill_name"]
            for case in payload["evals"]:
                if case["expected_skill"] is not None:
                    assert skill_name == case["expected_skill"]
                assert case["id"].startswith(f"{skill_name}-"), case["id"]
    text_paths = [
        RECIPE_ROOT / "README.md",
        *(
            path
            for path in SKILL_ROOT.rglob("*")
            if path.is_file() and path.resolve() != Path(__file__).resolve() and path.suffix in {".json", ".md", ".py"}
        ),
    ]
    for path in text_paths:
        text = path.read_text(encoding="utf-8")
        assert "bionemo-bionemo-phage-design" not in text, str(path)
        for retired in RETIRED_LEAF_SKILL_NAMES:
            assert retired not in text, str(path)


def test_recipe_codex_plugin_manifest_exposes_the_skill_bundle() -> None:
    manifest_path = SKILL_ROOT.parent / ".codex-plugin" / "plugin.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "bionemo-phage-design" == manifest["name"]
    assert "./skills/" == manifest["skills"]
    assert re.search(r"^\d+\.\d+\.\d+$", manifest["version"])
    assert manifest["description"].strip()
    assert manifest["author"]["name"].strip()

    interface = manifest["interface"]
    for field in (
        "displayName",
        "shortDescription",
        "longDescription",
        "developerName",
        "category",
    ):
        assert interface[field].strip(), field
    assert interface["capabilities"]
    assert len(interface["defaultPrompt"]) >= 1
    assert len(interface["defaultPrompt"]) <= 3
    assert all(len(prompt) <= 128 for prompt in interface["defaultPrompt"])


def test_controller_defines_a_recipe_local_workspace_contract() -> None:
    contract = SKILL_ROOT / "bionemo-phage-design" / "references" / "workspace-contract.md"
    assert contract.is_file(), contract
    text = contract.read_text(encoding="utf-8")
    for marker in (
        "recipe-local package",
        "recipes/evo2_phage_gen",
        "repository_root",
        "read-only Git",
        "dirty state",
        "results/",
        "bionemo-phage-generation",
    ):
        assert marker in text
    for forbidden in (
        "git clone",
        "git fetch",
        "git switch",
        "pull/1699",
        "jstjohn/evo2_phage_gen",
        "VERSION >= 2.4",
        "globally installed",
        "checkout bundle",
        "installed-versus-checkout",
    ):
        assert forbidden not in text


def test_recipe_local_runtime_documents_cannot_reenter_portable_bootstrap() -> None:
    runtime_documents = [
        path for path in SKILL_ROOT.rglob("*.md") if path.name == "SKILL.md" or "references" in path.parts
    ]
    forbidden_patterns = {
        "checkout clone": r"(?im)^\s*git\s+clone\b",
        "checkout fetch": r"(?im)^\s*git\s+fetch\b",
        "checkout switch": r"(?im)^\s*git\s+switch\b",
        "compatibility version selection": r"(?i)VERSION\s*(?:>=|>|==)\s*2\.4",
        "portable checkout acquisition": r"(?i)\b(?:acquire|locate)\s+(?:a\s+)?(?:compatible\s+)?checkout\b",
        "installation comparison": r"(?i)(?:globally installed|installed-versus-checkout|checkout bundle|skill installation is never the checkout)",
    }
    for path in runtime_documents:
        text = path.read_text(encoding="utf-8")
        for description, pattern in forbidden_patterns.items():
            assert re.search(pattern, text) is None, f"{description}: {path}"

    controller = (SKILL_ROOT / "bionemo-phage-design" / "SKILL.md").read_text(encoding="utf-8")
    assert "package integrity error" in controller
    assert "Search available skill roots" not in controller


def test_recipe_local_evals_do_not_describe_an_external_installation() -> None:
    eval_path = SKILL_ROOT / "bionemo-phage-design-adapt-execution" / "evals" / "evals.json"
    text = eval_path.read_text(encoding="utf-8")
    assert "skill may be installed outside the checkout" not in text
    assert "recipe-local package" in text


def test_controller_resolves_roots_in_dependency_order() -> None:
    text = (SKILL_ROOT / "bionemo-phage-design" / "SKILL.md").read_text(encoding="utf-8").lower()
    repository = text.index("resolve the absolute recipe and repository roots")
    workspace = text.index("select the recipe workspace")
    result = text.index("record absolute recipe and result roots")
    command = text.index("before emitting recipe commands")
    assert repository < workspace
    assert workspace < result
    assert result < command


def test_validation_examples_resolve_the_recipe_local_runner() -> None:
    validation = (SKILL_ROOT / "bionemo-phage-design" / "assets" / "VALIDATION.md").read_text(encoding="utf-8")
    skill_root_match = re.search(
        r'^PHAGE_SKILL_ROOT="\$\{PHAGE_SKILL_ROOT:-\$PWD/(?P<relative>[^"}]+)\}"$',
        validation,
        flags=re.MULTILINE,
    )
    assert skill_root_match is not None
    assert skill_root_match is not None
    relative_skill_root = skill_root_match.group("relative")
    assert relative_skill_root == "recipes/evo2_phage_gen/skills"
    assert 'PHAGE_SKILL_ROOT="${PHAGE_SKILL_ROOT:-$PWD/skills}"' not in validation

    runner_match = re.search(
        r'^PHAGE_EVAL_RUNNER="\$PHAGE_SKILL_ROOT/(?P<relative>[^"]+)"$',
        validation,
        flags=re.MULTILINE,
    )
    assert runner_match is not None
    assert runner_match is not None
    documented_runner = REPOSITORY_ROOT / relative_skill_root / runner_match.group("relative")
    assert documented_runner.is_file(), documented_runner
    assert documented_runner.resolve() == SCRIPT.resolve()

    for marker in (
        'python "$PHAGE_EVAL_RUNNER"',
        '--skill-root "$PHAGE_SKILL_ROOT"',
        "Git-tracked plugin directory",
    ):
        assert marker in validation


def test_controller_has_a_capacity_and_checkpoint_retention_gate() -> None:
    controller = (SKILL_ROOT / "bionemo-phage-design" / "SKILL.md").read_text(encoding="utf-8").lower()
    for marker in (
        "storage-planning.md",
        "total bases",
        "91 gb per sft checkpoint",
        "78 gb per rl checkpoint",
        "68 mb training-ready",
        "one checkpoint write",
        "latest resumable",
        "never prune active",
        "ask the user",
    ):
        assert marker in controller

    storage = (
        (SKILL_ROOT / "bionemo-phage-design" / "references" / "storage-planning.md")
        .read_text(encoding="utf-8")
        .lower()
    )
    for marker in (
        "b = sum",
        "1.13 bytes/base",
        "2.34 bytes/base",
        "6.40 bytes/base",
        "total generated bases",
    ):
        assert marker in storage


def test_sft_and_rl_context_uses_an_agreed_genome_length_rule() -> None:
    policy = (
        (SKILL_ROOT / "bionemo-phage-design-adapt-execution" / "references" / "resource-and-oom-policy.md")
        .read_text(encoding="utf-8")
        .lower()
    )
    for marker in (
        "after the training genomes are downloaded",
        "agreed",
        "default to p99.9",
        "align_up",
        "tokens per nucleotide",
        "worst-case serialization overhead",
        "required alignment",
        "conditioning/control",
        "model context limit",
        "sft context",
        "rl context",
        "expand or contract",
        "do not assume",
    ):
        assert marker in policy

    active_instruction_paths = [
        *SKILL_ROOT.glob("*/SKILL.md"),
        *SKILL_ROOT.glob("*/evals/*.json"),
        *SKILL_ROOT.glob("*/references/*.md"),
        RECIPE_ROOT / "README.md",
    ]
    fixed_context = re.compile(r"(?<!\d)(?:10(?:,|_|\s)?240|327(?:,|_|\s)?680)(?!\d)")
    prohibited_caveats = (
        " ".join(("materialized", "windows")),
        " ".join(("indexed", "windows")),
        "-".join(("chunk", "local")) + " learning",
        "-".join(("window", "local")) + " learning",
    )
    for path in active_instruction_paths:
        text = path.read_text(encoding="utf-8").lower()
        assert fixed_context.search(text) is None, str(path)
        for marker in prohibited_caveats:
            assert marker not in text, str(path)


def test_sft_alternate_base_model_uses_current_gpu_compatibility_table() -> None:
    operation = (
        (SKILL_ROOT / "bionemo-phage-design-operate-mbridge-sft" / "references" / "sft-operation.md")
        .read_text(encoding="utf-8")
        .lower()
    )

    for marker in (
        "phage recipe uses bf16",
        "7b 8k/1m",
        "different base-model family",
        "gpu/precision compatibility table",
        "evo2_megatron/readme.md",
        "before selecting or downloading",
    ):
        assert marker in operation


def test_sft_step_ceiling_is_recalibrated_after_collection() -> None:
    controller = (SKILL_ROOT / "bionemo-phage-design" / "SKILL.md").read_text(encoding="utf-8").lower()
    prepare = (SKILL_ROOT / "bionemo-phage-design-prepare-sft" / "SKILL.md").read_text(encoding="utf-8").lower()
    operate = (
        (SKILL_ROOT / "bionemo-phage-design-operate-mbridge-sft" / "SKILL.md").read_text(encoding="utf-8").lower()
    )
    operation = (
        (SKILL_ROOT / "bionemo-phage-design-operate-mbridge-sft" / "references" / "sft-operation.md")
        .read_text(encoding="utf-8")
        .lower()
    )

    assert "post-collection training-budget feedback" in controller
    for text in (prepare, operate, operation):
        assert "historical starting hypothesis" in text
        assert "not a fixed maximum" in text
        assert "above 12,000" in text
    for marker in (
        "usable non-padding token mass",
        "effective non-padding tokens per optimizer step",
        "target effective exposures",
        "planned_steps = ceil",
    ):
        assert marker in operation
    assert "at least 30" in operation
    assert "six-event patience" in operation
    assert "calibrated ceiling" in operate
    assert "set at most 12,000 optimizer steps" not in operate
    assert "set a 12,000-step maximum" not in prepare


def test_prepare_sft_defines_target_similarity_conditioning() -> None:
    skill = (SKILL_ROOT / "bionemo-phage-design-prepare-sft" / "SKILL.md").read_text(encoding="utf-8").lower()
    assert "target-conditioning.md" in skill

    contract = (
        (SKILL_ROOT / "bionemo-phage-design-prepare-sft" / "references" / "target-conditioning.md")
        .read_text(encoding="utf-8")
        .lower()
    )
    for marker in (
        "single unambiguous target",
        "default",
        "opt out",
        "unprefixed",
        "target_similarity.tsv",
        "conditioning.yaml",
        "reference hash",
        "identity and coverage",
        "one-to-one",
        "case-study replication",
        "adapted run",
        "rl prompts",
        "original paper",
        "king-2025-generative-phage-design/supplement.md",
    ):
        assert marker in contract


def test_active_skills_defer_biological_safety_policy_to_the_harness() -> None:
    active_instruction_paths = [
        *SKILL_ROOT.glob("*/SKILL.md"),
        *SKILL_ROOT.glob("*/evals/*.json"),
        *SKILL_ROOT.glob("*/references/*.md"),
    ]
    prohibited_markers = (
        "## safety boundary",
        "## biological safety policy",
        "custom biological policy",
        "prokaryotic-only policy",
    )
    for path in active_instruction_paths:
        text = path.read_text(encoding="utf-8").lower()
        assert "385" not in text, str(path)
        for marker in prohibited_markers:
            assert marker not in text, str(path)


def test_research_skill_keeps_its_artifact_contract_in_read_only_responses() -> None:
    skill_path = SKILL_ROOT / "bionemo-phage-design-research-evidence" / "SKILL.md"
    text = skill_path.read_text(encoding="utf-8")
    assert "read-only" in text.lower()
    for artifact in (
        "artifacts/EVIDENCE.md",
        "artifacts/SOURCES.yaml",
        "artifacts/SEARCH_LOG.jsonl",
        "artifacts/DATASET_CANDIDATES.yaml",
        "OUTPUTS.yaml",
        "SUMMARY.md",
        "RUNLOG.md",
    ):
        assert artifact in text


def test_execution_and_collection_evals_are_compatible_with_read_only_generation() -> None:
    adapt_path = SKILL_ROOT / "bionemo-phage-design-adapt-execution" / "evals" / "evals.json"
    adapt = json.loads(adapt_path.read_text(encoding="utf-8"))["evals"]
    local_preflight = next(case for case in adapt if case["id"].endswith("local-preflight"))
    lepton = next(case for case in adapt if case["id"].endswith("lepton-contract"))
    assert "read-only" in local_preflight["assertions"][-1].lower()
    assert "when available" in lepton["assertions"][0].lower()
    assert "orders" in lepton["assertions"][1].lower()

    collection_path = SKILL_ROOT / "bionemo-phage-design-collect-genomes" / "evals" / "evals.json"
    collection = json.loads(collection_path.read_text(encoding="utf-8"))["evals"]
    jumbo = next(case for case in collection if case["id"].endswith("current-jumbo-corpus"))
    assert "leaves the payload unverified" in jumbo["assertions"][2].lower()


def test_execution_skill_preserves_read_only_handoff_and_lepton_egress() -> None:
    skill_path = SKILL_ROOT / "bionemo-phage-design-adapt-execution" / "SKILL.md"
    text = skill_path.read_text(encoding="utf-8").lower()
    assert "read-only" in text
    lepton = text.split("- **lepton:**", maxsplit=1)[1].split("\n", maxsplit=1)[0]
    assert "egress" in lepton


def test_local_docker_credentials_are_optional_and_not_logged() -> None:
    skill_path = SKILL_ROOT / "bionemo-phage-design-adapt-execution" / "SKILL.md"
    contract_path = SKILL_ROOT / "bionemo-phage-design-adapt-execution" / "references" / "execution-contract.md"
    text = (skill_path.read_text(encoding="utf-8") + contract_path.read_text(encoding="utf-8")).lower()
    for marker in (
        "~/.aws",
        ".netrc",
        "read-only",
        "resolved container user home",
        "--env wandb_api_key",
        "never pass name=value",
        "never bake, copy, or log",
        "shared or untrusted runner",
        "does not block the scientific run",
    ):
        assert marker in text


def test_collection_skill_stops_when_prefix_tools_are_unavailable() -> None:
    skill_path = SKILL_ROOT / "bionemo-phage-design-collect-genomes" / "SKILL.md"
    text = " ".join(skill_path.read_text(encoding="utf-8").lower().split())
    assert "bounded validation command" in text
    assert "unverified" in text


def test_collection_read_only_response_keeps_query_and_access_provenance() -> None:
    skill_path = SKILL_ROOT / "bionemo-phage-design-collect-genomes" / "SKILL.md"
    text = skill_path.read_text(encoding="utf-8").lower()
    assert "every response" in text
    assert "exact retrieval query" in text
    assert "access date" in text


def test_sft_due_monitoring_reports_health_without_forcing_not_due_polling() -> None:
    skill_path = SKILL_ROOT / "bionemo-phage-design-operate-mbridge-sft" / "SKILL.md"
    text = skill_path.read_text(encoding="utf-8").lower()
    assert "every substantive or due monitoring decision" in text
    for metric in (
        "train and validation loss",
        "learning rate",
        "gradient norm",
        "throughput",
        "gpu utilization and memory",
        "checkpoint integrity",
        "free space",
    ):
        assert metric in text
    assert "not-due" in text
    assert "last-observed timestamp and staleness" in text
    assert "without querying" in text


def test_rl_relaunch_requires_independent_prior_job_terminal_check() -> None:
    skill_path = SKILL_ROOT / "bionemo-phage-design-operate-nemo-rl" / "SKILL.md"
    text = skill_path.read_text(encoding="utf-8").lower()
    assert "before any resume or relaunch" in text
    assert "prior process/job is absent or terminal" in text
    assert "never duplicate a live submission" in text


def test_rl_objective_audit_states_telemetry_contract_before_code() -> None:
    skill_path = SKILL_ROOT / "bionemo-phage-design-implement-rl-objectives" / "SKILL.md"
    text = skill_path.read_text(encoding="utf-8").lower()
    assert "even when this review stops before code" in text
    assert "numerator, denominator, observation count" in text
    assert "test comparing online scoring with final qc" in text
    assert "owning recipe `src/` and `tests/`" in text
    assert "installed-runtime name/order/dtype/device/shape/reduction checks" in text
    assert "tiny deterministic rl smoke" in text
    assert "reward calculation, logging, checkpoint writing, and restart metadata" in text


def test_external_tool_optimization_policy_is_shared_across_sft_rl_and_final_filters() -> None:
    policy_path = SKILL_ROOT / "bionemo-phage-design-adapt-execution" / "references" / "resource-and-oom-policy.md"
    text = " ".join(policy_path.read_text(encoding="utf-8").lower().split())
    for marker in (
        "external-tool filtering and scoring",
        "end-to-end and per-stage timings",
        "isolated and at the planned concurrency",
        "total throughput under load",
        "record workers and per-tool threads",
        "process or container startup",
        "warm, batched, or persistent",
        "byte or semantic parity",
        "deterministic input/output mapping",
        "actual operation and database layout",
        "do not generalize one tool's result",
    ):
        assert marker in text

    anchor = "resource-and-oom-policy.md#external-tool-filtering-and-scoring"
    for relative_path in (
        "bionemo-phage-design-prepare-sft/SKILL.md",
        "bionemo-phage-design-implement-rl-objectives/SKILL.md",
        "bionemo-phage-design-generate-and-screen/SKILL.md",
    ):
        workflow = (SKILL_ROOT / relative_path).read_text(encoding="utf-8").lower()
        assert anchor in workflow

    execution_skill = (SKILL_ROOT / "bionemo-phage-design-adapt-execution" / "SKILL.md").read_text(encoding="utf-8")
    for marker in (
        "explicitly record the accelerator decision",
        "actual operation and database layout",
        "control-panel parity",
    ):
        assert marker in execution_skill


def test_gpu_optimization_policy_is_shared_across_sft_rl_and_generation() -> None:
    policy_path = SKILL_ROOT / "bionemo-phage-design-adapt-execution" / "references" / "resource-and-oom-policy.md"
    text = " ".join(policy_path.read_text(encoding="utf-8").lower().split())
    for marker in (
        "gpu training, rl, and generation",
        "memory occupancy is a constraint, not the objective",
        "stable useful tokens or valid sequences per second",
        "representative target-length",
        "memory headroom",
        "checkpoint and resume",
    ):
        assert marker in text

    anchor = "resource-and-oom-policy.md#gpu-training-rl-and-generation"
    for relative_path in (
        "bionemo-phage-design-operate-mbridge-sft/SKILL.md",
        "bionemo-phage-design-operate-nemo-rl/SKILL.md",
        "bionemo-phage-design-generate-and-screen/SKILL.md",
    ):
        workflow = (SKILL_ROOT / relative_path).read_text(encoding="utf-8").lower()
        assert anchor in workflow


def test_behavioral_evals_require_use_of_bundled_docs_and_papers() -> None:
    adapt = json.loads(
        (SKILL_ROOT / "bionemo-phage-design-adapt-execution" / "evals" / "evals.json").read_text(encoding="utf-8")
    )["evals"]
    contention = next(case for case in adapt if case["id"].endswith("external-tool-contention"))
    contention_requirements = " ".join(contention["assertions"]).lower()
    assert "resource-and-oom-policy.md#external-tool-filtering-and-scoring" in contention_requirements
    assert "isolated" in contention_requirements
    assert "planned concurrency" in contention_requirements
    assert "output parity" in contention_requirements

    calibration = json.loads(
        (SKILL_ROOT / "bionemo-phage-design-calibrate-rl-sampling" / "evals" / "evals.json").read_text(
            encoding="utf-8"
        )
    )["evals"]
    king = next(case for case in calibration if case["id"].endswith("bundled-king-methods"))
    king_requirements = " ".join(king["assertions"]).lower()
    for marker in (
        "manifest.json",
        "king-2025-generative-phage-design/supplement.md",
        "bioRxiv v1".lower(),
        "1 to 11",
        "top-k = 4",
        "top-p = 1",
    ):
        assert marker in king_requirements

    research = json.loads(
        (SKILL_ROOT / "bionemo-phage-design-research-evidence" / "evals" / "evals.json").read_text(encoding="utf-8")
    )["evals"]
    black = next(case for case in research if case["id"].endswith("bundled-black-framework"))
    black_requirements = " ".join(black["assertions"]).lower()
    for marker in (
        "black-2026-design-efficiency/manifest.json",
        "10.64898/2026.06.12.731871",
        "evolutionary novelty",
        "design efficiency",
        "complete model-scaffold",
        "transfer",
    ):
        assert marker in black_requirements


def test_research_packet_requires_complete_objective_portfolio_rows() -> None:
    skill_path = SKILL_ROOT / "bionemo-phage-design-research-evidence" / "SKILL.md"
    text = skill_path.read_text(encoding="utf-8").lower()
    assert "portfolio coverage checklist" in text
    assert "topology/packaging" in text
    assert "desired and undesired directional changes" in text
    assert "each axis a decision-table row or mark it unresolved/not applicable" in text
    assert "material interactions across the portfolio" in text


def test_code_building_skills_prioritize_single_operator_research_failures() -> None:
    for skill_name in (
        "bionemo-phage-design-implement-rl-objectives",
        "bionemo-phage-design-adapt-execution",
    ):
        skill = (SKILL_ROOT / skill_name / "SKILL.md").read_text(encoding="utf-8").lower()
        for marker in (
            "single-operator research software",
            "scientific correctness",
            "reproducibility",
            "restartability",
            "accidental drift",
            "operator-owned scratch files",
        ):
            assert marker in skill


def test_monitoring_heartbeat_covers_all_long_running_work() -> None:
    controller = (SKILL_ROOT / "bionemo-phage-design" / "SKILL.md").read_text(encoding="utf-8").lower()
    adapter_root = SKILL_ROOT / "bionemo-phage-design-adapt-execution"
    execution = (
        (adapter_root / "SKILL.md").read_text(encoding="utf-8")
        + (adapter_root / "references" / "execution-contract.md").read_text(encoding="utf-8")
    ).lower()

    assert "not only sft or rl" in controller
    assert "every long-running stage" in execution
    for marker in (
        "background launch is not a completed handoff",
        "meaningful progress events",
        "verified terminal success or failure",
        "exit status",
        "actionable error",
    ):
        assert marker in execution

    for skill_name in ("bionemo-phage-design-collect-genomes", "bionemo-phage-design-prepare-sft"):
        leaf = (SKILL_ROOT / skill_name / "SKILL.md").read_text(encoding="utf-8").lower()
        assert "bionemo-phage-design-adapt-execution" in leaf
        assert "long-running" in leaf

    eval_path = adapter_root / "evals" / "evals.json"
    evals = json.loads(eval_path.read_text(encoding="utf-8"))["evals"]
    case = next(case for case in evals if case["id"].endswith("background-heartbeat"))
    prompt = case["prompt"].lower()
    assertions = " ".join(case["assertions"]).lower()
    eval_text = f"{prompt} {case['expected_output'].lower()} {assertions}"
    assert len(prompt) < 120
    assert len(case["assertions"]) <= 3
    for marker in ("long-running task", "background", "monitor"):
        assert marker in prompt
    for marker in ("every long-running or background task", "next due check", "verified terminal"):
        assert marker in assertions
    for workload_detail in ("genome", "phage", "download", "preprocessing", "filtering", "sft", "rl"):
        assert workload_detail not in eval_text


def test_entry_description() -> None:
    skill = (SKILL_ROOT / "bionemo-phage-design" / "SKILL.md").read_text(encoding="utf-8")
    frontmatter = skill.split("---", maxsplit=2)[1].lower()

    for marker in ("bacteriophage genome", "phage therapy", "antibiotic-resistant infections"):
        assert marker in frontmatter


def test_scanner_topology_guidance_matches_the_cli() -> None:
    command_resolution = (
        SKILL_ROOT / "bionemo-phage-design-generate-and-screen" / "references" / "command-resolution.md"
    ).read_text(encoding="utf-8")
    cli_source = (SCRIPT.parents[3] / "src" / "bionemo" / "evo2_phage_gen" / "sequence_safety_cli.py").read_text(
        encoding="utf-8"
    )

    for marker in ("circular -> no topology argument", "linear -> `--linear`", "reject any other topology"):
        assert marker in command_resolution
    assert 'scan.add_argument("--linear"' in cli_source
    assert 'scan.add_argument("--circular"' not in cli_source
