"""Typed evidence for Stage A physical feasibility and proposal compatibility."""

from __future__ import annotations

from typing import Any

from .phase3b_stage_a import canonical_sha256


PROPOSAL_FEASIBILITY_KIND = "successful_proposal_ledger_v1"
FACTORIZED_FEASIBILITY_KIND = "factorized_policy_free_path_v1"
FACTORIZED_EXECUTION_MODE = (
    "registered_acquisition_then_goal_registered_feedback_early_stop_v1"
)


def _sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"Invalid SHA-256 field: {field}")
    return value


def proposal_feasibility_evidence(
    oracle: dict[str, Any], *, goal: str
) -> dict[str, Any]:
    """Create a compact reference to a successful proposal-bank ledger."""

    if oracle.get("pass") is not True or oracle.get("goal_ever_achieved") is not True:
        raise ValueError(f"Cannot certify {goal} from an unsuccessful proposal ledger")
    return {
        "schema_version": 1,
        "kind": PROPOSAL_FEASIBILITY_KIND,
        "goal": goal,
        "pass": True,
        "policy_loaded": False,
        "proposal_ledger_sha256": canonical_sha256(oracle),
        "normalized_state_sha256": _sha(
            oracle.get("shared_normalized_state_sha256"),
            field="shared_normalized_state_sha256",
        ),
        "normalization_action_sha256": _sha(
            oracle.get("shared_normalization_action_sha256"),
            field="shared_normalization_action_sha256",
        ),
    }


def build_factorized_feasibility_evidence(
    *,
    result: dict[str, Any],
    manifest: dict[str, Any],
    contract: dict[str, Any],
    artifact_file_sha256: dict[str, str],
) -> dict[str, Any]:
    """Validate a raw v4 diagnostic and distil a provenance-bound certificate."""

    if (
        result.get("status") != "complete"
        or result.get("pass") is not True
        or result.get("policy_loaded") is not False
        or result.get("completed_full_suffix_baselines_reexecuted") != 0
        or result.get("source_proposal_full_tail_executed") is not False
        or manifest.get("status") != "complete"
        or manifest.get("pass") is not True
        or manifest.get("policy_loaded") is not False
        or manifest.get("completed_full_suffix_baselines_reexecuted") != 0
    ):
        raise ValueError("Raw factorized certificate did not pass its scope gates")
    contract_sha = canonical_sha256(contract)
    if (
        result.get("contract_sha256") != contract_sha
        or manifest.get("contract_sha256") != contract_sha
        or manifest.get("artifact_sha256", {}).get("result.json")
        != artifact_file_sha256.get("result.json")
        or manifest.get("artifact_sha256", {}).get("acquisition.json")
        != artifact_file_sha256.get("acquisition.json")
        or manifest.get("artifact_sha256", {}).get("contract.json")
        != artifact_file_sha256.get("contract.json")
    ):
        raise ValueError("Raw factorized artifact hashes are inconsistent")

    reconstruction = result.get("reconstruction", {})
    normalization = reconstruction.get("normalization", {})
    acquisition = result.get("acquisition", {})
    placement = result.get("placement", {})
    phases = placement.get("transport_phases")
    if (
        reconstruction.get("certificate_pass") is not True
        or normalization.get("source_proposal_replayed") is not False
        or normalization.get("normalization_done_ever") is not False
        or normalization.get("normalization_goal_ever") is not False
        or acquisition.get("pass") is not True
        or acquisition.get("bridge_pass") is not True
        or placement.get("status") != "complete"
        or placement.get("pass") is not True
        or placement.get("transport_pass") is not True
        or placement.get("bowl_released") is not True
        or placement.get("goal_ever_achieved") is not True
        or placement.get("wrong_goal_ever_achieved") is not False
        or placement.get("unexpected_done_before_goal") is not False
        or placement.get("final_goals")
        != {"drawer": False, "cabinet": True}
        or not isinstance(phases, list)
        or [phase.get("phase") for phase in phases]
        != ["clearance_lift", "clearance_transit", "target_descent"]
    ):
        raise ValueError("Raw factorized physical gates are incomplete")
    trace = acquisition.get("trace")
    stable_streak = int(acquisition.get("stable_grasp_streak", 0))
    first_stable_frame = int(
        acquisition.get("first_stable_grasp_source_frame", -1)
    )
    if (
        not isinstance(trace, list)
        or stable_streak < 1
        or len(trace) < stable_streak
        or acquisition.get("trace_sha256") != canonical_sha256(trace)
        or any(item.get("bowl_grasped") is not True for item in trace[-stable_streak:])
        or int(trace[-1].get("source_frame", -1)) != first_stable_frame
        or int(acquisition.get("source_actions_executed", -1)) != len(trace)
    ):
        raise ValueError("Raw factorized acquisition stability is invalid")
    transport = acquisition.get("factorized_transport", {})
    if transport != {
        "phase_budget_policy": "ceiling_with_early_stop_on_tolerance",
        "pad_to_budget": False,
        "goal_or_terminal_stop": True,
        "physical_tolerances": "unchanged_from_config",
    }:
        raise ValueError("Raw factorized horizon semantics changed")

    phase_rows = []
    for phase in phases:
        phase_result = phase.get("result", {})
        budgeted = int(phase_result.get("budgeted_action_steps", -1))
        executed = int(phase_result.get("executed_action_steps", -1))
        if (
            phase_result.get("pass") is not True
            or phase_result.get("padded_to_budget") is not False
            or phase_result.get("stop_on_goal_or_terminal") is not True
            or budgeted < 1
            or executed < 1
            or executed > budgeted
        ):
            raise ValueError("Raw factorized transport phase is invalid")
        phase_rows.append(
            {
                "phase": phase["phase"],
                "budget_ceiling_action_steps": budgeted,
                "executed_action_steps": executed,
                "active_action_steps": int(
                    phase_result["active_action_steps"]
                ),
                "final_position_error_m": float(
                    phase_result["final_position_error_m"]
                ),
                "stopped_on_goal": bool(phase_result["stopped_on_goal"]),
                "stopped_on_terminal": bool(
                    phase_result["stopped_on_terminal"]
                ),
                "bowl_grasped_after_phase": bool(
                    phase["bowl_grasped_after_phase"]
                ),
            }
        )
    if (
        sum(row["executed_action_steps"] for row in phase_rows)
        != int(placement.get("action_count", -1))
        or phase_rows[-1]["stopped_on_goal"] is not True
        or phase_rows[-1]["stopped_on_terminal"] is not False
    ):
        raise ValueError("Raw factorized action accounting is inconsistent")

    source = contract.get("source_evidence", {})
    if (
        source.get("root_state_sha256")
        != reconstruction.get("root_state_sha256")
        or source.get("normalized_state_sha256")
        != normalization.get("normalized_state_sha256")
        or source.get("normalization_action_sha256")
        != normalization.get("normalization_action_sha256")
        or source.get("normalization_action_steps")
        != normalization.get("normalization_action_steps")
        or source.get("completed_outcome", {}).get("pass") is not False
        or source.get("completed_outcome", {}).get("goal_ever_achieved")
        is not False
    ):
        raise ValueError("Factorized certificate is not bound to the negative root")

    evidence = {
        "schema_version": 1,
        "kind": FACTORIZED_FEASIBILITY_KIND,
        "execution_mode": FACTORIZED_EXECUTION_MODE,
        "goal": "cabinet",
        "pass": True,
        "policy_loaded": False,
        "root_state_sha256": _sha(
            reconstruction.get("root_state_sha256"), field="root_state_sha256"
        ),
        "normalized_state_sha256": _sha(
            normalization.get("normalized_state_sha256"),
            field="normalized_state_sha256",
        ),
        "normalization_action_sha256": _sha(
            normalization.get("normalization_action_sha256"),
            field="normalization_action_sha256",
        ),
        "normalization_action_steps": int(
            normalization["normalization_action_steps"]
        ),
        "source_proposal": {
            "proposal_index": int(contract["proposal_index"]),
            "episode_index": int(contract["proposal_episode"]),
            "action_sha256": _sha(
                source.get("proposal_action_sha256"),
                field="proposal_action_sha256",
            ),
            "source_actions_executed": int(
                acquisition["source_actions_executed"]
            ),
            "full_tail_executed": False,
        },
        "acquisition": {
            "acquisition_state_sha256": _sha(
                acquisition.get("acquisition_state_sha256"),
                field="acquisition_state_sha256",
            ),
            "bridge_action_count": int(acquisition["bridge_action_count"]),
            "bridge_action_sha256": _sha(
                acquisition.get("bridge_action_sha256"),
                field="bridge_action_sha256",
            ),
            "first_stable_grasp_source_frame": first_stable_frame,
            "stable_grasp_streak": stable_streak,
            "trace_sha256": _sha(
                acquisition.get("trace_sha256"), field="trace_sha256"
            ),
        },
        "placement": {
            "action_count": int(placement["action_count"]),
            "action_sha256": _sha(
                placement.get("action_sha256"), field="placement_action_sha256"
            ),
            "transport_phases": phase_rows,
            "final_goals": placement["final_goals"],
            "bowl_released": True,
            "wrong_goal_ever_achieved": False,
            "unexpected_done_before_goal": False,
        },
        "execution_scope": {
            "completed_full_suffix_baselines_reexecuted": 0,
            "source_proposal_full_tail_executed": False,
            "policy_forwards": int(
                contract.get("execution_scope", {}).get("policy_forwards", -1)
            ),
        },
        "source_artifact": {
            "diagnostic_revision": result.get("diagnostic_revision"),
            "condition": result.get("condition"),
            "contract_sha256": contract_sha,
            "contract_file_sha256": _sha(
                artifact_file_sha256.get("contract.json"),
                field="contract_file_sha256",
            ),
            "manifest_file_sha256": _sha(
                artifact_file_sha256.get("manifest.json"),
                field="manifest_file_sha256",
            ),
            "result_file_sha256": _sha(
                artifact_file_sha256.get("result.json"),
                field="result_file_sha256",
            ),
            "acquisition_file_sha256": _sha(
                artifact_file_sha256.get("acquisition.json"),
                field="acquisition_file_sha256",
            ),
            "source_checkpoint_file_sha256": _sha(
                source.get("checkpoint_file_sha256"),
                field="source_checkpoint_file_sha256",
            ),
        },
        "scientific_boundary": result.get("scientific_boundary"),
    }
    validate_factorized_feasibility_evidence(evidence)
    return evidence


def validate_factorized_feasibility_evidence(
    evidence: dict[str, Any],
    *,
    candidate_id: str | None = None,
    goal: str = "cabinet",
    root_state_sha256: str | None = None,
    normalized_state_sha256: str | None = None,
    normalization_action_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail closed on the compact, independently portable certificate."""

    if (
        evidence.get("schema_version") != 1
        or evidence.get("kind") != FACTORIZED_FEASIBILITY_KIND
        or evidence.get("execution_mode") != FACTORIZED_EXECUTION_MODE
        or evidence.get("goal") != goal
        or evidence.get("pass") is not True
        or evidence.get("policy_loaded") is not False
    ):
        raise ValueError("Invalid factorized feasibility certificate identity")
    for field, expected in (
        ("root_state_sha256", root_state_sha256),
        ("normalized_state_sha256", normalized_state_sha256),
        ("normalization_action_sha256", normalization_action_sha256),
    ):
        observed = _sha(evidence.get(field), field=field)
        if expected is not None and observed != expected:
            label = candidate_id or "candidate"
            raise ValueError(f"Factorized {field} mismatch for {label}")
    if int(evidence.get("normalization_action_steps", 0)) < 1:
        raise ValueError("Factorized normalization action count is invalid")
    source = evidence.get("source_proposal", {})
    acquisition = evidence.get("acquisition", {})
    placement = evidence.get("placement", {})
    scope = evidence.get("execution_scope", {})
    phases = placement.get("transport_phases")
    if (
        int(source.get("proposal_index", -1)) < 0
        or int(source.get("episode_index", -1)) < 0
        or int(source.get("source_actions_executed", 0)) < 1
        or source.get("full_tail_executed") is not False
        or int(acquisition.get("bridge_action_count", 0)) < 1
        or int(acquisition.get("first_stable_grasp_source_frame", -1))
        != int(source.get("source_actions_executed", -2))
        or int(acquisition.get("stable_grasp_streak", 0)) < 1
        or placement.get("final_goals")
        != {"drawer": False, "cabinet": True}
        or placement.get("bowl_released") is not True
        or placement.get("wrong_goal_ever_achieved") is not False
        or placement.get("unexpected_done_before_goal") is not False
        or scope
        != {
            "completed_full_suffix_baselines_reexecuted": 0,
            "source_proposal_full_tail_executed": False,
            "policy_forwards": 0,
        }
        or not isinstance(phases, list)
        or [item.get("phase") for item in phases]
        != ["clearance_lift", "clearance_transit", "target_descent"]
    ):
        raise ValueError("Invalid factorized feasibility certificate payload")
    executed = 0
    for phase in phases:
        budget = int(phase.get("budget_ceiling_action_steps", -1))
        steps = int(phase.get("executed_action_steps", -1))
        if steps < 1 or budget < steps:
            raise ValueError("Factorized placement exceeds its action ceiling")
        executed += steps
    if (
        executed != int(placement.get("action_count", -1))
        or phases[-1].get("stopped_on_goal") is not True
        or phases[-1].get("stopped_on_terminal") is not False
    ):
        raise ValueError("Invalid factorized placement action accounting")
    _sha(source.get("action_sha256"), field="source_action_sha256")
    for field in (
        "acquisition_state_sha256",
        "bridge_action_sha256",
        "trace_sha256",
    ):
        _sha(acquisition.get(field), field=field)
    _sha(placement.get("action_sha256"), field="placement_action_sha256")
    for field, value in evidence.get("source_artifact", {}).items():
        if field.endswith("sha256"):
            _sha(value, field=field)
    return {
        "pass": True,
        "kind": FACTORIZED_FEASIBILITY_KIND,
        "execution_mode": FACTORIZED_EXECUTION_MODE,
        "goal": goal,
        "normalized_state_sha256": evidence["normalized_state_sha256"],
        "normalization_action_sha256": evidence[
            "normalization_action_sha256"
        ],
        "placement_action_count": int(placement["action_count"]),
    }


def resolve_goal_feasibility_evidence(
    record: dict[str, Any], *, goal: str
) -> dict[str, Any]:
    """Return explicit evidence or infer the historical proposal certificate."""

    explicit = record.get("goal_feasibility_evidence", {}).get(goal)
    if explicit is not None:
        return explicit
    return proposal_feasibility_evidence(record["oracles"][goal], goal=goal)
