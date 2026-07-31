from __future__ import annotations

from typing import Any, Iterable

from .phase3b_stage_a import candidate_spec, canonical_sha256


COMPLETION_REVISION = "phase3b-stage-a-v35-additive-completion-v1"


def validate_completion_candidate_ids(
    candidate_ids: Iterable[str], *, expected_count: int
) -> tuple[str, ...]:
    selected = tuple(str(candidate_id) for candidate_id in candidate_ids)
    if len(selected) != expected_count or len(set(selected)) != len(selected):
        raise ValueError("Stage A completion selection count or uniqueness changed")
    specs = tuple(candidate_spec(candidate_id) for candidate_id in selected)
    if any(spec.drawer_aperture != "open" for spec in specs):
        raise ValueError("Stage A completion may contain only missing open roots")
    return selected


def proposal_inventory(proposals: Iterable[Any]) -> list[dict[str, Any]]:
    return [
        {
            "proposal_index": index,
            "episode_index": proposal.episode_index,
            "task_index": proposal.task_index,
            "action_sha256": proposal.action_sha256,
        }
        for index, proposal in enumerate(proposals)
    ]


def validate_imported_checkpoint(
    checkpoint: dict[str, Any],
    *,
    candidate_id: str,
    goal: str,
    root_state_sha256: str,
    source_contract_sha256: str,
    source_selection_lock_sha256: str,
    proposals: tuple[Any, ...],
    phase_proposals: tuple[Any, ...],
) -> dict[int, dict[str, Any]]:
    expected = {
        "candidate_id": candidate_id,
        "goal": goal,
        "root_state_sha256": root_state_sha256,
        "contract_sha256": source_contract_sha256,
        "selection_lock_sha256": source_selection_lock_sha256,
        "proposal_inventory_sha256": canonical_sha256(
            proposal_inventory(proposals)
        ),
        "proposal_execution_contract_sha256": canonical_sha256(
            [proposal.metadata for proposal in phase_proposals]
        ),
    }
    for field, value in expected.items():
        if checkpoint.get(field) != value:
            raise ValueError(
                f"Imported checkpoint mismatch for {candidate_id}/{goal}: {field}"
            )
    rows = checkpoint.get("results")
    if not isinstance(rows, list) or len(rows) != len(proposals):
        raise ValueError("Imported checkpoint is not an exhaustive proposal ledger")
    if checkpoint.get("status") != "complete" or len(
        str(checkpoint.get("oracle_sha256", ""))
    ) != 64:
        raise ValueError("Imported positive checkpoint is not complete")
    if int(checkpoint.get("result_count", -1)) != len(rows) or [
        row.get("proposal_index") for row in rows
    ] != list(range(len(rows))):
        raise ValueError("Imported checkpoint proposal indices are not contiguous")
    completed = {int(row["proposal_index"]): row["result"] for row in rows}
    if not any(result.get("pass") is True for result in completed.values()):
        raise ValueError("Imported positive checkpoint contains no success")
    return completed


def summarize_exhaustive_negative_checkpoint(
    checkpoint: dict[str, Any],
    *,
    candidate_id: str,
    goal: str,
    root_state_sha256: str,
    proposal_count: int,
) -> dict[str, Any]:
    if (
        checkpoint.get("candidate_id") != candidate_id
        or checkpoint.get("goal") != goal
        or checkpoint.get("root_state_sha256") != root_state_sha256
    ):
        raise ValueError("Negative checkpoint identity changed")
    rows = checkpoint.get("results")
    if not isinstance(rows, list) or len(rows) != proposal_count:
        raise ValueError("Negative checkpoint is not exhaustive")
    if int(checkpoint.get("result_count", -1)) != proposal_count or [
        row.get("proposal_index") for row in rows
    ] != list(range(proposal_count)):
        raise ValueError("Negative checkpoint proposal indices are not contiguous")
    results = [row["result"] for row in rows]
    if any(result.get("pass") is not False for result in results):
        raise ValueError("Negative checkpoint contains a passing proposal")
    if any(
        result.get("wrong_goal_ever_achieved") is not False
        or result.get("unexpected_done_before_goal") is not False
        for result in results
    ):
        raise ValueError("Negative checkpoint has a wrong-goal or terminal event")
    normalized = {result.get("normalized_state_sha256") for result in results}
    normalization_actions = {
        result.get("normalization_action_sha256") for result in results
    }
    if len(normalized) != 1 or len(normalization_actions) != 1:
        raise ValueError("Negative checkpoint did not use one normalized root")
    return {
        "status": "exhaustive_negative_evidence",
        "proposal_count": proposal_count,
        "successful_proposal_count": 0,
        "proposal_indices": list(range(proposal_count)),
        "normalized_state_sha256": next(iter(normalized)),
        "normalization_action_sha256": next(iter(normalization_actions)),
        "checkpoint_status": checkpoint.get("status"),
        "checkpoint_result_count": int(checkpoint["result_count"]),
    }


def oracle_pair_comparability(
    near: dict[str, Any], low: dict[str, Any]
) -> dict[str, Any]:
    by_goal = {}
    for goal in ("drawer", "cabinet"):
        near_oracle = near["oracles"][goal]
        low_oracle = low["oracles"][goal]
        same_bank = near_oracle.get("proposal_bank_sha256") == low_oracle.get(
            "proposal_bank_sha256"
        )
        same_execution = near_oracle.get(
            "proposal_execution_contract_sha256"
        ) == low_oracle.get("proposal_execution_contract_sha256")
        by_goal[goal] = {
            "estimable": bool(same_bank and same_execution),
            "same_proposal_bank": bool(same_bank),
            "same_execution_contract": bool(same_execution),
            "near_execution_mode": near_oracle.get("proposal_execution_mode"),
            "low_execution_mode": low_oracle.get("proposal_execution_mode"),
        }
    return {
        "by_goal": by_goal,
        "all_goals_estimable": all(item["estimable"] for item in by_goal.values()),
    }
