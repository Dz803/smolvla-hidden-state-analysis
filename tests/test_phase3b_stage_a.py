from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

import smolvla_analysis.phase3b_libero as phase3b_libero
from scripts.run_phase3b_stage_a import PROJECT, _load_config, _oracle_checkpoint
from smolvla_analysis.libero_state import LiberoStateSnapshot
from smolvla_analysis.phase3b_libero import (
    DemoTrace,
    ActionPhaseProposal,
    PolicyFreeController,
    PreparedOracleRoot,
    _validate_grasped_transport_phase,
    action_phase_suffix,
    build_landmark_registered_action_phase_proposal_bank,
    grasped_root_recovery_plan,
    grasped_root_transit_plan,
    registered_root_execution_anchor,
    run_goal_oracle_bank,
)
from smolvla_analysis.phase3b_proposal_analysis import (
    factorial_scalar_decomposition,
    factorial_proposal_decomposition,
    support_set_transitions,
    validate_proposal_coverage,
)
from smolvla_analysis.phase3b_stage_a import (
    FACTOR_LEVELS,
    StageACandidateSpec,
    build_selection_lock,
    canonical_sha256,
    candidate_spec,
    goal_distances,
    iter_candidate_specs,
    joint_support_distance,
    measure_joint_support,
    recovery_balanced_goal_axis_point,
    rotate_point_about_axis,
    snapshot_sha256,
    validate_selection_lock,
    validate_stage_a_records,
)


def test_registered_phase_bank_preserves_reference_bowl_offset(monkeypatch) -> None:
    source = DemoTrace(
        goal="cabinet",
        episode_index=474,
        task_index=18,
        frame_indices=np.asarray([1]),
        states=np.zeros((1, 8), dtype=np.float32),
        actions=np.zeros((1, 7), dtype=np.float32),
        action_sha256="a" * 64,
    )
    reference_anchor = np.asarray([0.2, -0.1, 1.1])
    reference_orientation = np.eye(3)
    reference = ActionPhaseProposal(
        source=source,
        suffix=source,
        anchor_position=reference_anchor,
        anchor_orientation=reference_orientation,
        metadata={
            "proposal_index": 0,
            "layout": "A",
            "anchor_eef_position": reference_anchor.tolist(),
        },
    )
    landmarks = {
        0: np.asarray([-0.08, 0.01, 0.90]),
        1: np.asarray([-0.10, -0.02, 0.90]),
    }

    class Controller:
        def __init__(self, environment):
            self.layout = 0

        def reset_layout(self, init_state_id, seed):
            self.layout = init_state_id

        def bowl_position(self):
            return landmarks[self.layout].copy()

    monkeypatch.setattr(
        phase3b_libero,
        "build_action_phase_proposal_bank",
        lambda *args, **kwargs: (reference,),
    )
    monkeypatch.setattr(phase3b_libero, "PolicyFreeController", Controller)
    config = {
        "environment": {"reset_seed": 0},
        "action_phase_oracle": {
            "execution_mode": "action_intrinsic_pregrasp_bowl_registered_v1",
            "anchor_rule": (
                "canonical_layout_a_pregrasp_anchor_translated_by_bowl_landmark"
            ),
            "registration": {
                "type": "translation_only",
                "reference_layout": "A",
                "landmark": "akita_black_bowl_1",
                "target_landmark_tolerance_m": 1e-9,
            },
        },
    }
    (registered,) = build_landmark_registered_action_phase_proposal_bank(
        object(), target_layout="B", proposals=(source,), config=config
    )
    np.testing.assert_allclose(
        registered.anchor_position - landmarks[1],
        reference_anchor - landmarks[0],
    )
    np.testing.assert_array_equal(
        registered.anchor_orientation, reference_orientation
    )
    assert registered.metadata["layout"] == "B"
    assert registered.metadata["landmark_registration"]["reference_layout"] == "A"
    assert registered.metadata["landmark_registration"]["target_layout"] == "B"


def _record(spec: StageACandidateSpec) -> dict:
    def oracle(goal: str) -> dict:
        proposal_bank = [
            {
                "proposal_index": 0,
                "goal": goal,
                "episode_index": 1,
                "task_index": 2,
                "frame_count": 100,
                "action_sha256": "f" * 64,
            }
        ]
        cost = {
            "budgeted_action_steps": 200,
            "executed_action_steps": 190,
            "active_servo_steps": 50,
            "demonstration_action_steps": 100,
            "executed_demonstration_action_steps": 90,
            "eef_path_length_m": 1.0,
            "control_effort": 2.0,
            "motion_control_effort": 0.5,
        }
        execution_mode = "action_intrinsic_pregrasp_phase_continuation_v2"
        phase_proposal = {
            "proposal_index": 0,
            "episode_index": 1,
            "task_index": 2,
            "source_action_sha256": "f" * 64,
        }
        execution_contract = [phase_proposal]
        normalization_preparation = {
            "execution_mode": "normalization_only",
            "source_proposal_replayed": False,
            "executed_action_steps": 100,
            "action_sha256": "2" * 64,
        }
        total_environment_action_steps = 190
        attempt = {
            "proposal_index": 0,
            "episode_index": 1,
            "task_index": 2,
            "action_sha256": "f" * 64,
            "proposal_execution_mode": execution_mode,
            "pass": True,
            "normalized_state_sha256": "1" * 64,
            "normalization_action_sha256": "2" * 64,
            "cost": cost,
        }
        if phase_proposal is not None:
            attempt["phase_proposal"] = phase_proposal
            attempt["action_phase_bridge"] = {"pass": True}
        return {
            "pass": True,
            "goal_ever_achieved": True,
            "demo_episode_index": 1,
            "demo_task_index": 2,
            "demo_action_sha256": "f" * 64,
            "proposal_bank_sha256": canonical_sha256(proposal_bank),
            "proposal_bank": proposal_bank,
            "proposal_execution_mode": execution_mode,
            "proposal_execution_contract": execution_contract,
            "proposal_execution_contract_sha256": canonical_sha256(
                execution_contract
            ),
            "proposal_attempts": [attempt],
            "proposal_attempt_count": 1,
            "proposal_success_count": 1,
            "proposal_success_fraction": 1.0,
            "successful_proposal_indices": [0],
            "selected_proposal_index": 0,
            "proposal_selection_rule": (
                "minimum_executed_steps_then_path_effort_index"
            ),
            "shared_normalized_state_sha256": "1" * 64,
            "shared_normalization_action_sha256": "2" * 64,
            "shared_normalization_action_steps": 100,
            "shared_normalization_active_servo_steps": 50,
            "total_attempted_action_steps": 190,
            "counterfactual_full_attempt_action_steps": 190,
            "normalization_preparation": normalization_preparation,
            "total_environment_action_steps": total_environment_action_steps,
            "cost": cost,
        }

    return {
        "candidate_id": spec.candidate_id,
        "factors": spec.as_dict(),
        "policy_loaded": False,
        "state_sha256": "a" * 64,
        "root_validation": {
            "pass": True,
            "goals": {"drawer": False, "cabinet": False},
        },
        "root_geometry": {
            "planned_goal_distances_m": {"drawer": 0.5, "cabinet": 0.4},
            "realized_goal_distances_m": {"drawer": 0.5, "cabinet": 0.4},
            "planned_recovery_distance_m": 0.3,
            "realized_recovery_distance_m": 0.3,
        },
        "certificate": {"pass": True},
        "support_measurement": {
            "pass": True,
            "reference_bank_sha256": "b" * 64,
            "reference_count": 10,
            "nearest": {"distance": 1.0},
            "factor_category_matches": {
                "drawer_aperture": True,
                "possession": True,
                "transit_locus": True,
            },
        },
        "oracles": {goal: oracle(goal) for goal in ("drawer", "cabinet")},
    }


def test_exact_32_cell_lattice_and_ids() -> None:
    specs = iter_candidate_specs()
    assert len(specs) == 32
    assert len({spec.candidate_id for spec in specs}) == 32
    for factor, levels in FACTOR_LEVELS.items():
        assert {getattr(spec, factor) for spec in specs} == set(levels)
    assert candidate_spec(specs[7].candidate_id) == specs[7]
    assert len({spec.family_id for spec in specs}) == 8
    assert len({spec.support_pair_id for spec in specs}) == 16


def _synthetic_proposal_coverage() -> pd.DataFrame:
    rows = []
    for spec in iter_candidate_specs():
        for goal in ("drawer", "cabinet"):
            outcomes = (
                True,
                spec.support_stratum == "transverse_low_support",
                spec.support_stratum == "demonstration_near",
            )
            for proposal_index, passed in enumerate(outcomes):
                rows.append(
                    {
                        "candidate_id": spec.candidate_id,
                        **{
                            factor: getattr(spec, factor)
                            for factor in FACTOR_LEVELS
                        },
                        "goal": goal,
                        "proposal_execution_mode": (
                            "action_intrinsic_pregrasp_phase_continuation_v2"
                        ),
                        "proposal_index": proposal_index,
                        "episode_index": 100 + proposal_index,
                        "pass": passed,
                    }
                )
    return pd.DataFrame(rows)


def test_proposal_surface_decomposition_is_exact() -> None:
    coverage = validate_proposal_coverage(_synthetic_proposal_coverage())
    components, summary = factorial_proposal_decomposition(coverage)
    for goal in ("drawer", "cabinet"):
        result = summary[goal]
        fractions = (
            result["proposal_generality_variance_fraction"]
            + result["common_state_variance_fraction"]
            + result["state_proposal_interaction_variance_fraction"]
        )
        assert fractions == pytest.approx(1.0, abs=1e-12)
        assert result["variance_reconstruction_error"] == pytest.approx(
            0.0, abs=1e-12
        )
        support = components[
            (components["goal"] == goal)
            & (components["term"] == "support_stratum")
        ].iloc[0]
        assert support["common_state_variance"] == pytest.approx(0.0)
        assert support["state_proposal_interaction_variance"] > 0.0


def test_scalar_factorial_decomposition_localizes_exact_support_effect() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": [spec.candidate_id for spec in iter_candidate_specs()],
            "value": [
                1.0
                if spec.support_stratum == "transverse_low_support"
                else -1.0
                for spec in iter_candidate_specs()
            ],
        }
    )
    components, summary = factorial_scalar_decomposition(
        frame, value_column="value"
    )
    support = components[components["term"] == "support_stratum"].iloc[0]
    assert summary["mean"] == pytest.approx(0.0)
    assert summary["variance_reconstruction_error"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert support["coefficient"] == pytest.approx(1.0)
    assert support["variance_fraction"] == pytest.approx(1.0)
    assert components.loc[
        components["term"] != "support_stratum", "component_variance"
    ].max() == pytest.approx(0.0)


def test_support_set_transitions_preserve_gained_and_lost_identity() -> None:
    transitions, summary = support_set_transitions(
        _synthetic_proposal_coverage()
    )
    assert np.allclose(transitions["success_set_jaccard"], 1.0 / 3.0)
    assert set(transitions["gained_success_count"]) == {1}
    assert set(transitions["lost_success_count"]) == {1}
    for goal in ("drawer", "cabinet"):
        assert summary[goal]["disjoint_success_set_count"] == 0
        assert summary[goal]["identical_success_set_count"] == 0


def test_proposal_coverage_rejects_mixed_execution_modes() -> None:
    coverage = _synthetic_proposal_coverage()
    drawer_index = coverage.index[coverage["goal"] == "drawer"][0]
    coverage.loc[drawer_index, "proposal_execution_mode"] = "another_mode"
    with pytest.raises(ValueError, match="mixes execution modes"):
        validate_proposal_coverage(coverage)


def test_candidate_spec_rejects_unknown_factor() -> None:
    with pytest.raises(ValueError, match="drawer_aperture"):
        StageACandidateSpec(
            "half_open", "on_table", "drawer_side", "demonstration_near", "A"
        )
    with pytest.raises(ValueError, match="Unknown Stage A"):
        candidate_spec("not-a-candidate")


def test_selection_lock_is_exact_and_contains_no_policy_outcome() -> None:
    lock = build_selection_lock(contract_sha256="c" * 64, construction_revision="r1")
    validate_selection_lock(
        lock, contract_sha256="c" * 64, construction_revision="r1"
    )
    assert lock["policy_outcomes_used"] is False
    changed = deepcopy(lock)
    changed["candidate_ids"] = changed["candidate_ids"][:-1]
    with pytest.raises(ValueError, match="Selection lock"):
        validate_selection_lock(
            changed, contract_sha256="c" * 64, construction_revision="r1"
        )


def test_canonical_hash_rejects_nonfinite_scientific_values() -> None:
    with pytest.raises(ValueError, match="JSON compliant"):
        canonical_sha256({"invalid": np.nan})


def test_transverse_rotation_preserves_both_goal_distances() -> None:
    near = np.asarray([0.18, -0.08, 1.20])
    drawer = np.asarray([0.03, -0.10, 1.09])
    cabinet = np.asarray([0.03, -0.25, 1.13])
    transverse = rotate_point_about_axis(near, drawer, cabinet, np.pi / 2.0)
    assert not np.allclose(near, transverse)
    assert goal_distances(
        near, drawer_goal=drawer, cabinet_goal=cabinet
    ) == pytest.approx(
        goal_distances(transverse, drawer_goal=drawer, cabinet_goal=cabinet),
        abs=1e-12,
    )
    with pytest.raises(ValueError, match="degenerate"):
        rotate_point_about_axis(near, drawer, drawer, np.pi / 2.0)


def test_recovery_balanced_rotation_uses_largest_admissible_angle() -> None:
    scripted = np.asarray([0.04235, -0.03241, 1.18893])
    drawer = np.asarray([0.03629, -0.08727, 1.09063])
    cabinet = np.asarray([0.03603, -0.24969, 1.12652])
    recovery = np.asarray([-0.20201, -0.00851, 1.17600])
    result = recovery_balanced_goal_axis_point(
        scripted,
        drawer,
        cabinet,
        recovery,
        maximum_angle_degrees=20.0,
        target_recovery_mismatch=0.08,
    )
    assert 0.0 < result["selected_angle_degrees"] < 20.0
    assert result["planned_recovery_mismatch"] == pytest.approx(0.08, abs=1e-12)
    assert result["pair_separation_m"] > 0.015
    rotated = result["point"]
    assert goal_distances(
        scripted, drawer_goal=drawer, cabinet_goal=cabinet
    ) == pytest.approx(
        goal_distances(rotated, drawer_goal=drawer, cabinet_goal=cabinet),
        abs=1e-12,
    )


def test_grasped_root_transit_plan_uses_fixed_clearance_route() -> None:
    start = np.asarray([-0.12, 0.07, 1.00])
    target = np.asarray([0.02, -0.11, 1.29])
    bounds = {
        "x": [-0.35, 0.35],
        "y": [-0.38, 0.25],
        "z": [0.94, 1.36],
    }
    budgets = {
        "clearance_lift": 150,
        "clearance_transit": 80,
        "target_descent": 50,
    }
    plan = grasped_root_transit_plan(
        start,
        target,
        clearance_margin_m=0.02,
        workspace_bounds=bounds,
        phase_budgets=budgets,
    )
    assert [phase["phase"] for phase in plan] == [
        "clearance_lift",
        "clearance_transit",
        "target_descent",
    ]
    assert [phase["budget"] for phase in plan] == [150, 80, 50]
    np.testing.assert_allclose(plan[0]["target_position"], [-0.12, 0.07, 1.31])
    np.testing.assert_allclose(plan[1]["target_position"], [0.02, -0.11, 1.31])
    np.testing.assert_allclose(plan[2]["target_position"], target)

    capped = grasped_root_transit_plan(
        start,
        np.asarray([0.02, -0.11, 1.35]),
        clearance_margin_m=0.02,
        workspace_bounds=bounds,
        phase_budgets=budgets,
    )
    assert capped[0]["target_position"][2] == pytest.approx(1.36)
    with pytest.raises(ValueError, match="phase budgets"):
        grasped_root_transit_plan(
            start,
            target,
            clearance_margin_m=0.02,
            workspace_bounds=bounds,
            phase_budgets={"clearance_lift": 280},
        )


def test_action_phase_suffix_uses_first_action_intrinsic_close_transition() -> None:
    actions = np.zeros((100, 7), dtype=np.float32)
    actions[70:, 6] = 1.0
    demo = DemoTrace(
        goal="drawer",
        episode_index=7,
        task_index=12,
        frame_indices=np.arange(100, dtype=np.int64),
        states=np.zeros((100, 8), dtype=np.float32),
        actions=actions,
        action_sha256="a" * 64,
    )
    suffix, metadata = action_phase_suffix(
        demo,
        maximum_pregrasp_lead_frames=50,
        minimum_anchor_prefix_frames=1,
        gripper_close_threshold=0.25,
    )
    assert metadata["first_gripper_close_index"] == 70
    assert metadata["suffix_start_index"] == 20
    assert metadata["anchor_after_frame"] == 19
    assert len(suffix.actions) == 80
    np.testing.assert_array_equal(suffix.frame_indices, np.arange(20, 100))
    assert suffix.action_sha256 == metadata["suffix_action_sha256"]

    early_actions = np.zeros((60, 7), dtype=np.float32)
    early_actions[28:, 6] = 1.0
    early_demo = DemoTrace(
        goal="cabinet",
        episode_index=8,
        task_index=18,
        frame_indices=np.arange(60, dtype=np.int64),
        states=np.zeros((60, 8), dtype=np.float32),
        actions=early_actions,
        action_sha256="b" * 64,
    )
    early_suffix, early_metadata = action_phase_suffix(
        early_demo,
        maximum_pregrasp_lead_frames=50,
        minimum_anchor_prefix_frames=1,
        gripper_close_threshold=0.25,
    )
    assert early_metadata["suffix_start_index"] == 1
    assert early_metadata["realized_pregrasp_lead_frames"] == 27
    assert len(early_suffix.actions) == 59

    without_transition = deepcopy(demo)
    object.__setattr__(
        without_transition, "actions", np.zeros((100, 7), dtype=np.float32)
    )
    with pytest.raises(ValueError, match="no gripper-close transition"):
        action_phase_suffix(
            without_transition,
            maximum_pregrasp_lead_frames=50,
            minimum_anchor_prefix_frames=1,
            gripper_close_threshold=0.25,
        )


def test_snapshot_hash_includes_full_runtime_state() -> None:
    first = LiberoStateSnapshot(
        mujoco_state=np.asarray([1.0, 2.0]),
        objects={},
        goal_predicates=(),
        contacts=(),
        grasped_objects=(),
        success=False,
        runtime_state={"sim_data": {"qacc_warmstart": {"shape": [1], "values": [0.1]}}},
    )
    second = LiberoStateSnapshot(
        mujoco_state=np.asarray([1.0, 2.0]),
        objects={},
        goal_predicates=(),
        contacts=(),
        grasped_objects=(),
        success=False,
        runtime_state={"sim_data": {"qacc_warmstart": {"shape": [1], "values": [0.2]}}},
    )
    assert snapshot_sha256(first) != snapshot_sha256(second)


def test_stage_a_record_gate_fails_closed() -> None:
    records = [_record(spec) for spec in iter_candidate_specs()]
    report = validate_stage_a_records(records)
    assert report == {
        "pass": True,
        "candidate_count": 32,
        "support_pair_count": 16,
        "max_planned_goal_distance_mismatch": 0.0,
        "max_realized_goal_distance_mismatch": 0.0,
        "max_planned_recovery_distance_mismatch": 0.0,
        "max_realized_recovery_distance_mismatch": 0.0,
        "max_budgeted_oracle_cost_mismatch": 0.0,
        "max_executed_oracle_step_mismatch": 0.0,
        "max_active_oracle_step_mismatch": 0.0,
        "max_oracle_eef_path_mismatch": 0.0,
        "max_oracle_motion_control_effort_mismatch": 0.0,
        "support_reference_bank_sha256": "b" * 64,
        "oracle_proposal_bank_sha256_by_goal": {
            "drawer": canonical_sha256(
                records[0]["oracles"]["drawer"]["proposal_bank"]
            ),
            "cabinet": canonical_sha256(
                records[0]["oracles"]["cabinet"]["proposal_bank"]
            ),
        },
        "oracle_proposal_execution_mode_by_goal": {
            "drawer": "action_intrinsic_pregrasp_phase_continuation_v2",
            "cabinet": "action_intrinsic_pregrasp_phase_continuation_v2",
        },
        "oracle_proposal_execution_contract_sha256_by_goal": {
            "drawer": [
                records[0]["oracles"]["drawer"][
                    "proposal_execution_contract_sha256"
                ]
            ],
            "cabinet": [
                records[0]["oracles"]["cabinet"][
                    "proposal_execution_contract_sha256"
                ]
            ],
        },
        "proposal_nonfirst_selection_fraction_by_goal": {
            "drawer": 0.0,
            "cabinet": 0.0,
        },
        "mean_proposal_success_fraction_by_goal": {
            "drawer": 1.0,
            "cabinet": 1.0,
        },
        "minimum_proposal_success_fraction_by_goal": {
            "drawer": 1.0,
            "cabinet": 1.0,
        },
        "support_pair_same_selected_proposal_fraction_by_goal": {
            "drawer": 1.0,
            "cabinet": 1.0,
        },
        "support_pair_minimum_shared_success_count_by_goal": {
            "drawer": 1,
            "cabinet": 1,
        },
        "support_pair_mean_success_set_jaccard_by_goal": {
            "drawer": 1.0,
            "cabinet": 1.0,
        },
        "support_pair_positive_direction_fraction": 0.0,
        "median_joint_support_distance_difference": 0.0,
        "policy_loaded_count": 0,
    }

    policy_violation = deepcopy(records)
    policy_violation[0]["policy_loaded"] = True
    with pytest.raises(ValueError, match="policy boundary"):
        validate_stage_a_records(policy_violation)

    failed_oracle = deepcopy(records)
    failed_oracle[0]["oracles"]["drawer"]["pass"] = False
    with pytest.raises(ValueError, match="drawer oracle"):
        validate_stage_a_records(failed_oracle)

    unbalanced = deepcopy(records)
    low_index = next(
        index
        for index, spec in enumerate(iter_candidate_specs())
        if spec.support_stratum == "transverse_low_support"
        and spec.support_pair_id == iter_candidate_specs()[0].support_pair_id
    )
    unbalanced[low_index]["oracles"]["cabinet"]["cost"][
        "budgeted_action_steps"
    ] = 300
    with pytest.raises(ValueError, match="oracle-cost"):
        validate_stage_a_records(unbalanced)

    hidden_work_imbalance = deepcopy(records)
    hidden_work_imbalance[low_index]["oracles"]["cabinet"]["cost"][
        "eef_path_length_m"
    ] = 1.5
    with pytest.raises(ValueError, match="eef_path_length_m"):
        validate_stage_a_records(hidden_work_imbalance)

    padded_effort_imbalance = deepcopy(records)
    padded_effort_imbalance[low_index]["oracles"]["cabinet"]["cost"][
        "motion_control_effort"
    ] = 0.8
    with pytest.raises(ValueError, match="motion_control_effort"):
        validate_stage_a_records(padded_effort_imbalance)

    planned_recovery_imbalance = deepcopy(records)
    planned_recovery_imbalance[low_index]["root_geometry"][
        "planned_recovery_distance_m"
    ] = 0.5
    with pytest.raises(ValueError, match="planned recovery-distance"):
        validate_stage_a_records(planned_recovery_imbalance)

    nonfinite_recovery = deepcopy(records)
    nonfinite_recovery[low_index]["root_geometry"][
        "planned_recovery_distance_m"
    ] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        validate_stage_a_records(nonfinite_recovery)

    recovery_diagnostic = deepcopy(records)
    recovery_diagnostic[low_index]["root_geometry"][
        "realized_recovery_distance_m"
    ] = 0.5
    diagnostic_report = validate_stage_a_records(recovery_diagnostic)
    assert diagnostic_report["max_realized_recovery_distance_mismatch"] > 0.0


def test_proposal_fallback_is_reported_and_bank_hash_must_match() -> None:
    records = [_record(spec) for spec in iter_candidate_specs()]
    low_index = next(
        index
        for index, spec in enumerate(iter_candidate_specs())
        if spec.support_stratum == "transverse_low_support"
    )
    second_proposal = {
        "proposal_index": 1,
        "goal": "drawer",
        "episode_index": 2,
        "task_index": 2,
        "frame_count": 100,
        "action_sha256": "9" * 64,
    }
    for record in records:
        drawer_oracle = record["oracles"]["drawer"]
        drawer_oracle["proposal_bank"].append(deepcopy(second_proposal))
        second_cost = deepcopy(drawer_oracle["cost"])
        second_cost["executed_action_steps"] = 195
        second_cost["executed_demonstration_action_steps"] = 95
        second_cost["eef_path_length_m"] = 1.1
        second_attempt = {
            "proposal_index": 1,
            "episode_index": 2,
            "task_index": 2,
            "action_sha256": "9" * 64,
            "proposal_execution_mode": (
                "action_intrinsic_pregrasp_phase_continuation_v2"
            ),
            "pass": True,
            "normalized_state_sha256": "1" * 64,
            "normalization_action_sha256": "2" * 64,
            "cost": second_cost,
        }
        second_phase_proposal = {
            "proposal_index": 1,
            "episode_index": 2,
            "task_index": 2,
            "source_action_sha256": "9" * 64,
        }
        second_attempt["phase_proposal"] = second_phase_proposal
        second_attempt["action_phase_bridge"] = {"pass": True}
        drawer_oracle["proposal_attempts"].append(second_attempt)
        drawer_oracle["proposal_execution_contract"].append(
            second_phase_proposal
        )
        drawer_oracle["proposal_execution_contract_sha256"] = canonical_sha256(
            drawer_oracle["proposal_execution_contract"]
        )
        drawer_oracle["proposal_bank_sha256"] = canonical_sha256(
            drawer_oracle["proposal_bank"]
        )
        drawer_oracle["proposal_attempt_count"] = 2
        drawer_oracle["proposal_success_count"] = 2
        drawer_oracle["proposal_success_fraction"] = 1.0
        drawer_oracle["successful_proposal_indices"] = [0, 1]
        drawer_oracle["total_attempted_action_steps"] = 285
        drawer_oracle["counterfactual_full_attempt_action_steps"] = 385
        drawer_oracle["total_environment_action_steps"] = 285

    drawer_oracle = records[low_index]["oracles"]["drawer"]
    drawer_oracle["proposal_attempts"][0]["pass"] = False
    drawer_oracle["proposal_success_count"] = 1
    drawer_oracle["proposal_success_fraction"] = 0.5
    drawer_oracle["successful_proposal_indices"] = [1]
    drawer_oracle["selected_proposal_index"] = 1
    drawer_oracle["demo_episode_index"] = 2
    drawer_oracle["demo_action_sha256"] = "9" * 64
    drawer_oracle["cost"] = drawer_oracle["proposal_attempts"][1]["cost"]
    report = validate_stage_a_records(records)
    assert report["proposal_nonfirst_selection_fraction_by_goal"][
        "drawer"
    ] == pytest.approx(1.0 / 32.0)
    assert report["minimum_proposal_success_fraction_by_goal"]["drawer"] == 0.5
    assert report[
        "support_pair_same_selected_proposal_fraction_by_goal"
    ]["drawer"] == pytest.approx(15.0 / 16.0)

    disjoint = deepcopy(records)
    low_spec = iter_candidate_specs()[low_index]
    near_index = next(
        index
        for index, spec in enumerate(iter_candidate_specs())
        if spec.support_pair_id == low_spec.support_pair_id
        and spec.support_stratum == "demonstration_near"
    )
    near_oracle = disjoint[near_index]["oracles"]["drawer"]
    near_oracle["proposal_attempts"][1]["pass"] = False
    near_oracle["proposal_success_count"] = 1
    near_oracle["proposal_success_fraction"] = 0.5
    near_oracle["successful_proposal_indices"] = [0]
    disjoint_report = validate_stage_a_records(disjoint)
    assert disjoint_report[
        "support_pair_minimum_shared_success_count_by_goal"
    ]["drawer"] == 0

    changed_bank = deepcopy(records)
    changed_bank[low_index]["oracles"]["drawer"][
        "proposal_bank_sha256"
    ] = "8" * 64
    with pytest.raises(ValueError, match="proposal-bank hash"):
        validate_stage_a_records(changed_bank)

    invalid_ledger = deepcopy(records)
    invalid_ledger[low_index]["oracles"]["drawer"]["proposal_attempts"][0][
        "pass"
    ] = True
    with pytest.raises(ValueError, match="proposal selection"):
        validate_stage_a_records(invalid_ledger)


def test_proposal_bank_rejects_different_normalized_roots(monkeypatch) -> None:
    proposals = tuple(
        DemoTrace(
            goal="drawer",
            episode_index=episode_index,
            task_index=12,
            frame_indices=np.asarray([0]),
            states=np.zeros((1, 8), dtype=np.float32),
            actions=np.zeros((1, 7), dtype=np.float32),
            action_sha256=str(episode_index) * 64,
        )
        for episode_index in (1, 2)
    )
    prepared = PreparedOracleRoot(
        snapshot=object(),
        phases={},
        actions=(np.zeros(7, dtype=np.float32),),
        eef_path_length_m=0.0,
        control_effort=0.0,
        motion_control_effort=0.0,
        active_servo_steps=0,
        done_count=0,
        normalization_goal_ever=False,
        normalization_done_ever=False,
        normalized_goals={"drawer": False, "cabinet": False},
        normalized_bowl_position_error_m=0.0,
    )

    normalization_hash = phase3b_libero._action_sha256(prepared.actions)
    calls = 0

    def result(call_count: int, demo: DemoTrace):
        passed = call_count == 2
        payload = {
            "goal": "drawer",
            "demo_episode_index": demo.episode_index,
            "demo_task_index": demo.task_index,
            "demo_action_sha256": demo.action_sha256,
            "pass": passed,
            "goal_ever_achieved": passed,
            "first_goal_demo_frame": 0 if passed else None,
            "wrong_goal_ever_achieved": False,
            "unexpected_done_before_goal": False,
            "normalized_bowl_position_error_m": 0.0,
            "normalized_state_sha256": str(call_count) * 64,
            "normalization_action_sha256": normalization_hash,
            "final_goals": {"drawer": passed, "cabinet": False},
            "cost": {
                "executed_action_steps": 10,
                "active_servo_steps": 2,
                "executed_demonstration_action_steps": 1,
                "eef_path_length_m": 1.0,
                "motion_control_effort": 1.0,
            },
        }
        return payload

    monkeypatch.setattr(
        phase3b_libero,
        "run_goal_oracle",
        lambda *args, **kwargs: {
            "source_proposal_replayed": False,
            "normalization_action_steps": 1,
            "normalization_action_sha256": normalization_hash,
            "_prepared_oracle_root": prepared,
        },
    )

    def replay_result(*args, demo, **kwargs):
        nonlocal calls
        calls += 1
        return result(calls, demo)

    monkeypatch.setattr(
        phase3b_libero,
        "run_goal_oracle_from_prepared_root",
        replay_result,
    )
    monkeypatch.setattr(
        phase3b_libero, "snapshot_sha256", lambda *args, **kwargs: "1" * 64
    )
    monkeypatch.setattr(
        phase3b_libero, "restore_libero_state", lambda *args, **kwargs: None
    )
    with pytest.raises(RuntimeError, match="did not share one normalized root"):
        run_goal_oracle_bank(
            object(),
            object(),
            spec=iter_candidate_specs()[0],
            goal="drawer",
            proposals=proposals,
            initial_bowl_position=np.zeros(3),
            initial_eef_position=np.zeros(3),
            initial_eef_orientation=np.eye(3),
            initial_joint_positions=np.zeros(7),
            recovery_waypoints=None,
            config={},
        )


def test_proposal_bank_evaluates_all_and_selects_minimum_cost(monkeypatch) -> None:
    proposals = tuple(
        DemoTrace(
            goal="drawer",
            episode_index=episode_index,
            task_index=12,
            frame_indices=np.asarray([0]),
            states=np.zeros((1, 8), dtype=np.float32),
            actions=np.zeros((1, 7), dtype=np.float32),
            action_sha256=str(episode_index) * 64,
        )
        for episode_index in (1, 2)
    )
    prepared = PreparedOracleRoot(
        snapshot=object(),
        phases={},
        actions=(np.zeros(7, dtype=np.float32),),
        eef_path_length_m=0.0,
        control_effort=0.0,
        motion_control_effort=0.0,
        active_servo_steps=0,
        done_count=0,
        normalization_goal_ever=False,
        normalization_done_ever=False,
        normalized_goals={"drawer": False, "cabinet": False},
        normalized_bowl_position_error_m=0.0,
    )
    calls = []

    normalization_hash = phase3b_libero._action_sha256(prepared.actions)

    def result(executed_steps: int, demo: DemoTrace):
        calls.append(executed_steps)
        payload = {
            "goal": "drawer",
            "demo_episode_index": demo.episode_index,
            "demo_task_index": demo.task_index,
            "demo_action_sha256": demo.action_sha256,
            "pass": True,
            "goal_ever_achieved": True,
            "first_goal_demo_frame": 0,
            "wrong_goal_ever_achieved": False,
            "unexpected_done_before_goal": False,
            "normalized_bowl_position_error_m": 0.0,
            "normalized_state_sha256": "1" * 64,
            "normalization_action_sha256": normalization_hash,
            "final_goals": {"drawer": True, "cabinet": False},
            "cost": {
                "executed_action_steps": executed_steps,
                "active_servo_steps": 2,
                "executed_demonstration_action_steps": 1,
                "eef_path_length_m": float(executed_steps),
                "motion_control_effort": float(executed_steps),
            },
        }
        return payload

    monkeypatch.setattr(
        phase3b_libero,
        "run_goal_oracle",
        lambda *args, **kwargs: {
            "source_proposal_replayed": False,
            "normalization_action_steps": 1,
            "normalization_action_sha256": normalization_hash,
            "_prepared_oracle_root": prepared,
        },
    )
    monkeypatch.setattr(
        phase3b_libero,
        "run_goal_oracle_from_prepared_root",
        lambda *args, demo, **kwargs: result(
            20 if demo.episode_index == 1 else 10, demo
        ),
    )
    monkeypatch.setattr(
        phase3b_libero, "snapshot_sha256", lambda *args, **kwargs: "1" * 64
    )

    oracle = run_goal_oracle_bank(
        object(),
        object(),
        spec=iter_candidate_specs()[0],
        goal="drawer",
        proposals=proposals,
        initial_bowl_position=np.zeros(3),
        initial_eef_position=np.zeros(3),
        initial_eef_orientation=np.eye(3),
        initial_joint_positions=np.zeros(7),
        recovery_waypoints=None,
        config={},
    )

    assert calls == [20, 10]
    assert oracle["proposal_attempt_count"] == 2
    assert oracle["successful_proposal_indices"] == [0, 1]
    assert oracle["selected_proposal_index"] == 1
    assert oracle["total_attempted_action_steps"] == 3
    assert oracle["counterfactual_full_attempt_action_steps"] == 30


def test_phase_proposal_bank_uses_normalization_only_preparation(
    monkeypatch,
) -> None:
    sources = tuple(
        DemoTrace(
            goal="drawer",
            episode_index=episode_index,
            task_index=12,
            frame_indices=np.asarray([0]),
            states=np.zeros((1, 8), dtype=np.float32),
            actions=np.zeros((1, 7), dtype=np.float32),
            action_sha256=str(episode_index) * 64,
        )
        for episode_index in (1, 2)
    )
    phase_proposals = tuple(
        ActionPhaseProposal(
            source=source,
            suffix=source,
            anchor_position=np.zeros(3),
            anchor_orientation=np.eye(3),
            metadata={
                "proposal_index": index,
                "episode_index": source.episode_index,
                "task_index": source.task_index,
                "source_action_sha256": source.action_sha256,
                "layout": "A",
            },
        )
        for index, source in enumerate(sources)
    )
    prepared = PreparedOracleRoot(
        snapshot=object(),
        phases={},
        actions=(np.zeros(7, dtype=np.float32),),
        eef_path_length_m=0.0,
        control_effort=0.0,
        motion_control_effort=0.0,
        active_servo_steps=0,
        done_count=0,
        normalization_goal_ever=False,
        normalization_done_ever=False,
        normalized_goals={"drawer": False, "cabinet": False},
        normalized_bowl_position_error_m=0.0,
    )

    normalization_hash = phase3b_libero._action_sha256(prepared.actions)

    def preparation_result(*args, **kwargs):
        return {
            "normalized_state_sha256": "1" * 64,
            "normalization_action_sha256": normalization_hash,
            "normalization_action_steps": 1,
            "source_proposal_replayed": False,
            "_prepared_oracle_root": prepared,
        }

    def phase_result(*args, proposal, **kwargs):
        executed_steps = 10 if proposal.source.episode_index == 1 else 9
        proposal_steps = executed_steps - 1
        return {
            "goal": "drawer",
            "demo_episode_index": proposal.source.episode_index,
            "demo_task_index": proposal.source.task_index,
            "demo_action_sha256": proposal.source.action_sha256,
            "pass": True,
            "goal_ever_achieved": True,
            "first_goal_demo_frame": 0,
            "wrong_goal_ever_achieved": False,
            "unexpected_done_before_goal": False,
            "normalized_bowl_position_error_m": 0.0,
            "normalized_state_sha256": "1" * 64,
            "normalization_action_sha256": normalization_hash,
            "final_goals": {"drawer": True, "cabinet": False},
            "phase_proposal": proposal.metadata,
            "phases": {"action_phase_bridge": {"pass": True}},
            "cost": {
                "executed_action_steps": executed_steps,
                "active_servo_steps": 2,
                "executed_demonstration_action_steps": proposal_steps,
                "eef_path_length_m": float(executed_steps),
                "motion_control_effort": float(executed_steps),
            },
        }

    monkeypatch.setattr(phase3b_libero, "run_goal_oracle", preparation_result)
    monkeypatch.setattr(
        phase3b_libero,
        "run_action_phase_oracle_from_prepared_root",
        phase_result,
    )
    monkeypatch.setattr(
        phase3b_libero, "snapshot_sha256", lambda *args, **kwargs: "1" * 64
    )
    oracle = run_goal_oracle_bank(
        object(),
        object(),
        spec=iter_candidate_specs()[0],
        goal="drawer",
        proposals=sources,
        initial_bowl_position=np.zeros(3),
        initial_eef_position=np.zeros(3),
        initial_eef_orientation=np.eye(3),
        initial_joint_positions=np.zeros(7),
        recovery_waypoints=None,
        config={
            "action_phase_oracle": {
                "execution_mode": (
                    "action_intrinsic_pregrasp_phase_continuation_v2"
                )
            }
        },
        action_phase_proposals=phase_proposals,
    )
    assert oracle["selected_proposal_index"] == 1
    assert oracle["proposal_attempt_count"] == 2
    assert oracle["total_attempted_action_steps"] == 18
    assert oracle["total_environment_action_steps"] == 18
    assert oracle["normalization_preparation"] == {
        "execution_mode": "normalization_only",
        "source_proposal_replayed": False,
        "executed_action_steps": 1,
        "action_sha256": normalization_hash,
    }


def test_oracle_checkpoint_is_atomic_resumable_and_contract_bound(
    tmp_path,
) -> None:
    source = DemoTrace(
        goal="drawer",
        episode_index=1,
        task_index=12,
        frame_indices=np.asarray([0]),
        states=np.zeros((1, 8), dtype=np.float32),
        actions=np.zeros((1, 7), dtype=np.float32),
        action_sha256="1" * 64,
    )
    phase = ActionPhaseProposal(
        source=source,
        suffix=source,
        anchor_position=np.zeros(3),
        anchor_orientation=np.eye(3),
        metadata={"proposal_index": 0, "layout": "A"},
    )
    (tmp_path / "checkpoints").mkdir()
    completed, record, finish = _oracle_checkpoint(
        tmp_path,
        candidate_id="candidate",
        goal="drawer",
        root_state_sha256="a" * 64,
        contract_sha256="b" * 64,
        selection_lock_sha256="c" * 64,
        proposals=(source,),
        phase_proposals=(phase,),
    )
    assert completed == {}
    result = {"pass": True, "cost": {"executed_action_steps": 1}}
    record(0, result)
    finish({"pass": True})

    resumed, _, _ = _oracle_checkpoint(
        tmp_path,
        candidate_id="candidate",
        goal="drawer",
        root_state_sha256="a" * 64,
        contract_sha256="b" * 64,
        selection_lock_sha256="c" * 64,
        proposals=(source,),
        phase_proposals=(phase,),
    )
    assert resumed == {0: result}
    with pytest.raises(ValueError, match="root_state_sha256"):
        _oracle_checkpoint(
            tmp_path,
            candidate_id="candidate",
            goal="drawer",
            root_state_sha256="d" * 64,
            contract_sha256="b" * 64,
            selection_lock_sha256="c" * 64,
            proposals=(source,),
            phase_proposals=(phase,),
        )


def test_policy_free_controller_exposes_hidden_native_horizon(monkeypatch) -> None:
    problem = SimpleNamespace(done=False)

    class RawEnvironment:
        def step(self, action):
            problem.done = True
            return {}, 0.0, False, {}

    class ScalarEnvironment:
        _env = RawEnvironment()

        @staticmethod
        def _format_raw_obs(raw):
            return raw

    controller = PolicyFreeController(SimpleNamespace(envs=[ScalarEnvironment()]))
    monkeypatch.setattr(
        phase3b_libero, "libero_problem_environment", lambda environment: problem
    )
    monkeypatch.setattr(controller, "eef_position", lambda: np.zeros(3))
    monkeypatch.setattr(controller, "bowl_position", lambda: np.zeros(3))
    monkeypatch.setattr(controller, "bowl_grasped", lambda: False)
    monkeypatch.setattr(
        phase3b_libero,
        "evaluate_common_goals",
        lambda environment: {"drawer": False, "cabinet": False},
    )

    _, _, done, _ = controller.step(np.zeros(7, dtype=np.float32))

    assert done is True
    assert controller.done_values == [True]


def test_policy_free_servo_can_stop_without_budget_padding(monkeypatch) -> None:
    controller = PolicyFreeController(SimpleNamespace())
    monkeypatch.setattr(controller, "eef_position", lambda: np.zeros(3))
    monkeypatch.setattr(controller, "eef_orientation", lambda: np.eye(3))
    monkeypatch.setattr(controller, "bowl_position", lambda: np.zeros(3))

    def step(action):
        controller.actions.append(np.asarray(action, dtype=np.float32).copy())
        controller.done_values.append(False)
        controller.goal_values.append({"drawer": False, "cabinet": False})
        controller.grasp_values.append(True)
        controller.grasp_relative_positions.append(np.zeros(3))
        return {}, 0.0, False, {}

    monkeypatch.setattr(controller, "step", step)
    early = controller.servo(
        target_position=np.zeros(3),
        target_orientation=np.eye(3),
        gripper=-1.0,
        budget=5,
        max_translation_action=0.35,
        position_tolerance_m=0.01,
        pad_to_budget=False,
    )
    assert early["pass"] is True
    assert early["executed_action_steps"] == 0
    assert early["budgeted_action_steps"] == 5
    assert early["padded_to_budget"] is False

    padded = controller.servo(
        target_position=np.zeros(3),
        target_orientation=np.eye(3),
        gripper=-1.0,
        budget=5,
        max_translation_action=0.35,
        position_tolerance_m=0.01,
    )
    assert padded["pass"] is True
    assert padded["executed_action_steps"] == 5
    assert padded["padded_to_budget"] is True


def test_v36_registered_grasp_acquisition_config_is_locked(tmp_path) -> None:
    config = _load_config(PROJECT / "configs/phase3b_stage_a_v36.yaml")
    construction = config["construction"]
    assert construction["open_grasped_acquisition_mode"] == (
        "registered_cabinet_phase_v1"
    )
    assert construction["registered_grasp_bridge_pad_to_budget"] is False
    assert construction["root_final_timestep"] == 560

    changed = deepcopy(config)
    changed["construction"]["registered_grasp_acquisition_episode_index"] = 382
    changed_path = tmp_path / "changed.yaml"
    changed_path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="locked grasp construction episode"):
        _load_config(changed_path)


def test_v37_binds_registered_anchor_to_normalized_bowl() -> None:
    config = _load_config(PROJECT / "configs/phase3b_stage_a_v37.yaml")
    assert config["construction"] == _load_config(
        PROJECT / "configs/phase3b_stage_a_v36.yaml"
    )["construction"]
    assert config["action_phase_oracle"]["root_landmark_binding"] == (
        "normalized_bowl_translation_v1"
    )
    source = DemoTrace(
        goal="drawer",
        episode_index=694,
        task_index=12,
        frame_indices=np.asarray([1]),
        states=np.zeros((1, 8), dtype=np.float32),
        actions=np.zeros((1, 7), dtype=np.float32),
        action_sha256="a" * 64,
    )
    proposal = ActionPhaseProposal(
        source=source,
        suffix=source,
        anchor_position=np.asarray([0.1, 0.2, 1.1]),
        anchor_orientation=np.eye(3),
        metadata={
            "root_landmark_binding": "normalized_bowl_translation_v1",
            "root_landmark_tolerance_m": 0.02,
            "landmark_registration": {
                "target_landmark_position": [0.0, 0.0, 0.9],
                "target_landmark_tolerance_m": 1e-9,
            },
        },
    )
    anchor, registration = registered_root_execution_anchor(
        proposal,
        np.asarray([0.005, -0.002, 0.903]),
        config=config,
    )
    np.testing.assert_allclose(anchor, [0.105, 0.198, 1.103])
    assert registration["mode"] == "normalized_bowl_translation_v1"
    assert registration["tolerance_m"] == 0.02

    nominal = deepcopy(proposal)
    nominal.metadata["root_landmark_binding"] = "nominal_target_landmark_v1"
    with pytest.raises(RuntimeError, match="exceeds its binding tolerance"):
        registered_root_execution_anchor(
            nominal,
            np.asarray([0.005, -0.002, 0.903]),
            config=config,
        )


def test_grasped_transport_requires_both_servo_pass_and_possession() -> None:
    class Controller:
        def __init__(self, grasped: bool) -> None:
            self.grasped = grasped

        def bowl_grasped(self) -> bool:
            return self.grasped

    valid_phase = {
        "pass": True,
        "final_position_error_m": 0.01,
        "grasp_preserved_every_step": True,
        "no_goal_every_step": True,
        "nonterminal_every_step": True,
    }
    _validate_grasped_transport_phase(
        Controller(True),
        valid_phase,
        candidate_id="candidate",
        phase="return",
    )
    with pytest.raises(RuntimeError, match="Servo phase return failed"):
        _validate_grasped_transport_phase(
            Controller(True),
            {"pass": False, "final_position_error_m": 0.12},
            candidate_id="candidate",
            phase="return",
        )
    with pytest.raises(RuntimeError, match="lost the bowl"):
        _validate_grasped_transport_phase(
            Controller(False),
            valid_phase,
            candidate_id="candidate",
            phase="return",
        )
    transient_loss = dict(
        valid_phase,
        grasp_preserved_every_step=False,
        max_consecutive_grasp_dropout_steps=1,
        max_grasp_relative_pose_deviation_m=0.01,
    )
    with pytest.raises(RuntimeError, match="transiently lost"):
        _validate_grasped_transport_phase(
            Controller(True),
            transient_loss,
            candidate_id="candidate",
            phase="return",
        )
    _validate_grasped_transport_phase(
        Controller(True),
        transient_loss,
        candidate_id="candidate",
        phase="return",
        max_grasp_dropout_steps=1,
        max_relative_pose_deviation_m=0.02,
    )


def test_grasped_root_recovery_plan_reverses_physical_waypoints() -> None:
    positions = np.asarray(
        [
            [-0.12, 0.07, 1.31],
            [0.02, -0.11, 1.31],
            [0.02, -0.11, 1.29],
        ]
    )
    lifted = np.asarray([-0.12, 0.07, 1.01])
    budgets = {
        "clearance_lift": 150,
        "clearance_transit": 120,
        "target_descent": 70,
    }
    plan = grasped_root_recovery_plan(
        positions, lifted, phase_budgets=budgets
    )
    assert [phase["phase"] for phase in plan] == [
        "reverse_target_descent",
        "reverse_clearance_transit",
        "reverse_clearance_lift",
    ]
    assert [phase["budget"] for phase in plan] == [70, 120, 150]
    np.testing.assert_allclose(plan[0]["target_position"], positions[1])
    np.testing.assert_allclose(plan[1]["target_position"], positions[0])
    np.testing.assert_allclose(plan[2]["target_position"], lifted)
    assert [phase["intermediate"] for phase in plan] == [True, True, False]
    with pytest.raises(ValueError, match="three finite transit positions"):
        grasped_root_recovery_plan(
            positions[:2], lifted, phase_budgets=budgets
        )


def _support_state(**updates) -> dict:
    state = {
        "reference_id": "reference",
        "goal": "drawer",
        "demo_episode_index": 1,
        "demo_task_index": 2,
        "demo_action_sha256": "c" * 64,
        "frame_index": 3,
        "layout": "A",
        "drawer_aperture": "closed",
        "possession": "on_table",
        "transit_locus": "drawer_side",
        "motion_event": "stationary",
        "eef_position": [0.0, 0.0, 1.0],
        "eef_orientation": np.eye(3).tolist(),
        "robot_joint_positions": [0.0] * 7,
        "bowl_position": [0.1, 0.0, 0.9],
        "drawer_joint": 0.0,
        "eef_motion": [0.0, 0.0, 0.0],
        "bowl_motion": [0.0, 0.0, 0.0],
        "action_motion": [0.0] * 6,
        "grasp_relative_position": [0.0, 0.0, 0.0],
        "drawer_goal_distance": 0.2,
        "cabinet_goal_distance": 0.3,
    }
    state.update(updates)
    return state


def _support_scales() -> dict[str, float]:
    return {
        "eef_position_m": 0.1,
        "eef_orientation_rad": 0.5,
        "robot_joint_rms_rad": 0.5,
        "bowl_position_m": 0.1,
        "drawer_joint_m": 0.05,
        "eef_motion_m": 0.02,
        "bowl_motion_m": 0.02,
        "action_motion_rms": 0.5,
        "grasp_relative_position_m": 0.05,
        "goal_distance_m": 0.15,
    }


def test_joint_support_cannot_be_won_by_action_only_similarity() -> None:
    query = _support_state()
    exact = _support_state(reference_id="exact")
    geometry_mismatch = _support_state(
        reference_id="action-only",
        bowl_position=[0.4, 0.0, 0.9],
        drawer_joint=-0.15,
        drawer_aperture="open",
    )
    exact_distance, _ = joint_support_distance(
        query,
        exact,
        scales=_support_scales(),
        categorical_mismatch_penalty=2.0,
    )
    mismatch_distance, components = joint_support_distance(
        query,
        geometry_mismatch,
        scales=_support_scales(),
        categorical_mismatch_penalty=2.0,
    )
    assert exact_distance == pytest.approx(0.0)
    assert mismatch_distance > 0.0
    assert components["bowl_position"] == pytest.approx(3.0)
    assert components["drawer_aperture_mismatch"] == pytest.approx(2.0)

    measurement = measure_joint_support(
        query,
        [geometry_mismatch, exact],
        scales=_support_scales(),
        categorical_mismatch_penalty=2.0,
    )
    assert measurement["nearest"]["reference"]["reference_id"] == "exact"
    assert measurement["event_matching_reference_count"] == 1
