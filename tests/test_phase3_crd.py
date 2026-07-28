from types import SimpleNamespace

import numpy as np
import pytest

from smolvla_analysis.phase3_crd import (
    CONTINUATION_SCHEDULES,
    DEFAULT_STATE_SPECS,
    GOAL_PREDICATES,
    action_statistics,
    certificate_within_tolerance,
    continuation_seed,
    evaluate_common_goals,
    expected_query_ids,
    factor_query_id,
    is_monotonic_numeric_tolerance_relaxation,
    is_monotonic_branch_source_upgrade,
    is_monotonic_state_capture_upgrade,
    iter_branch_specs,
    legacy_cross_instance_branch_ids,
    nested_field_max_abs_differences,
    predicted_archive_init_state,
    validate_branch_accounting,
    validate_paired_first_plan,
    validate_query_accounting,
)


class FakeProblem:
    def __init__(self):
        self.sim = object()

    @staticmethod
    def _eval_predicate(predicate):
        assert predicate[0] in {"in", "on"}
        return predicate[0] == "in"


def _fake_environment():
    problem = FakeProblem()
    control = SimpleNamespace(
        env=problem,
        get_sim_state=lambda: np.zeros(1),
        regenerate_obs_from_state=lambda state: state,
    )
    return SimpleNamespace(envs=[SimpleNamespace(_env=control)])


def test_phase3_branch_matrix_is_unique_and_bounded():
    branches = iter_branch_specs()

    assert len(DEFAULT_STATE_SPECS) == 10
    assert len(branches) == 160
    assert len({branch.branch_id for branch in branches}) == 160
    assert len({branch.query_id for branch in branches}) == 80
    assert len(expected_query_ids(include_factors=False)) == 80
    assert len(expected_query_ids(include_factors=True)) == 160
    refresh = legacy_cross_instance_branch_ids()
    assert len(refresh) == 30
    assert len(set(refresh)) == 30
    assert not any("step0050__goal_drawer__proposal_101" in item for item in refresh)


def test_common_goal_evaluator_is_independent_of_native_task():
    status = evaluate_common_goals(_fake_environment())

    assert set(status) == set(GOAL_PREDICATES)
    assert status == {"drawer": True, "cabinet": False}


def test_continuation_schedule_is_common_across_proposals():
    left = [continuation_seed(0, index) for index in range(3)]
    right = [continuation_seed(0, index) for index in range(3)]

    assert left == right
    assert len(set(left)) == 3
    assert continuation_seed(1, 0) != continuation_seed(0, 0)
    with pytest.raises(ValueError):
        continuation_seed(max(CONTINUATION_SCHEDULES) + 1, 0)


def test_historical_init_state_prediction_accounts_for_double_autoreset():
    assert predicted_archive_init_state(0, 0) == 0
    assert predicted_archive_init_state(3, 2) == 7
    assert predicted_archive_init_state(4, 4) == 12


def test_action_statistics_validate_shape_and_are_finite():
    values = action_statistics(np.ones((50, 7), dtype=np.float32))
    assert values["plan_rms"] == pytest.approx(1.0)
    assert values["plan_temporal_std"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="Hx7"):
        action_statistics(np.ones((50, 6)))


def test_state_certificate_separates_roundoff_from_pixel_changes():
    differences = {
        "pixels/image": 0.0,
        "pixels/image2": 0.0,
        "robot_state/eef/pos": 1.7e-16,
    }
    assert certificate_within_tolerance(2e-14, differences)
    assert not certificate_within_tolerance(2e-9, differences)
    assert not certificate_within_tolerance(2e-14, differences | {"pixels/image": 1.0})
    assert not certificate_within_tolerance(
        2e-14, differences | {"robot_state/eef/pos": 2e-9}
    )


def test_nested_field_comparison_fails_closed_on_schema_and_shape_changes():
    exact = nested_field_max_abs_differences(
        {"pixels": np.zeros((2, 2), dtype=np.uint8), "label": "same"},
        {"pixels": np.zeros((2, 2), dtype=np.uint8), "label": "same"},
    )
    assert exact == {"label": 0.0, "pixels": 0.0}

    missing = nested_field_max_abs_differences({"pixels": np.zeros(1)}, {})
    wrong_shape = nested_field_max_abs_differences(np.zeros(1), np.zeros(2))
    unequal_label = nested_field_max_abs_differences("left", "right")
    assert np.isinf(missing["pixels"])
    assert np.isinf(wrong_shape["<root>"])
    assert np.isinf(unequal_label["<root>"])


def test_contract_migration_allows_only_monotonic_numeric_tolerance_relaxation():
    existing = {
        "schema_version": 2,
        "branch_horizon": 150,
        "certificate_tolerances": {
            "mujoco_state_atol": 1e-10,
            "numeric_observation_atol": 1e-12,
            "pixel_observation_atol": 0.0,
        },
    }
    proposed = {
        **existing,
        "certificate_tolerances": existing["certificate_tolerances"]
        | {"numeric_observation_atol": 1e-10},
    }
    assert is_monotonic_numeric_tolerance_relaxation(existing, proposed)
    assert not is_monotonic_numeric_tolerance_relaxation(proposed, existing)
    assert not is_monotonic_numeric_tolerance_relaxation(
        existing, proposed | {"branch_horizon": 200}
    )


def test_contract_upgrade_allows_only_full_sim_state_field_addition():
    existing = {"schema_version": 2, "branch_horizon": 150}
    proposed = existing | {"full_sim_data_fields": ["qacc_warmstart", "ctrl"]}
    assert is_monotonic_state_capture_upgrade(existing, proposed)
    assert not is_monotonic_state_capture_upgrade(proposed, proposed)
    assert not is_monotonic_state_capture_upgrade(
        existing, proposed | {"branch_horizon": 200}
    )


def test_contract_upgrade_allows_only_per_process_archive_replay_addition():
    existing = {"schema_version": 2, "branch_horizon": 150}
    proposed = existing | {
        "branch_source_reconstruction": "archive_action_replay_current_process"
    }
    assert is_monotonic_branch_source_upgrade(existing, proposed)
    assert not is_monotonic_branch_source_upgrade(proposed, proposed)
    assert not is_monotonic_branch_source_upgrade(
        existing, proposed | {"branch_horizon": 200}
    )


def test_branch_accounting_reports_missing_and_rejects_unexpected():
    expected = iter_branch_specs(states=DEFAULT_STATE_SPECS[:1])
    result = validate_branch_accounting(expected, [{"branch_id": expected[0].branch_id}])
    assert result["completed"] == 1
    assert len(result["missing"]) == len(expected) - 1
    with pytest.raises(ValueError, match="unexpected"):
        validate_branch_accounting(expected, [{"branch_id": "not-a-real-branch"}])


def test_paired_first_plan_invariant_is_strict_and_schedule_aware():
    common = {
        "query_id": "state__goal_drawer__proposal_101",
        "initial_goal_status": {"drawer": False, "cabinet": False},
        "source_reconstruction": {"mode": "archive_action_replay_current_process"},
        "first10_effect": {"bowl_displacement_norm": 0.0},
        "first_plan_steps": 50,
        "first_plan_effect": {"bowl_displacement_norm": 0.2},
    }
    left = common | {"continuation_schedule": 0}
    right = common | {"continuation_schedule": 1}

    assert validate_paired_first_plan(left, right)["exact"]
    with pytest.raises(ValueError, match="first-plan invariant"):
        validate_paired_first_plan(
            left,
            right | {"first10_effect": {"bowl_displacement_norm": 1e-12}},
        )
    with pytest.raises(ValueError, match="continuation schedules"):
        validate_paired_first_plan(left, right | {"continuation_schedule": 0})


def test_factor_query_ids_and_query_ledger_are_strict():
    state = DEFAULT_STATE_SPECS[0]
    identifier = factor_query_id(state.state_id, "drawer", "wrist_mean")
    assert identifier.endswith("goal_drawer__factor_wrist_mean__proposal_101")
    expected = expected_query_ids(states=[state], include_factors=False)
    result = validate_query_accounting(
        expected[:1], states=[state], include_factors=False
    )
    assert result["completed"] == 1
    assert not result["complete"]
    with pytest.raises(ValueError, match="unexpected"):
        validate_query_accounting(
            ["not-a-real-query"], states=[state], include_factors=False
        )
