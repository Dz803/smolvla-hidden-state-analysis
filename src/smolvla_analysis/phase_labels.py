from __future__ import annotations


def label_phase(state: dict | None, progress: float, thresholds: dict[str, float]) -> tuple[str, str]:
    """Return (phase, evidence); never invent object-state phases when state is absent."""
    if not state:
        bins = ("search/approach", "alignment", "pre-grasp", "grasp", "lift", "transport", "place", "terminal")
        return bins[min(7, int(max(0.0, min(progress, 0.999999)) * 8))], "normalized_progress_fallback"
    if state.get("terminal"):
        return "terminal", "environment_terminal"
    object_goal = state.get("object_goal_distance_m")
    if object_goal is not None and object_goal <= thresholds["goal_distance_m"]:
        return "place", "object_goal_distance"
    if state.get("object_lift_m", 0.0) >= thresholds["lift_height_m"]:
        return ("transport" if state.get("object_motion_m", 0.0) > 0 else "lift"), "object_height_motion"
    if state.get("gripper_closed"):
        return "grasp", "gripper_transition"
    distance = state.get("eef_object_distance_m")
    if distance is None:
        return label_phase(None, progress, thresholds)
    if distance <= thresholds["grasp_distance_m"]:
        return "pre-grasp", "eef_object_distance"
    if distance <= thresholds["alignment_distance_m"]:
        return "alignment", "eef_object_distance"
    return "search/approach", "eef_object_distance"

