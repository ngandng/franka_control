import ast
import json
import os
import re
import numpy as np

from panda_joint_limits import (
    PANDA_JOINT_LOWER_LIMITS,
    PANDA_JOINT_UPPER_LIMITS,
    PANDA_JOINT_VELOCITY_LIMITS,
    TRAJECTORY_MAX_JOINT_VELOCITIES,
)


# ── Gripper constants ─────────────────────────────────────────────────────────
GRIPPER_OPEN   = 0.08   # metres (Franka max aperture)
GRIPPER_CLOSED = 0.055   # metres (requested transfer/grasp width)

# ── Interpolation settings ────────────────────────────────────────────────────
TIME_STEP          = 0.010      # 100 Hz
SECS_PER_WAYPOINT  = 2.0        # seconds to travel between two configurations
GRIPPER_ACTION_SEC = 1.0        # seconds spent opening / closing gripper
QUINTIC_SMOOTHSTEP_MAX_DERIVATIVE = 1.875

# The planner's robot order follows the world Y axis: robot 0 is at +Y and
# robot 1 is at -Y.  On the physical setup, +Y is the right arm and -Y is the
# left arm.  Keep this mapping in one place so the JSON field names cannot be
# accidentally interpreted in planner order.
PLANNER_RIGHT_ARM_INDEX = 0
PLANNER_LEFT_ARM_INDEX = 1


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Parse the planned-path text file
# ─────────────────────────────────────────────────────────────────────────────

def parse_planned_path(filepath: str) -> list[dict]:
    """
    Returns a list of configuration dicts, in path order:
        {
            "name":          str,
            "arms":          list[list[float]],  # one or two 7-joint arms
            "attachments":   dict[int, int|None],
            "is_transition": bool,
            "robot_states":  dict[int, str],
        }
    """
    with open(filepath) as f:
        text = f.read()

    # Map each configuration name (q1, q2, ...) to its orbit-level Robot State.
    # This drives gripper transitions: TRANSIT->TRANSFER closes, TRANSFER->TRANSIT opens.
    config_robot_states = {}
    orbit_blocks = re.finditer(r'(Orbit\d+:.*?)(?=\nOrbit\d+:|\Z)', text, re.DOTALL)
    for orbit_match in orbit_blocks:
        orbit_block = orbit_match.group(1)
        state_match = re.search(r"Robot States:\s*\{(.*?)\}", orbit_block)
        robot_states = {}
        if state_match:
            for robot_id, robot_state in re.findall(r"(\d+):\s*'([^']+)'", state_match.group(1)):
                robot_states[int(robot_id)] = robot_state
        for cfg_name in re.findall(r'\n\s*(q\d+)\s+\(Configuration\s+\d+\):', orbit_block):
            config_robot_states[cfg_name] = robot_states

    # Split on configuration headers  ── q1, q2, …
    blocks = re.split(r'\n  (q\d+) \(Configuration \d+\):', text)
    # blocks[0] = preamble, then alternating [name, body, name, body, …]

    configurations = []
    for i in range(1, len(blocks), 2):
        name = blocks[i].strip()
        body = blocks[i + 1]

        # Joint angles can be either [[...]] or [[...], [...]]
        m = re.search(r'Joint Angles:\s*(\[\[.*?\]\])', body, re.DOTALL)
        if not m:
            raise ValueError(f"Could not find joint angles for {name} in {filepath}")

        parsed_joints = ast.literal_eval(m.group(1))
        if not isinstance(parsed_joints, list) or not parsed_joints:
            raise ValueError(f"Invalid joint data for {name} in {filepath}")

        if isinstance(parsed_joints[0], (int, float)):
            arms = [list(float(v) for v in parsed_joints)]
        else:
            arms = [list(float(v) for v in arm) for arm in parsed_joints]

        for arm_index, arm_joints in enumerate(arms):
            if len(arm_joints) != 7:
                raise ValueError(
                    f"{filepath} contains {len(arm_joints)} joints for {name} arm {arm_index}, "
                    "but each arm must have exactly 7 joints."
                )

        # Attachments  ── e.g.  {0: 0}  or  {0: None}
        m = re.search(r'Attachments:\s*(\{.*?\})', body)
        raw_attach = {}
        if m:
            for k, v in re.findall(r'(\d+):\s*(\w+)', m.group(1)):
                raw_attach[int(k)] = None if v == 'None' else int(v)

        # Is transition
        is_trans = bool(re.search(r'Is Transition:\s*True', body))

        configurations.append({
            "name":          name,
            "arms":          arms,
            "attachments":   raw_attach,
            "is_transition": is_trans,
            "robot_states":  config_robot_states.get(name, {}),
        })

    return configurations


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Determine gripper state at each configuration
# ─────────────────────────────────────────────────────────────────────────────

def gripper_for_robot_state(robot_state: str | None, current_gripper: float) -> float:
    if robot_state == "TRANSFER":
        return GRIPPER_CLOSED
    if robot_state == "TRANSIT":
        return GRIPPER_OPEN
    return current_gripper


def infer_arm_count(configurations: list[dict]) -> int:
    if not configurations:
        raise ValueError("No configurations were parsed from the planned path.")

    arm_count = len(configurations[0]["arms"])
    if arm_count not in (1, 2):
        raise ValueError(f"Unsupported arm count: {arm_count}. Expected 1 or 2.")

    for cfg in configurations:
        if len(cfg["arms"]) != arm_count:
            raise ValueError("Mixed single-arm and dual-arm configurations are not supported.")

    return arm_count


def gripper_for_arm(robot_states: dict[int, str], arm_index: int, current_gripper: float) -> float:
    return gripper_for_robot_state(robot_states.get(arm_index), current_gripper)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Build trajectory waypoints
# ─────────────────────────────────────────────────────────────────────────────

# def interpolate(q_start, q_end, gripper, steps: int, t_start: float) -> list[dict]:
#     """Linear interpolation between two joint configs, fixed gripper."""
#     waypoints = []
#     for i in range(steps):
#         alpha = i / max(steps - 1, 1)
#         q = [s + alpha * (e - s) for s, e in zip(q_start, q_end)]
#         waypoints.append({
#             "time":    round(t_start + i * TIME_STEP, 4),
#             "joints":  [round(v, 5) for v in q],
#             "gripper": round(gripper, 5),
#         })
#     return waypoints

def interpolate(q_start, q_end, gripper, steps, t_start):
    waypoints = []
    for i in range(steps):
        alpha = quintic_smoothstep(i / max(steps - 1, 1))
        q = [s + alpha * (e - s) for s, e in zip(q_start, q_end)]
        waypoints.append({
            "time":    round(t_start + i * TIME_STEP, 4),
            "joints":  [round(v, 5) for v in q],
            "gripper": round(gripper, 5),
        })
    return waypoints


def quintic_smoothstep(progress: float) -> float:
    """Position profile with zero velocity and acceleration at both ends."""
    return 10 * progress**3 - 15 * progress**4 + 6 * progress**5


def gripper_transition(gripper_from, gripper_to, t_start: float) -> list[dict]:
    """Hold joints still and switch gripper target once, then hold it."""
    # We don't know arm joints here — caller patches them in
    steps = max(1, int(GRIPPER_ACTION_SEC / TIME_STEP))
    waypoints = []
    for i in range(steps):
        # One command edge, then hold to avoid repeated open/close triggers.
        g = gripper_to
        waypoints.append({
            "time":    round(t_start + i * TIME_STEP, 4),
            "joints":  None,   # filled by caller
            "gripper": round(g, 5),
        })
    return waypoints


def _format_waypoint(
    time_value: float,
    arms: list[list[float]],
    grippers: list[float],
    arm_count: int,
    event: str | None = None,
) -> dict:
    waypoint = {"time": round(time_value, 4)}
    if arm_count == 1:
        waypoint["joints"] = [round(v, 5) for v in arms[0]]
        waypoint["gripper"] = round(grippers[0], 5)
    else:
        # Planner robot 0 (+Y) is the physical right arm; planner robot 1
        # (-Y) is the physical left arm.
        waypoint["left_joints"] = [round(v, 5) for v in arms[PLANNER_LEFT_ARM_INDEX]]
        waypoint["right_joints"] = [round(v, 5) for v in arms[PLANNER_RIGHT_ARM_INDEX]]
        waypoint["left_gripper"] = round(grippers[PLANNER_LEFT_ARM_INDEX], 5)
        waypoint["right_gripper"] = round(grippers[PLANNER_RIGHT_ARM_INDEX], 5)
    if event is not None:
        waypoint["event"] = event
    return waypoint


def build_trajectory(configurations: list[dict]) -> list[dict]:
    arm_count = infer_arm_count(configurations)
    trajectory = []
    t = 0.0
    current_grippers = [GRIPPER_OPEN] * arm_count

    first_robot_states = configurations[0].get("robot_states", {})
    for arm_index in range(arm_count):
        current_grippers[arm_index] = gripper_for_arm(first_robot_states, arm_index, current_grippers[arm_index])

    for idx in range(len(configurations)):
        cfg      = configurations[idx]
        arms     = cfg["arms"]

        if idx == 0:
            # First waypoint — just hold home pose
            trajectory.append(_format_waypoint(t, arms, current_grippers, arm_count, event=f"START {cfg['name']}"))
            t += TIME_STEP
            continue

        prev_cfg    = configurations[idx - 1]
        prev_arms = prev_cfg["arms"]
        prev_grippers = list(current_grippers)

        prev_robot_states = prev_cfg.get("robot_states", {})
        curr_robot_states = cfg.get("robot_states", {})
        for arm_index in range(arm_count):
            if prev_robot_states.get(arm_index) != curr_robot_states.get(arm_index):
                current_grippers[arm_index] = gripper_for_arm(curr_robot_states, arm_index, current_grippers[arm_index])

        if current_grippers != prev_grippers:
            actions = []
            for arm_index in range(arm_count):
                if current_grippers[arm_index] != prev_grippers[arm_index]:
                    side = "RIGHT" if arm_index == PLANNER_RIGHT_ARM_INDEX else "LEFT"
                    action = "CLOSE" if current_grippers[arm_index] == GRIPPER_CLOSED else "OPEN"
                    actions.append(f"{side} {action} GRIPPER")

            g_wps = gripper_transition(prev_grippers[0], current_grippers[0], t)
            for wp in g_wps:
                wp["time"] = round(wp["time"], 4)
                if arm_count == 1:
                    wp["joints"] = [round(v, 5) for v in prev_arms[0]]
                else:
                    wp.pop("joints", None)
                    wp.pop("gripper", None)
                    wp["left_joints"] = [round(v, 5) for v in prev_arms[PLANNER_LEFT_ARM_INDEX]]
                    wp["right_joints"] = [round(v, 5) for v in prev_arms[PLANNER_RIGHT_ARM_INDEX]]
                    wp["left_gripper"] = round(current_grippers[PLANNER_LEFT_ARM_INDEX], 5)
                    wp["right_gripper"] = round(current_grippers[PLANNER_RIGHT_ARM_INDEX], 5)
            g_wps[0]["event"] = f"{'; '.join(actions)} AT {prev_cfg['name']} (BEFORE {cfg['name']})"
            trajectory.extend(g_wps)
            t += GRIPPER_ACTION_SEC


        # ── 3a. Move arm from prev to current ─────────────────────────────
        # The quintic profile has a maximum derivative of 1.875.  Size this
        # shared dual-arm segment from every individual joint's scaled velocity
        # limit.  There is deliberately no independent per-waypoint angle cap:
        # TRAJECTORY_VELOCITY_SCALE is the single trajectory speed control.
        # Adding one gives a duration of (steps - 1) * TIME_STEP between the
        # first and final target rather than treating the first target as a
        # movement interval.
        velocity_steps = max(
            int(np.ceil(
                QUINTIC_SMOOTHSTEP_MAX_DERIVATIVE * abs(end - start)
                / (velocity_limit * TIME_STEP)
            )) + 1
            for previous_arm, current_arm in zip(prev_arms, arms)
            for start, end, velocity_limit in zip(
                previous_arm, current_arm, TRAJECTORY_MAX_JOINT_VELOCITIES
            )
        )
        min_steps = max(2, int(np.ceil(SECS_PER_WAYPOINT / TIME_STEP)) + 1)
        move_steps = max(min_steps, velocity_steps)

        move_wps = []
        for i in range(move_steps):
            alpha = quintic_smoothstep(i / max(move_steps - 1, 1))
            interpolated_arms = [
                [s + alpha * (e - s) for s, e in zip(prev_arm, curr_arm)]
                for prev_arm, curr_arm in zip(prev_arms, arms)
            ]
            move_wps.append(
                _format_waypoint(
                    t + i * TIME_STEP,
                    interpolated_arms,
                    current_grippers,
                    arm_count,
                )
            )
        move_wps[0]["event"] = f"MOVE {prev_cfg['name']} -> {cfg['name']}"
        trajectory.extend(move_wps)
        t += move_steps * TIME_STEP

    return trajectory


def validate_panda_joint_limits(trajectory: list[dict]) -> None:
    """Reject trajectories beyond Panda position or physical velocity limits."""
    violations = []
    for waypoint_index, waypoint in enumerate(trajectory):
        for arm_name in ("left", "right"):
            joints = waypoint[f"{arm_name}_joints"]
            for joint_index, (value, lower, upper) in enumerate(
                zip(joints, PANDA_JOINT_LOWER_LIMITS, PANDA_JOINT_UPPER_LIMITS),
                start=1,
            ):
                if not lower <= value <= upper:
                    violations.append(
                        f"{arm_name} J{joint_index}={value:.5f} rad at t={waypoint['time']:.2f}s "
                        f"(allowed [{lower:.4f}, {upper:.4f}])"
                    )

    for previous, current in zip(trajectory, trajectory[1:]):
        time_delta = current["time"] - previous["time"]
        if time_delta <= 0:
            violations.append(
                f"Non-increasing trajectory timestamps at t={previous['time']:.2f}s and "
                f"t={current['time']:.2f}s."
            )
            continue
        for arm_name in ("left", "right"):
            for joint_index, (start, end, velocity_limit) in enumerate(
                zip(
                    previous[f"{arm_name}_joints"],
                    current[f"{arm_name}_joints"],
                    PANDA_JOINT_VELOCITY_LIMITS,
                ),
                start=1,
            ):
                velocity = abs(end - start) / time_delta
                if velocity > velocity_limit + 1e-6:
                    violations.append(
                        f"{arm_name} J{joint_index}={velocity:.5f} rad/s between "
                        f"t={previous['time']:.2f}s and t={current['time']:.2f}s "
                        f"(physical limit {velocity_limit:.4f} rad/s)"
                    )
                planned_velocity_limit = TRAJECTORY_MAX_JOINT_VELOCITIES[joint_index - 1]
                if velocity > planned_velocity_limit + 1e-3:
                    violations.append(
                        f"{arm_name} J{joint_index}={velocity:.5f} rad/s between "
                        f"t={previous['time']:.2f}s and t={current['time']:.2f}s "
                        f"(generated-path limit {planned_velocity_limit:.4f} rad/s)"
                    )

    if violations:
        preview = "\n  ".join(violations[:10])
        raise ValueError(
            f"Trajectory contains {len(violations)} Panda position/velocity-limit violations. "
            "Replan the source path; do not export it for hardware playback.\n"
            f"  {preview}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Save
# ─────────────────────────────────────────────────────────────────────────────

def generate_trajectory_from_planned_path(
    input_txt:  str = "simple_pick_place_2_objects_dual_arm.txt",
    output_dir: str = "path_data",
    output_file: str | None = None,
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading: {input_txt}")
    configurations = parse_planned_path(input_txt)

    # Deduplicate consecutive identical configs (transition configs appear in two orbits)
    deduped = [configurations[0]]
    for cfg in configurations[1:]:
        if (
            cfg["arms"] != deduped[-1]["arms"]
            or cfg.get("robot_states") != deduped[-1].get("robot_states")
        ):
            deduped.append(cfg)
    configurations = deduped

    arm_count = infer_arm_count(configurations)
    if output_file is None:
        output_file = "dual_arm_trajectory.json" #if arm_count == 2 else "trajectory_from_planned_path.json"

    print(f"Found {len(configurations)} configurations:")
    for cfg in configurations:
        if arm_count == 1:
            g = "CLOSED" if gripper_for_robot_state(cfg.get("robot_states", {}).get(0), GRIPPER_OPEN) == GRIPPER_CLOSED else "OPEN"
            print(
                f"  {cfg['name']:4s}  joints={cfg['arms'][0]}  "
                f"robot_state={cfg.get('robot_states')}  gripper={g}  transition={cfg['is_transition']}"
            )
        else:
            left_g = "CLOSED" if gripper_for_robot_state(cfg.get("robot_states", {}).get(PLANNER_LEFT_ARM_INDEX), GRIPPER_OPEN) == GRIPPER_CLOSED else "OPEN"
            right_g = "CLOSED" if gripper_for_robot_state(cfg.get("robot_states", {}).get(PLANNER_RIGHT_ARM_INDEX), GRIPPER_OPEN) == GRIPPER_CLOSED else "OPEN"
            print(
                f"  {cfg['name']:4s}  left={cfg['arms'][PLANNER_LEFT_ARM_INDEX]}  "
                f"right={cfg['arms'][PLANNER_RIGHT_ARM_INDEX]}  "
                f"grippers=(left={left_g}, right={right_g})  transition={cfg['is_transition']}"
            )

    trajectory = build_trajectory(configurations)
    if arm_count == 2:
        validate_panda_joint_limits(trajectory)

    # Print events for quick sanity check
    print("\nTrajectory events:")
    for wp in trajectory:
        if "event" in wp:
            print(f"  t={wp['time']:6.3f}s  {wp['event']}")

    out_path = os.path.join(output_dir, output_file)
    with open(out_path, "w") as f:
        json.dump(trajectory, f, indent=2)

    total_time = trajectory[-1]["time"]
    print(f"\n✅ Trajectory written to {out_path}")
    print(f"   Waypoints : {len(trajectory)}")
    print(f"   Total time: {total_time:.2f}s")


if __name__ == "__main__":
    generate_trajectory_from_planned_path(
        input_txt="simple_pick_place_2_objects_dual_arm.txt",
    )
