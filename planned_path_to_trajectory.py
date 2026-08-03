import ast
import json
import os
import re
import numpy as np


# ── Gripper constants ─────────────────────────────────────────────────────────
GRIPPER_OPEN   = 0.08   # metres (Franka max aperture)
GRIPPER_CLOSED = 0.04   # metres (half closed)

# ── Interpolation settings ────────────────────────────────────────────────────
TIME_STEP          = 0.020      # 50 Hz
SECS_PER_WAYPOINT  = 2.0        # seconds to travel between two configurations
GRIPPER_ACTION_SEC = 1.0        # seconds spent opening / closing gripper
MAX_ALLOWED_VELOCITY = 0.3      # rad/s
MAX_DEGREES_PER_STEP = 0.3      # max joint movement per 20ms step (degrees)
MAX_RAD_PER_STEP = np.deg2rad(MAX_DEGREES_PER_STEP)


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
        # Cosine easing: slow start, fast middle, slow end
        alpha = (1 - np.cos(np.pi * i / max(steps - 1, 1))) / 2
        q = [s + alpha * (e - s) for s, e in zip(q_start, q_end)]
        waypoints.append({
            "time":    round(t_start + i * TIME_STEP, 4),
            "joints":  [round(v, 5) for v in q],
            "gripper": round(gripper, 5),
        })
    return waypoints


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
        waypoint["left_joints"] = [round(v, 5) for v in arms[0]]
        waypoint["right_joints"] = [round(v, 5) for v in arms[1]]
        waypoint["left_gripper"] = round(grippers[0], 5)
        waypoint["right_gripper"] = round(grippers[1], 5)
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
                    side = "LEFT" if arm_index == 0 else "RIGHT"
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
                    wp["left_joints"] = [round(v, 5) for v in prev_arms[0]]
                    wp["right_joints"] = [round(v, 5) for v in prev_arms[1]]
                    wp["left_gripper"] = round(current_grippers[0], 5)
                    wp["right_gripper"] = round(current_grippers[1], 5)
            g_wps[0]["event"] = f"{'; '.join(actions)} AT {prev_cfg['name']} (BEFORE {cfg['name']})"
            trajectory.extend(g_wps)
            t += GRIPPER_ACTION_SEC


        # ── 3a. Move arm from prev to current ─────────────────────────────
        joint_deltas = [
            np.abs(np.array(curr_arm) - np.array(prev_arm))
            for curr_arm, prev_arm in zip(arms, prev_arms)
        ]
        max_joint_delta = max(float(np.max(delta)) for delta in joint_deltas)

        # Steps based on resolution: how many steps to keep each step ≤ MAX_RAD_PER_STEP
        resolution_steps = int(np.ceil(max_joint_delta / MAX_RAD_PER_STEP))

        # Never fewer than 2 steps, never fewer than minimum duration
        min_steps = max(2, int(SECS_PER_WAYPOINT / TIME_STEP))
        move_steps = max(min_steps, resolution_steps)

        move_wps = []
        for i in range(move_steps):
            alpha = (1 - np.cos(np.pi * i / max(move_steps - 1, 1))) / 2
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


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Save
# ─────────────────────────────────────────────────────────────────────────────

def generate_trajectory_from_planned_path(
    input_txt:  str = "planned_path_seed_44_1r_1o.txt",
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
        output_file = "dual_arm_trajectory.json" if arm_count == 2 else "trajectory_from_planned_path.json"

    print(f"Found {len(configurations)} configurations:")
    for cfg in configurations:
        if arm_count == 1:
            g = "CLOSED" if gripper_for_robot_state(cfg.get("robot_states", {}).get(0), GRIPPER_OPEN) == GRIPPER_CLOSED else "OPEN"
            print(
                f"  {cfg['name']:4s}  joints={cfg['arms'][0]}  "
                f"robot_state={cfg.get('robot_states')}  gripper={g}  transition={cfg['is_transition']}"
            )
        else:
            left_g = "CLOSED" if gripper_for_robot_state(cfg.get("robot_states", {}).get(0), GRIPPER_OPEN) == GRIPPER_CLOSED else "OPEN"
            right_g = "CLOSED" if gripper_for_robot_state(cfg.get("robot_states", {}).get(1), GRIPPER_OPEN) == GRIPPER_CLOSED else "OPEN"
            print(
                f"  {cfg['name']:4s}  left={cfg['arms'][0]}  right={cfg['arms'][1]}  "
                f"robot_states={cfg.get('robot_states')}  grippers=({left_g}, {right_g})  transition={cfg['is_transition']}"
            )

    trajectory = build_trajectory(configurations)

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
        input_txt="dual_arm_trajectory.txt",
    )
