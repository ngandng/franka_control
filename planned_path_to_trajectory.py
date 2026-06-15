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
            "name":        str,
            "joints":      list[float],          # 7 arm joints
            "attachments": dict[str, int|None],  # robot_id -> obj_id or None
            "is_transition": bool,
        }
    """
    with open(filepath) as f:
        text = f.read()

    # Map each configuration name (q1, q2, ...) to its orbit-level Robot State.
    # This drives gripper transitions: TRANSIT->TRANSFER closes, TRANSFER->TRANSIT opens.
    config_robot_state = {}
    orbit_blocks = re.finditer(r'(Orbit\d+:.*?)(?=\nOrbit\d+:|\Z)', text, re.DOTALL)
    for orbit_match in orbit_blocks:
        orbit_block = orbit_match.group(1)
        state_match = re.search(r"Robot States:\s*\{0:\s*'([^']+)'\}", orbit_block)
        robot_state = state_match.group(1) if state_match else None
        for cfg_name in re.findall(r'\n\s*(q\d+)\s+\(Configuration\s+\d+\):', orbit_block):
            config_robot_state[cfg_name] = robot_state

    # Split on configuration headers  ── q1, q2, …
    blocks = re.split(r'\n  (q\d+) \(Configuration \d+\):', text)
    # blocks[0] = preamble, then alternating [name, body, name, body, …]

    configurations = []
    for i in range(1, len(blocks), 2):
        name = blocks[i].strip()
        body = blocks[i + 1]

        # Joint angles  ── first list found in the block
        m = re.search(r'Joint Angles:\s*\[\[(.*?)\]\]', body)
        joints = [float(v) for v in m.group(1).split(',')] if m else []

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
            "joints":        joints,
            "attachments":   raw_attach,
            "is_transition": is_trans,
            "robot_state":   config_robot_state.get(name),
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


def build_trajectory(configurations: list[dict]) -> list[dict]:
    trajectory = []
    t = 0.0
    current_gripper = gripper_for_robot_state(
        configurations[0].get("robot_state") if configurations else None,
        GRIPPER_OPEN,
    )

    for idx in range(len(configurations)):
        cfg      = configurations[idx]
        joints   = cfg["joints"]
        gripper  = current_gripper

        if idx == 0:
            # First waypoint — just hold home pose
            trajectory.append({
                "time":    round(t, 4),
                "joints":  [round(v, 5) for v in joints],
                "gripper": round(gripper, 5),
                "event":   f"START {cfg['name']}",
            })
            t += TIME_STEP
            continue

        prev_cfg    = configurations[idx - 1]
        prev_joints = prev_cfg["joints"]
        prev_gripper = current_gripper

        prev_robot_state = prev_cfg.get("robot_state")
        curr_robot_state = cfg.get("robot_state")
        if prev_robot_state != curr_robot_state:
            current_gripper = gripper_for_robot_state(curr_robot_state, current_gripper)
        gripper = current_gripper

        # Do gripper transitions before moving toward the next configuration.
        # Trigger strictly on Robot State changes, independent of Is Transition.
        if gripper != prev_gripper:
            action = "CLOSE GRIPPER" if gripper == GRIPPER_CLOSED else "OPEN GRIPPER"
            g_wps = gripper_transition(prev_gripper, gripper, t)
            for wp in g_wps:
                wp["joints"] = [round(v, 5) for v in prev_joints]  # hold current pose while gripper changes
            g_wps[0]["event"] = f"{action} AT {prev_cfg['name']} (BEFORE {cfg['name']})"
            trajectory.extend(g_wps)
            t += GRIPPER_ACTION_SEC


        # ── 3a. Move arm from prev to current ─────────────────────────────
        joint_deltas = np.abs(np.array(joints) - np.array(prev_joints))
        max_joint_delta = np.max(joint_deltas)

        # Steps based on resolution: how many steps to keep each step ≤ MAX_RAD_PER_STEP
        resolution_steps = int(np.ceil(max_joint_delta / MAX_RAD_PER_STEP))

        # Never fewer than 2 steps, never fewer than minimum duration
        min_steps = max(2, int(SECS_PER_WAYPOINT / TIME_STEP))
        move_steps = max(min_steps, resolution_steps)

        move_wps = interpolate(prev_joints, joints, gripper, move_steps, t)
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
    output_file: str = "trajectory_from_planned_path.json",
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading: {input_txt}")
    configurations = parse_planned_path(input_txt)

    # Deduplicate consecutive identical configs (transition configs appear in two orbits)
    deduped = [configurations[0]]
    for cfg in configurations[1:]:
        if (
            cfg["joints"] != deduped[-1]["joints"]
            or cfg.get("robot_state") != deduped[-1].get("robot_state")
        ):
            deduped.append(cfg)
    configurations = deduped

    print(f"Found {len(configurations)} configurations:")
    for cfg in configurations:
        g = "CLOSED" if gripper_for_robot_state(cfg.get("robot_state"), GRIPPER_OPEN) == GRIPPER_CLOSED else "OPEN"
        print(f"  {cfg['name']:4s}  joints={cfg['joints']}  "
              f"robot_state={cfg.get('robot_state')}  gripper={g}  transition={cfg['is_transition']}")

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
        input_txt="simple_pick_place_planned_path_seed_44.txt",
    )
