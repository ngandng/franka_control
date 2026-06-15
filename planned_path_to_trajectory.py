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
GRIPPER_ACTION_SEC = 0.2        # seconds spent opening / closing gripper
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
        })

    return configurations


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Determine gripper state at each configuration
# ─────────────────────────────────────────────────────────────────────────────

def gripper_state(cfg: dict) -> float:
    """Open if no object attached, closed otherwise."""
    for robot_id, obj_id in cfg["attachments"].items():
        if obj_id is not None:
            return GRIPPER_CLOSED
    return GRIPPER_OPEN


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
    """Hold joints still, animate gripper open/close."""
    # We don't know arm joints here — caller patches them in
    steps = max(1, int(GRIPPER_ACTION_SEC / TIME_STEP))
    waypoints = []
    for i in range(steps):
        alpha = i / max(steps - 1, 1)
        g = gripper_from + alpha * (gripper_to - gripper_from)
        waypoints.append({
            "time":    round(t_start + i * TIME_STEP, 4),
            "joints":  None,   # filled by caller
            "gripper": round(g, 5),
        })
    return waypoints


def build_trajectory(configurations: list[dict]) -> list[dict]:
    trajectory = []
    t = 0.0

    for idx in range(len(configurations)):
        cfg      = configurations[idx]
        joints   = cfg["joints"]
        gripper  = gripper_state(cfg)

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
        prev_gripper = gripper_state(prev_cfg)


        # ── 3a. Move arm from prev to current ─────────────────────────────
        joint_deltas = np.abs(np.array(joints) - np.array(prev_joints))
        max_joint_delta = np.max(joint_deltas)

        # Steps based on resolution: how many steps to keep each step ≤ MAX_RAD_PER_STEP
        resolution_steps = int(np.ceil(max_joint_delta / MAX_RAD_PER_STEP))

        # Never fewer than 2 steps, never fewer than minimum duration
        min_steps = max(2, int(SECS_PER_WAYPOINT / TIME_STEP))
        move_steps = max(min_steps, resolution_steps)

        move_wps = interpolate(prev_joints, joints, prev_gripper, move_steps, t)
        move_wps[0]["event"] = f"MOVE {prev_cfg['name']} -> {cfg['name']}"
        trajectory.extend(move_wps)
        t += move_steps * TIME_STEP


        # ── 3b. Gripper action at transition configurations ────────────────
        if cfg["is_transition"] and gripper != prev_gripper:
            action   = "CLOSE GRIPPER" if gripper == GRIPPER_CLOSED else "OPEN GRIPPER"
            g_wps    = gripper_transition(prev_gripper, gripper, t)
            for wp in g_wps:
                wp["joints"] = [round(v, 5) for v in joints]  # hold position
            g_wps[0]["event"] = f"{action} @ {cfg['name']}"
            trajectory.extend(g_wps)
            t += GRIPPER_ACTION_SEC

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
        if cfg["joints"] != deduped[-1]["joints"]:
            deduped.append(cfg)
    configurations = deduped

    print(f"Found {len(configurations)} configurations:")
    for cfg in configurations:
        g = "CLOSED" if gripper_state(cfg) == GRIPPER_CLOSED else "OPEN"
        print(f"  {cfg['name']:4s}  joints={cfg['joints']}  "
              f"gripper={g}  transition={cfg['is_transition']}")

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
