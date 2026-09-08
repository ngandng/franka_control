"""Franka Panda joint limits used by trajectory generation and hardware replay."""

# Joint order is J1 through J7.  Values are in radians and radians/second.
PANDA_JOINT_LOWER_LIMITS = (
    -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,
)
PANDA_JOINT_UPPER_LIMITS = (
     2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973,
)
PANDA_JOINT_VELOCITY_LIMITS = (
    2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100,
)

# Uniform speed scale for generated trajectories and hardware replay.  For
# example, 0.10 commands 10% of each physical joint's velocity limit.
TRAJECTORY_VELOCITY_SCALE = 0.2
TRAJECTORY_MAX_JOINT_VELOCITIES = tuple(
    TRAJECTORY_VELOCITY_SCALE * limit
    for limit in PANDA_JOINT_VELOCITY_LIMITS
)

# Move to the start pose under the same per-joint scaled velocity limits.
START_MOVE_MAX_JOINT_VELOCITIES = TRAJECTORY_MAX_JOINT_VELOCITIES
