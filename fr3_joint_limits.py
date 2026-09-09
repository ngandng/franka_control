"""Joint limits used by trajectory generation and hardware replay."""

# Joint order is J1 through J7.  Values are in radians and radians/second.
# FR3 position limits.
FR3_JOINT_LOWER_LIMITS = (
    -2.7437, -1.7837, -2.9007, -3.0421, -2.8065,  0.5445, -3.0159,
)
FR3_JOINT_UPPER_LIMITS = (
     2.7437,  1.7837,  2.9007, -0.1518,  2.8065,  4.5169,  3.0159,
)
FR3_JOINT_VELOCITY_LIMITS = (
    2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100,
)

# Uniform speed scale for generated trajectories and hardware replay.  For
# example, 0.10 commands 10% of each physical joint's velocity limit.
TRAJECTORY_VELOCITY_SCALE = 0.2
TRAJECTORY_MAX_JOINT_VELOCITIES = tuple(
    TRAJECTORY_VELOCITY_SCALE * limit
    for limit in FR3_JOINT_VELOCITY_LIMITS
)

# Move to the start pose under the same per-joint scaled velocity limits.
START_MOVE_MAX_JOINT_VELOCITIES = TRAJECTORY_MAX_JOINT_VELOCITIES
