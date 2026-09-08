
import numpy as np
import pybullet as p
import pybullet_data
import json
import time


home_q = [0, -0.5, 0, -2.5, 0, 2.0, 0.8]

# Simulate each command interval with five physics substeps.  The command
# period itself is read from the trajectory JSON, so 50 Hz and 100 Hz files
# both replay at their recorded simulated timing.
PHYSICS_STEPS_PER_COMMAND = 5
PLAYBACK_SPEED = 5.0  # 1.0 = real time; 5.0 = five times faster


def run_simulation_check(filename="path_data/dual_arm_trajectory.json", playback_speed=PLAYBACK_SPEED):
    if playback_speed <= 0:
        raise ValueError("playback_speed must be greater than zero.")

    with open(filename, 'r') as f:
        trajectory = json.load(f)
    if len(trajectory) < 2:
        raise ValueError("Trajectory must contain at least two timed waypoints.")
    command_period = trajectory[1]["time"] - trajectory[0]["time"]
    if command_period <= 0:
        raise ValueError("Trajectory timestamps must be strictly increasing.")
    if any(abs(step["time"] - previous["time"] - command_period) > 1e-6
           for previous, step in zip(trajectory, trajectory[1:])):
        raise ValueError("PyBullet player requires uniformly spaced trajectory timestamps.")
    physics_time_step = command_period / PHYSICS_STEPS_PER_COMMAND

    # Setup standard PyBullet physics server
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setTimeStep(physics_time_step)
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    # ======= PLANE AND FIXED OBJECTS =======
    plane = p.loadURDF("plane.urdf")

    table_position = [0.4, 0.0, 0.0]
    table = p.loadURDF("models/custom_table/custom_table.urdf",
                          basePosition=table_position,
                          baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]),
                          useFixedBase=True)
    table_height = 0.88


    # ======= MOVABLE OBJECTS ==========
    obj_positions = [[table_position[0]+0.15, table_position[1]+0.2, 0.925],
                     [table_position[0]+0.15, table_position[1]-0.2, 0.925]]

    obj_ori = p.getQuaternionFromEuler([0,0,0])
    for obj_pos in obj_positions:
        obj = p.loadURDF("cube_small.urdf", obj_pos, obj_ori, globalScaling=1.2)

    # ======= FRANKA ROBOT =======
    table_surface_z = table_position[2] + table_height + 0.02       # 2cm support of our robot
    # Physical setup: right arm is on +Y and left arm is on -Y.
    left_arm_id = p.loadURDF("franka_panda/panda.urdf",
                          basePosition=[table_position[0]-0.2, table_position[1]-0.5, table_surface_z],
                          baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), 
                          useFixedBase=True)
    right_arm_id = p.loadURDF("franka_panda/panda.urdf",
                          basePosition=[table_position[0]-0.2, table_position[1]+0.5, table_surface_z],
                          baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                          useFixedBase=True)
    franka_joint_indices = [0, 1, 2, 3, 4, 5, 6]
    franka_finger_indices = [9, 10]  # left and right finger joints; each = half the gripper width
    for i, joint_angle in zip(franka_joint_indices, home_q):
        p.resetJointState(left_arm_id, i, joint_angle)             # reset robot to neutral home pose
        p.resetJointState(right_arm_id, i, joint_angle)            # reset right arm to neutral home pose

    print(
        f"Loaded trajectory with {len(trajectory)} steps at {1 / command_period:.0f} Hz. "
        "Running pre-check..."
    )
    time.sleep(1)

    # Playback the trajectory file
    for step_data in trajectory:
        # Support both schemas:
        # 1) single-arm: joints/gripper
        # 2) dual-arm: left_joints/right_joints and left_gripper/right_gripper
        if "left_joints" in step_data and "right_joints" in step_data:
            left_target_joints = step_data["left_joints"]
            right_target_joints = step_data["right_joints"]

            p.setJointMotorControlArray(
                bodyUniqueId=left_arm_id,
                jointIndices=franka_joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=left_target_joints
            )
            p.setJointMotorControlArray(
                bodyUniqueId=right_arm_id,
                jointIndices=franka_joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=right_target_joints
            )

            if "left_gripper" in step_data:
                left_finger_pos = step_data["left_gripper"] / 2.0
                p.setJointMotorControlArray(
                    bodyUniqueId=left_arm_id,
                    jointIndices=franka_finger_indices,
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=[left_finger_pos, left_finger_pos]
                )
            if "right_gripper" in step_data:
                right_finger_pos = step_data["right_gripper"] / 2.0
                p.setJointMotorControlArray(
                    bodyUniqueId=right_arm_id,
                    jointIndices=franka_finger_indices,
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=[right_finger_pos, right_finger_pos]
                )
        else:
            target_joints = step_data["joints"]

            # Backward compatibility: command the same single-arm trajectory on both robots.
            p.setJointMotorControlArray(
                bodyUniqueId=left_arm_id,
                jointIndices=franka_joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=target_joints
            )
            p.setJointMotorControlArray(
                bodyUniqueId=right_arm_id,
                jointIndices=franka_joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=target_joints
            )

            if "gripper" in step_data:
                finger_pos = step_data["gripper"] / 2.0
                p.setJointMotorControlArray(
                    bodyUniqueId=left_arm_id,
                    jointIndices=franka_finger_indices,
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=[finger_pos, finger_pos]
                )
                p.setJointMotorControlArray(
                    bodyUniqueId=right_arm_id,
                    jointIndices=franka_finger_indices,
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=[finger_pos, finger_pos]
                )

        # Advance a full 20 ms of simulation time for each trajectory target.
        # Sleep only controls how fast it is shown on screen, not the simulated
        # robot dynamics.
        for _ in range(PHYSICS_STEPS_PER_COMMAND):
            p.stepSimulation()
            time.sleep(physics_time_step / playback_speed)

    print("Simulation check finished. If the arm did not crash, it is safe for hardware.")
    print("PyBullet window will stay open. Press 'q' in the GUI or Ctrl+C in terminal to exit.")

    try:
        while p.isConnected():
            keys = p.getKeyboardEvents()
            if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                break
            p.stepSimulation()
            time.sleep(physics_time_step / playback_speed)
    except KeyboardInterrupt:
        pass

    p.disconnect()

if __name__ == "__main__":
    run_simulation_check()
