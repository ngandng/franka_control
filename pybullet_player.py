
import numpy as np
import pybullet as p
import pybullet_data
import json
import time


home_q = [0, -0.5, 0, -2.5, 0, 2.0, 0.8]


def run_simulation_check(filename="path_data/trajectory_from_planned_path.json"):
    # Setup standard PyBullet physics server
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")


    plane = p.loadURDF("plane.urdf")

    table_position = [0.0, 0.0, 0.0]
    table = p.loadURDF("models/custom_table/custom_table.urdf",
                          basePosition=table_position,
                          baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]),
                          useFixedBase=True)
    table_height = 0.88
    
    
    table_surface_z = table_position[2] + table_height + 0.02       # 2cm support of our robot
    robot_base_position = [-0.2, 0.0, table_surface_z]
    robot_id = p.loadURDF("franka_panda/panda.urdf",
                          basePosition=robot_base_position,
                          baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), 
                          useFixedBase=True)
    franka_joint_indices = [0, 1, 2, 3, 4, 5, 6]
    franka_finger_indices = [9, 10]  # left and right finger joints; each = half the gripper width
    for i, joint_angle in zip(franka_joint_indices, home_q):
        p.resetJointState(robot_id, i, joint_angle)             # reset robot to neutral home pose

    # Load your saved planning data file
    with open(filename, 'r') as f:
        trajectory = json.load(f)
        
    print(f"Loaded trajectory with {len(trajectory)} steps. Running pre-check...")
    time.sleep(1)

    # Playback the trajectory file
    for step_data in trajectory:
        start_time = time.monotonic()
        
        target_joints = step_data["joints"]

        # Command PyBullet's position controllers to match the file
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=franka_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_joints
        )

        # Set gripper finger positions if the trajectory includes gripper state.
        # Each finger joint = half the total gripper width.
        if "gripper" in step_data:
            finger_pos = step_data["gripper"] / 2.0
            p.setJointMotorControlArray(
                bodyUniqueId=robot_id,
                jointIndices=franka_finger_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=[finger_pos, finger_pos]
            )

        p.stepSimulation()
        
        # Maintain uniform playback timing (50Hz / 20ms)
        elapsed = time.monotonic() - start_time
        if elapsed < 0.020:
            time.sleep(0.020 - elapsed)
            
    print("Simulation check finished. If the arm did not crash, it is safe for hardware.")
    print("PyBullet window will stay open. Press 'q' in the GUI or Ctrl+C in terminal to exit.")

    try:
        while p.isConnected():
            keys = p.getKeyboardEvents()
            if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                break
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        pass

    p.disconnect()

if __name__ == "__main__":
    run_simulation_check()