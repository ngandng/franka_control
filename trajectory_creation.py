import json
import os
import numpy as np

def generate_and_save_trajectory(folder="path_data", filename="example_path.json"):
    # Ensure the output folder exists
    os.makedirs(folder, exist_ok=True)

    # Full file path
    filepath = os.path.join(folder, filename)
    
    # Suppose your motion planner outputs a path of waypoints at 50Hz (20ms intervals)
    time_step = 0.020  
    total_time = 5.0   # 5 seconds
    steps = int(total_time / time_step)
    
    # Baseline starting position (Franka Home)
    home_q = [0, -0.5, 0, -2.5, 0, 2.0, 0.8]
    
    trajectory_data = []
    
    for step in range(steps):
        t = step * time_step
        
        target_q = list(home_q)
        # Animate Joint 3 (index 3). Amplitude = 0.3 rad (~17 degrees)
        # Frequency = 0.4 Hz (2 full cycles over 5 seconds)
        target_q[3] += 0.3 * np.sin(2 * np.pi * 0.4 * t) 
        
        trajectory_data.append({
            "time": round(t, 4),
            "joints": [round(q, 5) for q in target_q]
        })
        
    # Save to a reusable file
    with open(filepath, 'w') as f:
        json.dump(trajectory_data, f, indent=4)
    print(f"✅ Trajectory profile successfully written to {filepath}")

if __name__ == "__main__":
    generate_and_save_trajectory()