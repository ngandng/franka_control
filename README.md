# Testing and control FR3 

## Step 1. Setting up
- Prefered os: Ubuntu
- Turn on the realtime kernel to meets Franka Control Interface (FCI) demand:
    - First check with `uname -a`. If it shows `PREEMPT_RT` we are ready. Otherwise, update to Ubuntu pro and turn on the real time kernel

## Step 2. Physical Setup and Network
- Connect ethernet cable directly from ethernet port on the Franka Control Box with your workstation. Note that PCI operates over a strict 1kHz UDP network protocol, so avoid and network switches or routers if possible.
- Static IP Configuration: On your Ubuntu workstation, open network setting and config your wired connection with:
    - IPv4 -> Manual
    - IP address: 172.16.0.x (avoid x=2 as it is the ip address of the robot)
    - Netmask: 255.255.255.0
    - Gateway: leave blank
    - Apply and test the connection `ping 172.16.0.2`. (Robot light is white)

## Step 3. Installation
- Create virtual environment and install dependencies
    ```
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

- Install `libfranka`
    ```
    # Replace with your desired version and Ubuntu codename
    VERSION=0.20.0      # 0.20.0 works with us
    CODENAME=noble      # or jammy, focal (To check ubuntu codename `lsb_release -a`)

    wget https://github.com/frankarobotics/libfranka/releases/download/${VERSION}/libfranka_${VERSION}_${CODENAME}_amd64.deb
    sudo dpkg -i libfranka_${VERSION}_${CODENAME}_amd64.deb
    ```
- Verify installation `dpkg -l | grep libfranka`
- Install `pylibfranka` if you work with python
    ```
    pip install pylibfranka
    ```
## Step 4: Setting up the robot via the web interface
Before running any external code, you must grant the workstation permission to control the joints via the browser-based dashboard called Franka Desk.

- **Open Franka Desk**: Open a browser window on your workstation and navigate to http://172.16.0.2. (Robot light can be white or red)
- **Release Emergency Button**: (Robot light change to blue)
- **Unlock the Joints:** If the robot was just powered on, you will see a prompt to unlock the joint brakes. Click Unlock. You will hear a series of physical "clicks" from the arm. (Robot light is blue)
- **Activate FCI**: Click to robot IP button (top right, next to Language), expand the menu, look for the FCI (Franka Control Interface) tab, and click Activate. (Robot light turns green)

⚠️ Safety Check: Ensure the entire workspace radius around the arm is completely clear of obstacles, and hold the physical E-Stop button in your hand before proceeding.

## Step 5: Running your code
- A quick start example to verify the installation and connections
    ```
    import pylibfranka

    # Connect to the robot
    robot = pylibfranka.Robot("172.16.0.2")
    state = robot.read_once()

    # Print joint positions
    print(f"Joint positions: {state.q}")
    ```

- An example of simple moving the robot
    ```
    python3 test_trajectory.py
    ```

## Usefull links
- FCI document: https://frankarobotics.github.io/docs/overview.html
- libfranka library: https://github.com/frankarobotics/libfranka
- Easy to follow tutorial of controlling FR3 via FCI: https://youtu.be/91wFDNHVXI4?si=Z9_2LZHEZCUTiwgX

