## Installation on Ubuntu 24.04 LTS
Step 1: Install Prerequisites & Build Tools
```
sudo apt update && sudo apt upgrade -y
sudo apt install git wget cmake build-essential libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev -y
```
Step 2: Clone the Repository
```
git clone https://github.com/realsenseai/librealsense.git
cd librealsense
```
Step 3: Configure and Build the SDK
```
mkdir build && cd build
cmake .. -DFORCE_RSUSB_BACKEND=true -DBUILD_EXAMPLES=true -DBUILD_WITH_CUDA=false
```
Compile and install SDK:
```
make -j$(nproc)
sudo make install
```
Step 4: Run Post-Installation Scripts
```
cd ../
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```
Verify installation
```
realsense-viewer
```

## Inside our project code
```
source .venv/bin/activate
pip install pyrealsense2 opencv-python
```