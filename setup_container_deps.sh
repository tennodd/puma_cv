set -euo pipefail

echo "[setup] Updating apt index..."
apt-get update -qq

echo "[setup] Installing libfreenect (Kinect v1 driver lib)..."
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libfreenect-dev \
    libfreenect0.5

echo "[setup] Installing ROS 2 image processing packages..."
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ros-jazzy-depth-image-proc \
    ros-jazzy-image-transport \
    ros-jazzy-camera-info-manager \
    ros-jazzy-image-tools

echo "[setup] Installing v2 detector dependency (Open3D for plane-fit RANSAC)..."
pip install --break-system-packages --ignore-installed open3d
pip uninstall --break-system-packages -y numpy setuptools

echo "[setup] Done. Remember to colcon build kinect_ros2 after this."
