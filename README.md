# puma_cv

Visual recognition subsystem for the PUMA 560 robotic manipulator team project. Built on Kinect v1.

## Stack

- ROS 2 Jazzy
- Python 3.12
- OpenCV 4.6, Open3D 0.19, NumPy 1.26
- Kinect v1 via libfreenect

## Packages

- `cv_interfaces/` — message definitions (`Detection.msg`)
- `cv_webcam_demo/` — detector node
- `kinect_ros2/` — Kinect v1 driver

## Quick start

Place all three packages into the `src/` of a ROS 2 workspace, then:

```bash
# Install Open3D
bash setup_container_deps.sh

# Build
colcon build --packages-select cv_interfaces cv_webcam_demo kinect_ros2 --symlink-install
source install/setup.bash

# Launch
ros2 launch cv_webcam_demo webcam_cv.launch.py
```

## Output

Topic: `/detections`
Type: `cv_interfaces/msg/Detection`

## Tuning

All detector parameters are exposed via ROS dynamic reconfigure. Use `rqt_reconfigure` or `ros2 param set` for live tuning. Use `rqt` to watch image streams in real time
