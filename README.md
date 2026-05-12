# puma_cv

Visual recognition subsystem for the PUMA 560 robotic manipulator team project. Built on Kinect v1 + classical computer vision (OpenCV, Open3D).

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
# Install Open3D (not available via rosdep)
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

Per-frame: bounding box (pixels), 3D camera-frame coordinates (meters), shape class, score, source state.

## Tuning

All detector parameters are exposed via ROS dynamic reconfigure. Use `rqt_reconfigure` or `ros2 param set` for live tuning.

## TF

Static `base_link → kinect_rgb` published as placeholder identity transform. Real values to be substituted when manipulator URDF is finalized.
