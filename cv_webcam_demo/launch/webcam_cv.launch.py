from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        Node(
            package="kinect_ros2",
            executable="kinect_ros2_node",
            name="kinect_ros2",
            output="screen",
            parameters=[{
                "lock_auto_exposure":      True,
                "lock_auto_white_balance": True,
            }],
        ),

        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_to_kinect_tf",
            arguments=[
                "--x", "0.0",
                "--y", "0.0",
                "--z", "0.0",
                "--roll", "0.0",
                "--pitch", "0.0",
                "--yaw", "0.0",
                "--frame-id", "base_link",
                "--child-frame-id", "kinect_rgb",
            ],
        ),

        Node(
            package="cv_webcam_demo",
            executable="kinect_detect",
            name="kinect_detect",
            output="screen",
            parameters=[{
                # contour scoring / classification
                "aspect_max":            3.0,
                "aspect_min":            0.2,
                "center_bias":           0.6,
                "epsilon_frac":          0.08,
                "max_area_frac":         0.3,
                "max_depth_stdev_mm":    150.0,
                "min_area":              100,
                "min_circularity":       0.45,
                "min_score":             0.4,
                "target_shape":          "any",

                # LAB ΔE chroma
                "delta_e_thresh":        12.0,
                "valid_L_max":           250.0,
                "valid_L_min":           10.0,
                "use_clahe":             True,
                "rgb_blur_ksize":        0,

                # depth-bump
                "depth_bump_thresh_m":   0.009,
                "depth_bump_erode_px":   0,

                # temporal smoothing
                "saliency_lpf_alpha":    0.2,
                "saliency_median_ksize": 0,

                # table mask
                "plane_band_m":          0.01,
                "min_table_area_px":     5000,
                "table_mask_erode_px":   5,

                # morphology
                "morph_open":            1,
                "morph_close":           8,

                # RANSAC
                "voxel_leaf_m":          0.002,
                "ransac_dist_thresh_m":  0.025,
                "ransac_max_iter":       1500,
                "ransac_max_attempts":   3,
                "min_inlier_count":      1000,
                "plane_min_d_m":         0.3,
                "plane_max_d_m":         1.5,
                "plane_refit_every":     1,

                # workspace + normal gates
                "workspace_x_min":      -10.0,
                "workspace_x_max":       10.0,
                "workspace_y_min":      -10.5,
                "workspace_y_max":       1.3,
                "workspace_z_min":      -10.3,
                "workspace_z_max":       10.2,
                "plane_normal_min_z":    0.0,

                # plane tracking
                "plane_track_band_m":    0.025,
                "plane_lpf_alpha":       0.4,
                "plane_reseed_every_sec": 5.0,
                "min_track_inliers":     1000,

                # registration
                "depth_to_rgb_x":        0.021,
                "depth_to_rgb_y":       -0.005,
                "depth_to_rgb_z":        0.0,

                # tracking state machine
                "reacquire_every":       1,
                "lost_timeout_sec":      0.8,

                # output
                "draw":                  True,
            }],
        ),

    ])
