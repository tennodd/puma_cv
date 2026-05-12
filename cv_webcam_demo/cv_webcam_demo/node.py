import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2

from cv_interfaces.msg import Detection

from .detection import _DetectionMixin
from .plane import _PlaneMixin
from .registration import _RegistrationMixin
from .saliency import _SaliencyMixin
from .tracker import _TrackerMixin


class KinectDetectV2Node(
    _DetectionMixin,
    _SaliencyMixin,
    _RegistrationMixin,
    _PlaneMixin,
    _TrackerMixin,
    Node,
):
    def __init__(self):
        super().__init__("kinect_detect")
        self._bridge = CvBridge()

        self.declare_parameter("voxel_leaf_m",         0.005)
        self.declare_parameter("ransac_dist_thresh_m", 0.010)
        self.declare_parameter("ransac_max_iter",      1500)
        self.declare_parameter("plane_lpf_alpha",      0.4)
        self.declare_parameter("min_inlier_count",     500)
        self.declare_parameter("plane_min_d_m",        0.30)
        self.declare_parameter("plane_max_d_m",        1.50)
        self.declare_parameter("workspace_x_min",     -1.0)
        self.declare_parameter("workspace_x_max",      1.0)
        self.declare_parameter("workspace_y_min",     -0.5)
        self.declare_parameter("workspace_y_max",      0.8)
        self.declare_parameter("workspace_z_min",      0.30)
        self.declare_parameter("workspace_z_max",      2.50)
        self.declare_parameter("plane_normal_min_z",   0.85)
        self.declare_parameter("ransac_max_attempts",  3)
        self.declare_parameter("plane_refit_every",    10)
        self.declare_parameter("plane_track_band_m",      0.025)
        self.declare_parameter("plane_reseed_every_sec",  5.0)
        self.declare_parameter("min_track_inliers",       1000)

        self.declare_parameter("plane_band_m",        0.015)
        self.declare_parameter("min_table_area_px",   5000)
        self.declare_parameter("table_mask_erode_px",  5)
        self.declare_parameter("delta_e_thresh",      18.0)
        self.declare_parameter("valid_L_min",         13.0)
        self.declare_parameter("valid_L_max",        242.0)
        self.declare_parameter("use_clahe",           True)
        self.declare_parameter("depth_bump_thresh_m", 0.010)
        self.declare_parameter("depth_bump_erode_px", 2)

        self.declare_parameter("depth_to_rgb_x", 0.025)
        self.declare_parameter("depth_to_rgb_y", 0.0)
        self.declare_parameter("depth_to_rgb_z", 0.0)
        self.declare_parameter("saliency_lpf_alpha",  0.4)
        self.declare_parameter("rgb_blur_ksize",       3)
        self.declare_parameter("saliency_median_ksize", 3)

        self.declare_parameter("morph_open",          5)
        self.declare_parameter("morph_close",        21)
        self.declare_parameter("min_area",         2000)
        self.declare_parameter("max_area_frac",    0.40)
        self.declare_parameter("aspect_min",       0.35)
        self.declare_parameter("aspect_max",        3.0)
        self.declare_parameter("max_depth_stdev_mm", 150.0)
        self.declare_parameter("epsilon_frac",     0.03)
        self.declare_parameter("min_circularity",  0.72)
        self.declare_parameter("center_bias",       0.6)
        self.declare_parameter("min_score",         0.0)
        self.declare_parameter("target_shape",    "any")
        self.declare_parameter("reacquire_every",     3)
        self.declare_parameter("lost_timeout_sec",  0.6)
        self.declare_parameter("draw",             True)

        self._latest_rgb     = None
        self._latest_depth   = None
        self._cam_info       = None
        self._depth_cam_info = None
        self._latest_cloud   = None
        self._plane_smoothed = None
        self._saliency_smoothed = None
        self._tracker         = None
        self._tracking        = False
        self._last_bbox       = None
        self._last_class      = "none"
        self._last_area       = 0.0
        self._last_score      = 0.0
        self._last_det_center = None
        self._last_z_mm       = 0
        self._last_good_ts    = time.time()
        self._frame_i         = 0
        self._plane_log_i     = 0
        self._timing_log_i    = 0
        self._t_table         = 0.0
        self._t_saliency      = 0.0
        self._t_clean         = 0.0
        self._t_contours      = 0.0
        self._t_detect_total  = 0.0

        self._plane_lock    = threading.Lock()
        self._plane_request = threading.Event()
        self._shutdown      = threading.Event()
        self._worker_busy   = False

        self._last_seed_time   = 0.0
        self._tracking_active  = False
        self._plane_track_log_i = 0
        self._diag_log_i        = 0

        self._plane_thread = threading.Thread(
            target=self._plane_worker, name="plane_worker", daemon=True)
        self._plane_thread.start()

        self.create_subscription(Image,       "/image_raw",                    self._on_rgb,        10)
        self.create_subscription(Image,       "/depth/image_raw",              self._on_depth,      10)
        self.create_subscription(CameraInfo,  "/camera_info",                  self._on_info,       10)
        self.create_subscription(CameraInfo,  "/depth/camera_info",            self._on_depth_info, 10)
        self.create_subscription(PointCloud2, "/points",                       self._on_cloud,      10)

        self._pub_img   = self.create_publisher(Image,       "/camera/image_annotated",   10)
        self._pub_det   = self.create_publisher(Detection,   "/detections",               10)
        self._pub_mask  = self.create_publisher(Image,       "/camera/debug_plane_mask",  10)
        self._pub_morph = self.create_publisher(Image,       "/camera/debug_depth_morph", 10)
        self._pub_debug_channels = self.create_publisher(Image, "/camera/debug_channels", 10)

        self.get_logger().info("KinectDetectV2 ready (plane-anchored saliency, split package).")

    def _on_info(self, msg):
        self._cam_info = msg

    def _on_depth_info(self, msg):
        self._depth_cam_info = msg

    def _on_rgb(self, msg):
        try:
            rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            self._latest_rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.get_logger().warn(f"rgb decode failed: {e}")

    def _on_cloud(self, msg):
        try:
            structured = point_cloud2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True)
            pts = np.stack(
                [structured["x"], structured["y"], structured["z"]],
                axis=-1,
            ).astype(np.float32, copy=False)
        except Exception as e:
            self.get_logger().warn(f"cloud decode failed: {e}")
            return
        if pts.size == 0:
            return
        with self._plane_lock:
            self._latest_cloud = pts.reshape(-1, 3)

    def _on_depth(self, msg):
        self._t_table = 0.0
        self._t_saliency = 0.0
        self._t_clean = 0.0
        self._t_contours = 0.0
        self._t_detect_total = 0.0

        t0 = time.perf_counter()
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        except Exception as e:
            self.get_logger().warn(f"depth decode failed: {e}")
            return
        t_decode = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        depth_reg = self._register_depth(depth)
        t_register = (time.perf_counter() - t0) * 1000.0
        if depth_reg is None:
            return
        self._latest_depth = depth_reg

        if self._latest_rgb is None:
            return

        refit_every = max(1, int(self.get_parameter("plane_refit_every").value))
        t0 = time.perf_counter()
        should_request = (self._frame_i % refit_every) == 0
        if should_request:
            with self._plane_lock:
                if not self._worker_busy:
                    self._plane_request.set()
        t_fit = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._process(msg.header)
        t_process = (time.perf_counter() - t0) * 1000.0

        self._timing_log_i += 1
        if self._timing_log_i % 30 == 0:
            self.get_logger().info(
                f"timing ms: depth_decode={t_decode:.1f} register={t_register:.1f} "
                f"fit_plane={t_fit:.1f} process={t_process:.1f} | "
                f"table_mask={self._t_table:.1f} saliency={self._t_saliency:.1f} "
                f"clean={self._t_clean:.1f} contours={self._t_contours:.1f} "
                f"detect_total={self._t_detect_total:.1f}"
            )

    def destroy_node(self):
        self._shutdown.set()
        self._plane_request.set()
        if self._plane_thread.is_alive():
            self._plane_thread.join(timeout=2.0)
        super().destroy_node()


def main():
    rclpy.init()
    node = KinectDetectV2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
