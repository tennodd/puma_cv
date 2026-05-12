import time

import numpy as np
import open3d as o3d


class _PlaneMixin:
    def _fit_plane(self):
        if self._latest_cloud is None or self._latest_cloud.shape[0] < 100:
            return None
        voxel        = float(self.get_parameter("voxel_leaf_m").value)
        dist         = float(self.get_parameter("ransac_dist_thresh_m").value)
        iters        = int(self.get_parameter("ransac_max_iter").value)
        min_in       = int(self.get_parameter("min_inlier_count").value)
        d_min        = float(self.get_parameter("plane_min_d_m").value)
        d_max        = float(self.get_parameter("plane_max_d_m").value)
        max_attempts = int(self.get_parameter("ransac_max_attempts").value)
        normal_z_min = float(self.get_parameter("plane_normal_min_z").value)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self._latest_cloud)
        pcd_ds = pcd.voxel_down_sample(voxel_size=voxel)
        pts_np = np.asarray(pcd_ds.points)
        wx_min = float(self.get_parameter("workspace_x_min").value)
        wx_max = float(self.get_parameter("workspace_x_max").value)
        wy_min = float(self.get_parameter("workspace_y_min").value)
        wy_max = float(self.get_parameter("workspace_y_max").value)
        wz_min = float(self.get_parameter("workspace_z_min").value)
        wz_max = float(self.get_parameter("workspace_z_max").value)
        in_box = (
            (pts_np[:, 0] >= wx_min) & (pts_np[:, 0] <= wx_max) &
            (pts_np[:, 1] >= wy_min) & (pts_np[:, 1] <= wy_max) &
            (pts_np[:, 2] >= wz_min) & (pts_np[:, 2] <= wz_max)
        )
        pts_filtered = pts_np[in_box]
        if pts_filtered.shape[0] < min_in:
            self.get_logger().debug(
                f"REJECTED: workspace filter left only {pts_filtered.shape[0]} pts"
            )
            return None
        pcd_ds = o3d.geometry.PointCloud()
        pcd_ds.points = o3d.utility.Vector3dVector(pts_filtered)

        for attempt in range(max_attempts):
            n_pts = len(pcd_ds.points)
            if n_pts < min_in * 2:
                self.get_logger().debug(
                    f"REJECTED: not enough points ({n_pts}) on attempt {attempt+1}")
                return None

            try:
                plane_model, inliers = pcd_ds.segment_plane(
                    distance_threshold=dist, ransac_n=3, num_iterations=iters)
            except Exception as e:
                self.get_logger().warn(f"RANSAC failed: {e}")
                return None

            if len(inliers) < min_in:
                self.get_logger().debug(
                    f"REJECTED: low inliers ({len(inliers)}/{n_pts}) on attempt {attempt+1}")
                return None

            plane = np.array(plane_model, dtype=np.float64)  # [a, b, c, d]
            if plane[3] < 0.0:
                plane = -plane

            if d_min <= plane[3] <= d_max:
                if abs(plane[2]) < normal_z_min:
                    self.get_logger().debug(
                        f"REJECTED: |nz|={abs(plane[2]):.2f} < {normal_z_min} "
                        f"(attempt {attempt+1}, inliers={len(inliers)})"
                    )
                    pcd_ds = pcd_ds.select_by_index(inliers, invert=True)
                    continue

                self._plane_log_i += 1
                if self._plane_log_i % 3 == 0:
                    a, b, c, d = plane
                    self.get_logger().info(
                        f"plane fit: n=({a:+.3f},{b:+.3f},{c:+.3f}) d={d:+.3f}m "
                        f"|nz|={abs(c):.2f} inliers={len(inliers)}/{n_pts} attempts={attempt+1}"
                    )
                return plane

            self.get_logger().debug(
                f"REJECTED: d={plane[3]:+.3f}m out of [{d_min},{d_max}] "
                f"(attempt {attempt+1}, inliers={len(inliers)})")
            pcd_ds = pcd_ds.select_by_index(inliers, invert=True)

        return None

    def _smooth_plane(self, new_plane):
        if self._plane_smoothed is None:
            self._plane_smoothed = new_plane.copy()
            return
        a = float(self.get_parameter("plane_lpf_alpha").value)
        if np.dot(new_plane[:3], self._plane_smoothed[:3]) < 0.0:
            new_plane = -new_plane
        self._plane_smoothed = a * new_plane + (1.0 - a) * self._plane_smoothed
        n_norm = np.linalg.norm(self._plane_smoothed[:3]) + 1e-9
        self._plane_smoothed[:3] /= n_norm
        self._plane_smoothed[3]  /= n_norm

    def _track_plane(self):
        with self._plane_lock:
            if self._latest_cloud is None or self._plane_smoothed is None:
                return None
            prev_plane = self._plane_smoothed.copy()

        cloud = self._latest_cloud
        if cloud.shape[0] < 100:
            return None

        band   = float(self.get_parameter("plane_track_band_m").value)
        min_in = int(self.get_parameter("min_track_inliers").value)

        a0, b0, c0, d0 = prev_plane
        signed = cloud[:, 0] * a0 + cloud[:, 1] * b0 + cloud[:, 2] * c0 + d0
        inlier_mask = np.abs(signed) < band
        inliers = cloud[inlier_mask]
        n_in    = int(inliers.shape[0])
        n_total = int(cloud.shape[0])

        if n_in < min_in:
            return "LOST"

        centroid = inliers.mean(axis=0)
        centered = inliers - centroid
        try:
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        except Exception as e:
            self.get_logger().warn(f"plane track SVD failed: {e}")
            return None

        normal = Vt[-1].astype(np.float64)
        d_new  = -float(np.dot(normal, centroid))

        if float(np.dot(normal, prev_plane[:3])) < 0.0:
            normal = -normal
            d_new  = -d_new

        new_plane = np.array(
            [normal[0], normal[1], normal[2], d_new], dtype=np.float64)

        self._plane_track_log_i += 1
        if self._plane_track_log_i % 3 == 0:
            a, b, c, d = new_plane
            self.get_logger().info(
                f"plane track: n=({a:+.3f},{b:+.3f},{c:+.3f}) d={d:+.3f}m "
                f"inliers={n_in}/{n_total}"
            )
        return new_plane

    def _plane_worker(self):
        while not self._shutdown.is_set():
            if not self._plane_request.wait(timeout=1.0):
                continue
            self._plane_request.clear()

            with self._plane_lock:
                self._worker_busy = True

            reseed_every_sec = float(
                self.get_parameter("plane_reseed_every_sec").value)
            now_t = time.monotonic()
            need_seed = (
                not self._tracking_active
                or (now_t - self._last_seed_time) >= reseed_every_sec
            )

            if need_seed:
                try:
                    new_plane = self._fit_plane()
                except Exception as e:
                    self.get_logger().warn(f"plane worker RANSAC failed: {e}")
                    new_plane = None
                if new_plane is not None:
                    self._last_seed_time  = now_t
                    self._tracking_active = True
            else:
                try:
                    result = self._track_plane()
                except Exception as e:
                    self.get_logger().warn(f"plane worker tracking failed: {e}")
                    result = None
                if isinstance(result, str):  # "LOST"
                    self._tracking_active = False
                    new_plane = None
                else:
                    new_plane = result

            with self._plane_lock:
                if new_plane is not None:
                    self._smooth_plane(new_plane)
                self._worker_busy = False
