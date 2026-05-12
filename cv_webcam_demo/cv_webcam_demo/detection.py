import math
import time

import cv2
import numpy as np

from cv_interfaces.msg import Detection

from .helpers import bbox_iou, clamp_bbox, classify_shape, right_angle_score


class _DetectionMixin:
    def _detect(self, depth_mm):
        t_det0 = time.perf_counter()
        self._diag_log_i += 1
        H, W = depth_mm.shape[:2]

        min_area    = int(self.get_parameter("min_area").value)
        max_area_frac = float(self.get_parameter("max_area_frac").value)
        aspect_min  = float(self.get_parameter("aspect_min").value)
        aspect_max  = float(self.get_parameter("aspect_max").value)
        eps_frac    = float(self.get_parameter("epsilon_frac").value)
        min_circ    = float(self.get_parameter("min_circularity").value)
        center_bias = float(self.get_parameter("center_bias").value)
        max_depth_stdev_mm = float(self.get_parameter("max_depth_stdev_mm").value)
        target      = str(self.get_parameter("target_shape").value).strip().lower()

        cx_img   = W * 0.5
        cy_img   = H * 0.5
        max_dist = math.sqrt(cx_img ** 2 + cy_img ** 2) + 1e-6

        if self._latest_rgb is None:
            self._t_detect_total = (time.perf_counter() - t_det0) * 1000.0
            return None, 0.0

        t0 = time.perf_counter()
        table_mask = self._build_table_mask(depth_mm)
        self._t_table = (time.perf_counter() - t0) * 1000.0
        if table_mask is None:
            self._t_detect_total = (time.perf_counter() - t_det0) * 1000.0
            return None, 0.0

        t0 = time.perf_counter()
        saliency, dE_arr, bump_arr, A_t, B_t = self._build_saliency_mask(
            self._latest_rgb, depth_mm, table_mask, return_diag=True)
        self._t_saliency = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        cleaned_mask = self._clean_mask(saliency)
        self._t_clean = (time.perf_counter() - t0) * 1000.0

        try:
            self._pub_mask.publish(self._bridge.cv2_to_imgmsg(table_mask,    encoding="mono8"))
            self._pub_morph.publish(self._bridge.cv2_to_imgmsg(cleaned_mask, encoding="mono8"))
        except Exception:
            pass

        t_contours0 = time.perf_counter()
        contours, _ = cv2.findContours(
            cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best         = None
        best_score   = 0.0
        best_contour = None

        for c in contours:
            area = float(cv2.contourArea(c))
            if area < min_area:
                continue
            if area > max_area_frac * H * W:
                continue

            x, y, w, h = cv2.boundingRect(c)
            aspect     = w / float(h) if h > 0 else 1.0
            if not (aspect_min <= aspect <= aspect_max):
                continue

            hull      = cv2.convexHull(c)
            hull_area = float(cv2.contourArea(hull)) + 1e-6
            if area / hull_area < 0.70:
                continue

            contour_mask = np.zeros_like(depth_mm, dtype=np.uint8)
            cv2.drawContours(contour_mask, [c], -1, 255, thickness=cv2.FILLED)
            valid_depth = depth_mm[(contour_mask > 0) & (depth_mm > 0)]
            if valid_depth.size < 10:
                continue
            depth_stdev = float(np.std(valid_depth))
            if depth_stdev > max_depth_stdev_mm:
                continue

            peri = cv2.arcLength(c, True)
            if peri < 1e-3:
                continue

            approx = cv2.approxPolyDP(c, eps_frac * peri, True)
            class_name, quad_pts, angle_deg = classify_shape(approx, area, min_circ)

            if target != "any" and class_name != target:
                continue

            cx_obj           = x + w * 0.5
            cy_obj           = y + h * 0.5
            dist             = math.sqrt((cx_obj - cx_img) ** 2 + (cy_obj - cy_img) ** 2)
            center_proximity = max(0.0, 1.0 - dist / max_dist)

            score  = area
            score *= 1.0 + center_bias * center_proximity
            if quad_pts is not None:
                score *= 0.6 + 0.4 * right_angle_score(quad_pts)

            if class_name in ("rect", "square"):
                quad_pts = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)

            center_xy = (int(round(cx_obj)), int(round(cy_obj)))
            if class_name in ("rect", "square"):
                rect_center = cv2.minAreaRect(c)[0]
                center_xy = (int(rect_center[0]), int(rect_center[1]))
            elif class_name == "circle":
                try:
                    (ecx, ecy), _ = cv2.minEnclosingCircle(approx)
                    center_xy = (int(ecx), int(ecy))
                except cv2.error:
                    pass

            if score > best_score:
                best_score   = score
                best         = ((x, y, w, h), class_name, area, angle_deg, quad_pts, center_xy)
                best_contour = c

        self._t_contours = (time.perf_counter() - t_contours0) * 1000.0

        if best is not None and best_contour is not None and (self._diag_log_i % 15 == 0):
            cmask = np.zeros_like(depth_mm, dtype=np.uint8)
            cv2.drawContours(cmask, [best_contour], -1, 255, thickness=cv2.FILLED)
            inside = cmask > 0
            n_inside = int(np.count_nonzero(inside))
            if n_inside > 0:
                de_thr   = float(self.get_parameter("delta_e_thresh").value)
                bump_thr = float(self.get_parameter("depth_bump_thresh_m").value)
                de_in    = dE_arr[inside]
                bump_in  = bump_arr[inside]

                de_above   = de_in   > de_thr
                bump_above = bump_in > bump_thr
                px_dE_only   = int(np.count_nonzero(de_above   & ~bump_above))
                px_bump_only = int(np.count_nonzero(~de_above  &  bump_above))
                px_both      = int(np.count_nonzero(de_above   &  bump_above))

                with self._plane_lock:
                    plane_snap = (
                        self._plane_smoothed.copy()
                        if self._plane_smoothed is not None else None
                    )
                if plane_snap is not None:
                    pa, pb, pc, pd = plane_snap
                    plane_str = (
                        f"({pa:+.3f},{pb:+.3f},{pc:+.3f},{pd:+.3f}m)"
                    )
                else:
                    plane_str = "None"

                self.get_logger().info(
                    f"diag: dE_mean={float(de_in.mean()):.1f} dE_std={float(de_in.std()):.1f} "
                    f"bump_mean={float(bump_in.mean()):.1f} bump_std={float(bump_in.std()):.1f} | "
                    f"A_t={A_t:.1f} B_t={B_t:.1f} | "
                    f"plane={plane_str} | "
                    f"px_dE_only={px_dE_only} px_bump_only={px_bump_only} px_both={px_both}"
                )

        self._t_detect_total = (time.perf_counter() - t_det0) * 1000.0
        return best, best_score

    def _sample_depth_mm(self, depth_mm, cx, cy, win=3):
        H, W = depth_mm.shape[:2]
        x0   = max(0, cx - win)
        x1   = min(W, cx + win + 1)
        y0   = max(0, cy - win)
        y1   = min(H, cy + win + 1)
        patch = depth_mm[y0:y1, x0:x1]
        valid = patch[(patch > 0)]
        if valid.size == 0:
            return 0
        return int(np.median(valid))

    def _process(self, header):
        if self._latest_rgb is None or self._latest_depth is None:
            return

        rgb_bgr  = self._latest_rgb.copy()
        depth_mm = self._latest_depth

        if rgb_bgr.shape[:2] != depth_mm.shape[:2]:
            return

        self._frame_i  += 1
        draw            = bool(self.get_parameter("draw").value)
        reacquire_every = int(self.get_parameter("reacquire_every").value)
        lost_timeout    = float(self.get_parameter("lost_timeout_sec").value)

        H, W = rgb_bgr.shape[:2]

        bbox       = None
        class_name = "none"
        area       = 0.0
        angle_deg  = 0.0
        quad_pts   = None
        det_center = None
        source     = "none"
        z_mm       = 0

        run_detect = (not self._tracking) or (
            reacquire_every > 0 and (self._frame_i % reacquire_every == 0)
        )

        if run_detect:
            result, det_score = self._detect(depth_mm)
            min_score = float(self.get_parameter("min_score").value)
            if result is not None and (min_score <= 0.0 or det_score >= min_score):
                bbox, class_name, area, angle_deg, quad_pts, det_center = result
                source             = "detect"
                if not self._tracking or bbox_iou(bbox, self._last_bbox) < 0.7:
                    self._tracker_init(rgb_bgr, bbox)
                self._last_bbox    = bbox
                self._last_class   = class_name
                self._last_area    = area
                self._last_score   = float(det_score)
                self._last_det_center = det_center
                self._last_good_ts = time.time()
            else:
                ok, tr_bbox = self._tracker_update(rgb_bgr)
                if ok:
                    bbox               = tr_bbox
                    class_name         = self._last_class
                    source             = "track"
                    self._last_bbox    = bbox
                    self._last_good_ts = time.time()
        else:
            ok, tr_bbox = self._tracker_update(rgb_bgr)
            if ok:
                bbox               = tr_bbox
                class_name         = self._last_class
                source             = "track"
                self._last_bbox    = bbox
                self._last_good_ts = time.time()
            else:
                self._tracking = False
                self._tracker  = None

        now       = time.time()
        timed_out = (now - self._last_good_ts) > lost_timeout

        if bbox is None and not timed_out and self._last_bbox is not None:
            bbox       = self._last_bbox
            class_name = self._last_class
            source     = "hold"

        det              = Detection()
        det.header       = header
        det.header.frame_id = "kinect_rgb"

        if bbox is None:
            det.class_name = "none"
            det.x = det.y = det.width = det.height = 0
            det.center_x = det.center_y = 0
            det.area = 0.0
            det.angle_deg = 0.0
            det.x_cam = det.y_cam = det.z_cam = 0.0
            det.score = 0.0
            det.source = source
            self._last_z_mm = 0
        else:
            x, y, w, h = clamp_bbox(*bbox, W, H)
            center_for_frame = det_center if det_center is not None else self._last_det_center
            if center_for_frame is not None:
                cx = max(0, min(int(center_for_frame[0]), W - 1))
                cy = max(0, min(int(center_for_frame[1]), H - 1))
            else:
                cx, cy = x + w // 2, y + h // 2

            if source == "detect":
                z_mm = self._sample_depth_mm(depth_mm, cx, cy)
            else:
                z_mm = self._sample_depth_mm(depth_mm, cx, cy)
                if z_mm == 0:
                    z_mm = self._last_z_mm
            self._last_z_mm = z_mm

            x_cam_m = 0.0
            y_cam_m = 0.0
            z_cam_m = 0.0
            if z_mm > 0 and self._cam_info is not None:
                K = np.array(self._cam_info.k, dtype=np.float32).reshape(3, 3)
                fx, fy = float(K[0, 0]), float(K[1, 1])
                cxK, cyK = float(K[0, 2]), float(K[1, 2])
                z_m = float(z_mm) * 0.001
                x_cam_m = (float(cx) - cxK) * z_m / fx
                y_cam_m = (float(cy) - cyK) * z_m / fy
                z_cam_m = z_m

            det.class_name = class_name
            det.x, det.y = x, y
            det.width, det.height = w, h
            det.center_x, det.center_y = cx, cy
            det.area = float(area) if area > 0 else (self._last_area if self._last_area > 0.0 else float(w * h))
            det.angle_deg = float(angle_deg)
            det.x_cam = x_cam_m
            det.y_cam = y_cam_m
            det.z_cam = z_cam_m
            det.score = float(self._last_score)
            det.source = source

            if draw:
                color = (0, 255, 0)   if source == "detect" else \
                        (0, 200, 255) if source == "track"  else \
                        (128, 128, 255)
                cv2.rectangle(rgb_bgr, (x, y), (x + w, y + h), color, 2)
                cv2.circle(rgb_bgr, (cx, cy), 5, color, -1)
                cv2.putText(rgb_bgr, f"{class_name} z={z_mm}mm [{source}]",
                            (x, max(0, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if quad_pts is not None and len(quad_pts) == 4:
                    cv2.polylines(rgb_bgr, [quad_pts.astype(int)], True, color, 2)
                if source == "detect" and angle_deg != 0.0:
                    cv2.putText(rgb_bgr, f"ang={angle_deg:.1f}",
                                (x, min(H - 5, y + h + 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        self._pub_det.publish(det)

        try:
            out        = self._bridge.cv2_to_imgmsg(rgb_bgr, encoding="bgr8")
            out.header = header
            self._pub_img.publish(out)
        except Exception:
            pass
