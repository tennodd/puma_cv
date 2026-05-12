import cv2
import numpy as np


class _SaliencyMixin:
    def _build_table_mask(self, depth_mm):
        with self._plane_lock:
            if self._cam_info is None or self._plane_smoothed is None:
                return None
            plane = self._plane_smoothed.copy()
        K = np.array(self._cam_info.k, dtype=np.float32).reshape(3, 3)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        Z = depth_mm.astype(np.float32) * 0.001  # mm -> m
        valid = Z > 0.0

        H, W = depth_mm.shape
        uu, vv = np.meshgrid(
            np.arange(W, dtype=np.float32),
            np.arange(H, dtype=np.float32),
        )
        X = (uu - cx) * Z / fx
        Y = (vv - cy) * Z / fy

        a, b, c, d = plane
        signed = a * X + b * Y + c * Z + d  # meters

        band = float(self.get_parameter("plane_band_m").value)
        on_plane = ((np.abs(signed) <= band) & valid).astype(np.uint8) * 255

        n, labels, stats, _ = cv2.connectedComponentsWithStats(on_plane, connectivity=8)
        if n <= 1:
            return None

        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.size == 0:
            return None
        largest_idx = int(np.argmax(areas)) + 1
        largest_area = int(stats[largest_idx, cv2.CC_STAT_AREA])

        min_table_px = int(self.get_parameter("min_table_area_px").value)
        if largest_area < min_table_px:
            return None

        table_mask = np.where(labels == largest_idx, 255, 0).astype(np.uint8)

        contours, _ = cv2.findContours(
            table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            filled = np.zeros_like(table_mask)
            cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
            table_mask = filled

        erode_px = int(self.get_parameter("table_mask_erode_px").value)
        if erode_px > 0:
            k = 2 * erode_px + 1
            table_mask = cv2.erode(table_mask, np.ones((k, k), np.uint8))

        return table_mask

    def _build_saliency_mask(self, rgb_bgr, depth_mm, table_mask, return_diag=False):
        H_t, W_t = table_mask.shape
        dE_arr   = np.zeros((H_t, W_t), dtype=np.float32)
        bump_arr = np.zeros((H_t, W_t), dtype=np.float32)
        A_t      = 128.0
        B_t      = 128.0

        use_clahe = bool(self.get_parameter("use_clahe").value)

        ksize = int(self.get_parameter("rgb_blur_ksize").value)
        if ksize > 0 and ksize % 2 == 1:
            bgr = cv2.GaussianBlur(rgb_bgr, (ksize, ksize), 0)
        else:
            bgr = rgb_bgr
        if use_clahe:
            lab_pre = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            l_pre, a_pre, b_pre = cv2.split(lab_pre)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_eq = clahe.apply(l_pre)
            bgr = cv2.cvtColor(cv2.merge((l_eq, a_pre, b_pre)), cv2.COLOR_LAB2BGR)

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[..., 0]
        A = lab[..., 1]
        B = lab[..., 2]

        L_min = float(self.get_parameter("valid_L_min").value)
        L_max = float(self.get_parameter("valid_L_max").value)
        valid = (L >= L_min) & (L <= L_max) & (table_mask > 0)

        if np.any(valid):
            L_t = float(np.median(L[valid]))
            A_t = float(np.median(A[valid]))
            B_t = float(np.median(B[valid]))

            dE = np.sqrt(
                (A - A_t) ** 2 + (B - B_t) ** 2
            )
            dE_arr = dE.astype(np.float32, copy=False)

        valid_d = np.zeros_like(table_mask, dtype=bool)
        with self._plane_lock:
            plane_snapshot = (
                self._plane_smoothed.copy()
                if (self._cam_info is not None and self._plane_smoothed is not None)
                else None
            )
        if plane_snapshot is not None:
            K = np.array(self._cam_info.k, dtype=np.float32).reshape(3, 3)
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            Z = depth_mm.astype(np.float32) * 0.001  # mm -> m
            valid_d = Z > 0.0

            H, W = depth_mm.shape
            uu, vv = np.meshgrid(
                np.arange(W, dtype=np.float32),
                np.arange(H, dtype=np.float32),
            )
            X = (uu - cx) * Z / fx
            Y = (vv - cy) * Z / fy

            a, b, c, d = plane_snapshot
            signed = a * X + b * Y + c * Z + d  # meters
            bump_arr = signed.astype(np.float32, copy=False)

        de_thr   = float(self.get_parameter("delta_e_thresh").value)
        bump_thr = float(self.get_parameter("depth_bump_thresh_m").value)

        dE_strength = (dE_arr / max(de_thr, 1e-6)).astype(np.float32, copy=False)
        dE_strength = np.where(valid, dE_strength, 0.0).astype(np.float32, copy=False)

        bump_in_table = (table_mask > 0) & valid_d
        bump_strength = (bump_arr / max(bump_thr, 1e-6)).astype(np.float32, copy=False)
        bump_strength = np.where(bump_in_table, bump_strength, 0.0).astype(np.float32, copy=False)

        raw_strength = np.maximum(dE_strength, bump_strength)

        de_fired   = dE_strength   > 1.0
        bump_fired = bump_strength > 1.0
        debug_channels = np.zeros((dE_strength.shape[0], dE_strength.shape[1], 3),
                                  dtype=np.uint8)
        both      = de_fired & bump_fired
        de_only   = de_fired & ~bump_fired
        bump_only = bump_fired & ~de_fired
        debug_channels[de_only]   = (0,   0,   255)   # red    (BGR)
        debug_channels[bump_only] = (255, 0,   0)     # blue
        debug_channels[both]      = (255, 255, 255)   # white
        try:
            self._pub_debug_channels.publish(
                self._bridge.cv2_to_imgmsg(debug_channels, encoding="bgr8"))
        except Exception:
            pass

        if (self._saliency_smoothed is None
                or self._saliency_smoothed.shape != raw_strength.shape):
            self._saliency_smoothed = raw_strength.copy()
        else:
            alpha = float(self.get_parameter("saliency_lpf_alpha").value)
            self._saliency_smoothed = (
                alpha * raw_strength + (1.0 - alpha) * self._saliency_smoothed
            ).astype(np.float32, copy=False)

        salient = (self._saliency_smoothed > 1.0).astype(np.uint8) * 255

        med_ksize = int(self.get_parameter("saliency_median_ksize").value)
        if med_ksize > 0 and med_ksize % 2 == 1:
            salient = cv2.medianBlur(salient, med_ksize)

        erode_px = int(self.get_parameter("depth_bump_erode_px").value)
        if erode_px > 0:
            k = 2 * erode_px + 1
            salient = cv2.erode(salient, np.ones((k, k), np.uint8))

        if return_diag:
            return salient, dE_arr, bump_arr, A_t, B_t
        return salient

    def _clean_mask(self, mask):
        ok = int(self.get_parameter("morph_open").value)
        ck = int(self.get_parameter("morph_close").value)
        ok = max(1, ok + (ok % 2 == 0))
        ck = max(1, ck + (ck % 2 == 0))
        m  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((ok, ok), np.uint8))
        m  = cv2.morphologyEx(m,    cv2.MORPH_CLOSE, np.ones((ck, ck), np.uint8))
        return m
