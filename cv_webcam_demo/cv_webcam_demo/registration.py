import numpy as np


class _RegistrationMixin:
    def _register_depth(self, depth_mm):
        if self._cam_info is None or self._depth_cam_info is None:
            return None

        H, W = depth_mm.shape
        Kd = np.array(self._depth_cam_info.k, dtype=np.float32).reshape(3, 3)
        Kr = np.array(self._cam_info.k,       dtype=np.float32).reshape(3, 3)
        fx_d, fy_d, cx_d, cy_d = Kd[0, 0], Kd[1, 1], Kd[0, 2], Kd[1, 2]
        fx_r, fy_r, cx_r, cy_r = Kr[0, 0], Kr[1, 1], Kr[0, 2], Kr[1, 2]
        tx = float(self.get_parameter("depth_to_rgb_x").value)
        ty = float(self.get_parameter("depth_to_rgb_y").value)
        tz = float(self.get_parameter("depth_to_rgb_z").value)

        valid = depth_mm > 0
        if not np.any(valid):
            return np.zeros_like(depth_mm)

        v_d, u_d = np.where(valid)
        Z = depth_mm[v_d, u_d].astype(np.float32) * 0.001  # mm -> m

        X = (u_d.astype(np.float32) - cx_d) * Z / fx_d
        Y = (v_d.astype(np.float32) - cy_d) * Z / fy_d

        Xr = X - tx
        Yr = Y - ty
        Zr = Z - tz

        front = Zr > 1e-6
        Xr, Yr, Zr = Xr[front], Yr[front], Zr[front]

        u_r = (fx_r * Xr / Zr + cx_r + 0.5).astype(np.int32)
        v_r = (fy_r * Yr / Zr + cy_r + 0.5).astype(np.int32)

        in_bounds = (u_r >= 0) & (u_r < W) & (v_r >= 0) & (v_r < H)
        u_r, v_r, Zr = u_r[in_bounds], v_r[in_bounds], Zr[in_bounds]

        Z_mm = (Zr * 1000.0).astype(np.uint16)

        flat_idx = v_r.astype(np.int64) * W + u_r.astype(np.int64)
        order = np.lexsort((Z_mm, flat_idx))
        flat_sorted = flat_idx[order]
        Z_sorted    = Z_mm[order]

        unique_idx, first_idx = np.unique(flat_sorted, return_index=True)
        out_flat = np.zeros(H * W, dtype=np.uint16)
        out_flat[unique_idx] = Z_sorted[first_idx]
        return out_flat.reshape(H, W)
