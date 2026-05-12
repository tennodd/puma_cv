from .helpers import make_tracker


class _TrackerMixin:
    def _tracker_init(self, frame, bbox):
        self._tracker = make_tracker()
        if self._tracker is None:
            self._tracking = False
            return False
        x, y, w, h     = bbox
        self._tracking = bool(
            self._tracker.init(frame, (float(x), float(y), float(w), float(h)))
        )
        return self._tracking

    def _tracker_update(self, frame):
        if not self._tracking or self._tracker is None:
            return False, None
        ok, box = self._tracker.update(frame)
        if not ok:
            self._tracking = False
            self._tracker  = None
            return False, None
        return True, tuple(map(int, box))
