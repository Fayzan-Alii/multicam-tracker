from __future__ import annotations

import cv2
import numpy as np

from multicam_tracker.viz.colors import id_color


def draw_boxes(image: np.ndarray, items, prefix: str = "", trails: dict | None = None) -> np.ndarray:
    """items: iterable of (identity, box). trails: identity -> list of foot pixels to draw."""
    out = image.copy()
    for ident, pts in (trails or {}).items():
        if len(pts) >= 2:
            cv2.polylines(out, [np.round(np.asarray(pts)).astype(np.int32)], False, id_color(ident), 2)
    for ident, box in items:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        color = id_color(ident)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{prefix}{ident}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return out


def label_panel(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.putText(out, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5)
    cv2.putText(out, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    return out
