"""Drawing of detections (fixed infrastructure, students do not modify this file).

Resolution independence
-----------------------
OpenCV draws with sizes in *pixels*. If a 4000x3000 photo is later shrunk to fit a 600 px
wide view, a 2 px line becomes invisible; a 160x120 thumbnail enlarged to the same view
makes the same line huge. Therefore every frame is first brought to a "render size"
(longest side between MIN_RENDER_SIDE and MAX_RENDER_SIDE, see make_canvas) and line width,
font size and label padding are then derived from that canvas size. The annotated image
always looks the same relative to its content, whatever the resolution of the source.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import COCO_CLASSES, MAX_RENDER_SIDE, MIN_RENDER_SIDE

FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_TEXT_COLOR = (20, 20, 20)  # dark text on the coloured label background


def make_canvas(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (canvas, scale): `frame` resized so its longest side is within the render range.

    scale = canvas size / frame size; multiply frame coordinates by it to get canvas coordinates.
    The input is returned unchanged (scale 1.0) when it is already inside the range.
    """
    height, width = frame.shape[:2]
    side = max(height, width)
    target = min(max(side, MIN_RENDER_SIDE), MAX_RENDER_SIDE)
    if target == side:
        return frame, 1.0
    scale = target / side
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, size, interpolation=interpolation), scale


@dataclass(frozen=True)
class AnnotationStyle:
    box_thickness: int
    font_scale: float
    font_thickness: int


def annotation_style(canvas_shape, line_scale: float = 1.0, text_scale: float = 1.0) -> AnnotationStyle:
    """Sizes proportional to the canvas: at 640 px -> 2 px boxes and a 0.5 font scale."""
    side = max(canvas_shape[:2])
    font_scale = side / 1250.0 * text_scale
    return AnnotationStyle(
        box_thickness=max(1, round(side * 0.0035 * line_scale)),
        font_scale=font_scale,
        font_thickness=max(1, round(font_scale * 2.0)),
    )


def _class_name(class_id: int, class_names) -> str:
    names = class_names if class_names is not None else COCO_CLASSES
    return names[class_id] if 0 <= class_id < len(names) else f"class {class_id}"


def _draw_label(image, text, x1, y1, color, style: AnnotationStyle) -> None:
    """Filled label above the box (inside the box if there is no room above)."""
    height, width = image.shape[:2]
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, style.font_scale, style.font_thickness)
    pad = max(2, round(text_h * 0.35))
    tag_w, tag_h = text_w + 2 * pad, text_h + baseline + 2 * pad
    x = int(min(max(0, x1), max(0, width - tag_w)))
    y = int(y1 - tag_h)
    if y < 0:
        y = int(min(max(0, y1), max(0, height - tag_h)))
    cv2.rectangle(image, (x, y), (x + tag_w, y + tag_h), color, -1)
    cv2.putText(
        image, text, (x + pad, y + pad + text_h), FONT, style.font_scale,
        LABEL_TEXT_COLOR, style.font_thickness, cv2.LINE_AA,
    )


def annotate_detections(
    canvas: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    color: tuple[int, int, int],
    show_labels: bool = True,
    show_confidence: bool = True,
    line_scale: float = 1.0,
    text_scale: float = 1.0,
    class_names=None,
    scale: float = 1.0,
    max_boxes: int | None = None,
) -> np.ndarray:
    """Draw xyxy detections on a copy of `canvas` (the frame from make_canvas).

    boxes are in the coordinates of the ORIGINAL frame; `scale` (from make_canvas) maps them
    to the canvas. If max_boxes is given only that many highest-score boxes are drawn, and the
    best ones are drawn last so they stay on top.
    """
    output = canvas.copy()
    count = len(boxes)
    if count == 0:
        return output
    order = np.argsort(scores)[::-1]
    if max_boxes is not None:
        order = order[:max_boxes]
    style = annotation_style(output.shape, line_scale, text_scale)

    for index in order[::-1]:  # lowest score first -> best detection ends up on top
        x1, y1, x2, y2 = (int(round(float(v) * scale)) for v in boxes[index])
        cv2.rectangle(output, (x1, y1), (x2, y2), color, style.box_thickness)
        parts = []
        if show_labels:
            parts.append(_class_name(int(class_ids[index]), class_names))
        if show_confidence:
            parts.append(f"{float(scores[index]):.2f}")
        if parts:
            _draw_label(output, " ".join(parts), x1, y1, color, style)
    return output
