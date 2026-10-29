from __future__ import annotations

import numpy as np


def inference(
    frame: np.ndarray,
    detector,
    confidence_threshold: float,
    nms_threshold: float,
    processing_outputs: dict,
    postprocess=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Already implemented. Do NOT modify.

    Runs the pretrained YOLO model, then hands all raw proposals to postprocess_detections.
    """
    boxes, scores, class_ids = detector.predict(frame)  # e.g. 25200 raw proposals
    processing_outputs["proposals"] = (boxes, scores, class_ids)  # for the GUI only
    postprocess = postprocess or postprocess_detections  # GUI may swap in the official NMS
    return postprocess(boxes, scores, class_ids, confidence_threshold, nms_threshold)


def postprocess_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    confidence_threshold: float,
    nms_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    TODO: Implement confidence filtering and Non-Maximum Suppression (NMS).

    IMPORTANT:

    - This is the only file students should modify.
    - Keep both function signatures unchanged and do not modify inference().
    - Do not add other top-level functions or classes to this file; if you want a helper
      (for example an IoU function), define it inside postprocess_detections.
    - Do not draw anything and do not use cv2.dnn.NMSBoxes, torchvision.ops.nms or
      ultralytics NMS here. The point of the lab is to write NMS yourself with NumPy.

    Inputs (one entry per raw model proposal, e.g. 25200 of them):

        boxes:      float32 array, shape (N, 4), pixel coordinates (x1, y1, x2, y2)
        scores:     float32 array, shape (N,), confidence of each proposal
                    (objectness x best class probability)
        class_ids:  int array, shape (N,), index of the best class of each proposal
        confidence_threshold: float in [0, 1]
        nms_threshold:        IoU threshold in [0, 1]

    Return exactly this tuple, with the surviving detections only:

        (final_boxes, final_scores, final_class_ids)
        shapes (M, 4), (M,), (M,) and the same dtypes/xyxy format as the inputs.

    Expected implementation sequence:

    1. Confidence filtering: keep only proposals with score >= confidence_threshold.
    2. NMS is applied separately for every class (use np.unique(class_ids)), so a "person"
       box never suppresses a "chair" box.
    3. For one class: sort its boxes by score, highest first.
    4. Repeat until no boxes are left:
       a. Take the highest-scoring remaining box and keep it as a final detection.
       b. Compute the IoU between that box and all remaining boxes of the same class
          (vectorized with NumPy: intersection = overlap width * overlap height,
          union = area_a + area_b - intersection).
       c. Discard every remaining box whose IoU is greater than nms_threshold.
    5. Collect the kept boxes of all classes and return them as the three arrays.

    Edge cases to handle: no proposal passes the threshold (return empty arrays with the
    shapes (0, 4), (0,), (0,)), and a class with a single box.

    Keep the implementation deterministic and free of global state or GUI code.
    """
    return (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
    )
