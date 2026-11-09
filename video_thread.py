from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from annotation import annotate_detections, make_canvas
from config import (
    CANDIDATE_COLOR,
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    FINAL_COLOR,
    FPS_WINDOW,
    FRAME_TIMEOUT_SECONDS,
    MAX_DRAWN_CANDIDATES,
    MODEL_PATH,
    NMS_ULTRALYTICS,
    TARGET_FPS,
    DetectionParameters,
    FrameResult,
)
from detection import inference, postprocess_detections
from detector import YoloDetector, confidence_filter
from nms_official import official_postprocess


def _empty_detections():
    return (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
    )


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    return QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()


def read_image(path: str):
    """Read an image file (works with non-ASCII paths on Windows). Returns BGR or None."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        return None


class VideoThread(QThread):
    """Capture frames (camera, image or video) and run YOLO + post-processing off the GUI thread.

    One thread lives for the whole session: the model is loaded once and the input source is
    switched with set_source() (no restart, no model reload).
    """

    frame_ready = pyqtSignal(object)  # FrameResult
    status_changed = pyqtSignal(str)
    processing_error = pyqtSignal(str)  # error raised by postprocess_detections
    source_error = pyqtSignal(str)  # input source problem ("" = no problem)
    model_changed = pyqtSignal(str, object)  # model path, class names
    model_failed = pyqtSignal(str)

    def __init__(
        self,
        camera_index: int = 0,
        source_type: str = "camera",
        file_path: str = "",
        model_path: str | Path = MODEL_PATH,
        parent=None,
    ):
        super().__init__(parent)
        self._camera_index = camera_index
        self._source_type = source_type
        self._file_path = file_path
        self._parameters: DetectionParameters | None = None
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._running.set()
        self._restart = threading.Event()
        self._wake = threading.Event()  # wakes the thread when it is idle (paused / still image)
        self._in_flight = threading.Event()  # a frame was sent and the GUI has not shown it yet
        self._in_flight_since = 0.0
        self._model_path = Path(model_path)
        self._pending_model: Path | None = self._model_path
        self._pending_source: tuple[str, str, int] | None = None
        self._dirty = True  # parameters or model changed since the last processed frame
        self._paused = False
        self._last_status = ""
        self._last_error = ""
        self._last_source_error = ""
        self._frame_times: deque[float] = deque(maxlen=FPS_WINDOW)

    # ------------------------------------------------------------------ GUI-facing API
    def set_parameters(self, parameters: DetectionParameters) -> None:
        with self._lock:
            if parameters != self._parameters:
                self._dirty = True
            self._parameters = parameters
        self._wake.set()

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._paused = paused
        self._wake.set()

    def set_model(self, model_path: str | Path) -> None:
        """Load another ONNX model (done inside the worker thread)."""
        with self._lock:
            self._pending_model = Path(model_path)
        self._wake.set()

    def set_source(self, source_type: str, file_path: str = "", camera_index: int = 0) -> None:
        """Switch to the webcam ('camera') or to an image/video file ('file'); un-pauses."""
        with self._lock:
            self._pending_source = (source_type, file_path, int(camera_index))
            self._paused = False
        self._wake.set()

    def restart_source(self) -> None:
        self._restart.set()
        self._wake.set()

    def frame_consumed(self) -> None:
        """Called by the GUI after it displayed a frame (back-pressure: no queue of stale frames)."""
        self._in_flight.clear()
        self._wake.set()

    def stop(self) -> None:
        self._running.clear()
        self._wake.set()
        self.wait()

    # ------------------------------------------------------------------ helpers
    def _snapshot(self):
        """Return (parameters, paused, dirty, pending_model, pending_source); clear the change flags."""
        with self._lock:
            dirty, model, source = self._dirty, self._pending_model, self._pending_source
            self._dirty = False
            self._pending_model = None
            self._pending_source = None
            return self._parameters, self._paused, dirty, model, source

    def _emit_status(self, message: str) -> None:
        if message != self._last_status:
            self._last_status = message
            self.status_changed.emit(message)

    def _emit_error(self, message: str) -> None:
        if message != self._last_error:
            self._last_error = message
            self.processing_error.emit(message)

    def _emit_source_error(self, message: str) -> None:
        if message != self._last_source_error:
            self._last_source_error = message
            self.source_error.emit(message)

    def _request_redraw(self) -> None:
        with self._lock:
            self._dirty = True

    def _gui_is_busy(self) -> bool:
        if not self._in_flight.is_set():
            return False
        return time.perf_counter() - self._in_flight_since < FRAME_TIMEOUT_SECONDS

    def _current_fps(self, live: bool) -> float:
        if not live or len(self._frame_times) < 2:
            return 0.0
        if time.perf_counter() - self._frame_times[-1] > 1.0:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    @staticmethod
    def _validate(result):
        boxes, scores, class_ids = result
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        class_ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)
        if not (len(boxes) == len(scores) == len(class_ids)):
            raise ValueError("postprocess_detections returned arrays of different lengths")
        return boxes, scores, class_ids

    def _open_source(self):
        """Return (capture, still_image). Exactly one of them is not None on success."""
        if self._source_type == "file":
            image = read_image(self._file_path)
            if image is not None:
                return None, image
            cap = cv2.VideoCapture(self._file_path)
        else:
            cap = cv2.VideoCapture(self._camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # low latency
        if not cap.isOpened():
            cap.release()
            return None, None
        return cap, None

    def _frame_period(self, cap) -> float:
        """Seconds per frame: the video's own rate for files, TARGET_FPS for the webcam."""
        if self._source_type == "file" and cap is not None:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if 1.0 < fps <= 120.0:
                return 1.0 / fps
        return 1.0 / TARGET_FPS

    # ------------------------------------------------------------------ processing
    def _process(self, detector, frame, parameters, live: bool) -> None:
        detector.set_class_filter(parameters.class_filter)
        processing_outputs: dict = {}
        timing = {"post_ms": 0.0}
        base = official_postprocess if parameters.nms_mode == NMS_ULTRALYTICS else postprocess_detections

        def timed_postprocess(*args):
            started = time.perf_counter()
            try:
                return base(*args)
            finally:
                timing["post_ms"] = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        try:
            final = self._validate(
                inference(
                    frame,
                    detector,
                    parameters.confidence_threshold,
                    parameters.nms_threshold,
                    processing_outputs,
                    timed_postprocess,
                )
            )
            self._emit_error("")
        except Exception as error:  # a bug in the student's code must not kill the app
            self._emit_error(f"{type(error).__name__}: {error}")
            final = _empty_detections()
        total_ms = (time.perf_counter() - started) * 1000.0
        post_ms = timing["post_ms"]

        proposals = processing_outputs.get("proposals", _empty_detections())
        candidates = confidence_filter(*proposals, parameters.confidence_threshold)

        # Draw on a canvas of "render size" so annotations look the same at any resolution.
        canvas, scale = make_canvas(frame)
        candidate_view = annotate_detections(
            canvas, *candidates, CANDIDATE_COLOR, parameters.show_labels, parameters.show_confidence,
            line_scale=0.5, text_scale=0.8, class_names=detector.class_names, scale=scale,
            max_boxes=MAX_DRAWN_CANDIDATES,
        )
        final_view = annotate_detections(
            canvas, *final, FINAL_COLOR, parameters.show_labels, parameters.show_confidence,
            class_names=detector.class_names, scale=scale,
        )
        if live:
            self._frame_times.append(time.perf_counter())
        height, width = frame.shape[:2]
        self._in_flight.set()
        self._in_flight_since = time.perf_counter()
        self.frame_ready.emit(
            FrameResult(
                candidate_view=bgr_to_qimage(candidate_view),
                final_view=bgr_to_qimage(final_view),
                num_candidates=len(candidates[1]),
                num_final=len(final[1]),
                mean_confidence=float(final[1].mean()) if len(final[1]) else 0.0,
                fps=self._current_fps(live),
                model_ms=max(0.0, total_ms - post_ms),
                postprocess_ms=post_ms,
                frame_width=width,
                frame_height=height,
            )
        )

    def _load_model(self, path: Path, current):
        """Try to load a model. Returns the new detector, or the current one if loading fails."""
        try:
            detector = YoloDetector(path)
        except Exception as error:  # missing file, not a YOLO model, missing onnxruntime, ...
            self.model_failed.emit(f"{path.name}: {error}")
            self._emit_status(f"Model error: {error}")
            return current
        self.model_changed.emit(str(path), list(detector.class_names))
        self._emit_status(f"Model loaded: {path.name}")
        return detector

    def run(self) -> None:
        detector = None
        cap = None
        still_image = None
        last_frame = None
        frame_processed = False
        was_paused = False
        frame_period = 1.0 / TARGET_FPS
        try:
            while self._running.is_set():
                self._wake.clear()  # clear BEFORE reading the state so no wake-up is lost
                if self._restart.is_set():
                    self._restart.clear()
                    if cap is not None:
                        cap.release()
                    cap, still_image, last_frame, frame_processed = None, None, None, False
                    self._frame_times.clear()

                parameters, paused, dirty, pending_model, pending_source = self._snapshot()
                if pending_source is not None:
                    self._source_type, self._file_path, self._camera_index = pending_source
                    if cap is not None:
                        cap.release()
                    cap, still_image, last_frame, frame_processed = None, None, None, False
                    self._frame_times.clear()
                    was_paused = False
                    self._emit_source_error("")
                    dirty = True
                if pending_model is not None:
                    detector = self._load_model(pending_model, detector)
                    dirty = True
                if parameters is None or detector is None:
                    self._wake.wait(0.2)  # nothing to do yet (no parameters or no valid model)
                    if detector is None:
                        self._request_redraw()
                    continue

                if cap is None and still_image is None:
                    cap, still_image = self._open_source()
                    if cap is None and still_image is None:
                        if self._source_type == "file":
                            name = Path(self._file_path).name
                            self._emit_source_error(f"Cannot open '{name}': not a readable image or video.")
                            self._emit_status("Source unavailable. Choose another file.")
                            self._wake.wait()  # no point in retrying: wait for the user
                        else:
                            self._emit_source_error(
                                f"Camera {self._camera_index} cannot be opened. "
                                "Check the camera index or choose an image/video file."
                            )
                            self._emit_status("Camera unavailable. Retrying...")
                            self._wake.wait(1.0)
                        continue
                    self._emit_source_error("")
                    self._emit_status("Source connected")
                    last_frame, frame_processed = None, False
                    frame_period = self._frame_period(cap)

                loop_started = time.perf_counter()
                if still_image is not None:
                    last_frame = still_image
                frozen = (paused or still_image is not None) and last_frame is not None

                if frozen:
                    # Paused (or a single image): do NOTHING unless something changed, and then
                    # process the same frame exactly once.
                    if paused:
                        self._emit_status("Paused")
                    was_paused = paused
                    if not dirty and frame_processed:
                        self._wake.wait()  # sleep until the GUI changes something
                        continue
                    if self._gui_is_busy():  # the GUI has not shown the last frame yet: retry soon
                        self._request_redraw()
                        self._wake.wait(0.02)
                        continue
                    frame = last_frame
                else:
                    if was_paused and cap is not None and self._source_type == "camera":
                        for _ in range(5):  # drop stale frames buffered while paused
                            cap.grab()
                    if was_paused:
                        self._emit_status("Source connected")
                    was_paused = False
                    ret, frame = cap.read()
                    if not ret:
                        if self._source_type == "file":  # video file: loop
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap.read()
                        if not ret:
                            self._emit_status("Frame read failed. Reconnecting...")
                            cap.release()
                            cap = None
                            self._wake.wait(0.25)
                            continue
                    last_frame = frame
                    if self._gui_is_busy():  # drop this frame instead of queueing stale ones
                        time.sleep(0.005)
                        continue

                self._process(detector, frame, parameters, live=not frozen)
                frame_processed = True
                if not frozen:
                    elapsed = time.perf_counter() - loop_started
                    time.sleep(max(0.0, frame_period - elapsed))
        finally:
            if cap is not None:
                cap.release()
            self._emit_status("Source stopped")
