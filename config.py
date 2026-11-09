from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

APP_NAME = "CVML Lab 07 - YOLO Detection"
ORGANIZATION_NAME = "Sharif CVML Lab"

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "assets" / "yolov5s.onnx"
INPUT_SIZE = 640  # 640x640 input: 8400 proposals (anchor-free export) or 25200 (classic YOLOv5 export)

CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
TARGET_FPS = 30

DEFAULT_CAMERA_INDEX = 0
DEFAULT_THEME = "dark"

NMS_STUDENT = "Student"
NMS_ULTRALYTICS = "Ultralytics"

DEFAULT_NMS_MODE = NMS_STUDENT
DEFAULT_CONFIDENCE_PERCENT = 25
DEFAULT_NMS_IOU_PERCENT = 45
MIN_CONFIDENCE_PERCENT = 5  # a very low threshold makes a pure-Python NMS slow
DEFAULT_SHOW_LABELS = True
DEFAULT_SHOW_CONFIDENCE = True

DEFAULT_SETTINGS_MESSAGE = "Settings: Ready"

CANDIDATE_COLOR = (0, 165, 255)  # BGR, orange
FINAL_COLOR = (60, 220, 140)     # BGR, green

# Annotation / display. Frames are brought to a "render size" before anything is drawn, so
# boxes, text and labels look the same for a 160x120 thumbnail and a 4000x3000 photo.
MIN_RENDER_SIDE = 640    # frames whose longest side is smaller are enlarged to this size
MAX_RENDER_SIDE = 1600   # frames whose longest side is larger are reduced to this size
MAX_DRAWN_CANDIDATES = 300  # at most this many (highest-score) candidates are drawn

FPS_WINDOW = 10               # frames used for the running FPS estimate
FRAME_TIMEOUT_SECONDS = 1.0   # a frame the GUI has not shown after this long is re-sent

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")

FILE_FILTER = (
    "Media Files (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff "
    "*.mp4 *.avi *.mov *.mkv *.webm *.m4v);;"
    "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;"
    "Videos (*.mp4 *.avi *.mov *.mkv *.webm *.m4v);;"
    "All Files (*)"
)

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


@dataclass
class FrameResult:
    """Everything the GUI needs to show one processed frame (sent from the video thread)."""

    candidate_view: object = None  # QImage: candidates (after confidence filter, before NMS)
    final_view: object = None      # QImage: final detections (after NMS)
    num_candidates: int = 0
    num_final: int = 0
    mean_confidence: float = 0.0
    fps: float = 0.0               # 0.0 = frozen (pause / still image)
    model_ms: float = 0.0          # YOLO network time
    postprocess_ms: float = 0.0    # confidence filtering + NMS time
    frame_width: int = 0
    frame_height: int = 0


@dataclass(frozen=True)
class DetectionParameters:
    confidence_threshold: float
    nms_threshold: float
    nms_mode: str
    class_filter: tuple[int, ...] | None  # None = all classes
    show_labels: bool
    show_confidence: bool
