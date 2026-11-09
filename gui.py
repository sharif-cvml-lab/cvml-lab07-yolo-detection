from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSettings, QSize, Qt
from PyQt6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStyle,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from config import (
    APP_NAME,
    COCO_CLASSES,
    DEFAULT_CAMERA_INDEX,
    DEFAULT_CONFIDENCE_PERCENT,
    DEFAULT_NMS_IOU_PERCENT,
    DEFAULT_NMS_MODE,
    DEFAULT_SETTINGS_MESSAGE,
    DEFAULT_SHOW_CONFIDENCE,
    DEFAULT_SHOW_LABELS,
    DEFAULT_THEME,
    FILE_FILTER,
    FrameResult,
    MODEL_PATH,
    MIN_CONFIDENCE_PERCENT,
    NMS_STUDENT,
    NMS_ULTRALYTICS,
    ORGANIZATION_NAME,
    DetectionParameters,
)
from video_thread import VideoThread


DARK_STYLE = """
QMainWindow, QWidget {
    background: #11151b;
    color: #e7ebf0;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}
QFrame#Header, QFrame#SidebarCard, QFrame#VideoCard, QFrame#StatusBar {
    background: #171c23;
    border: 1px solid #28303a;
    border-radius: 12px;
}
QFrame#VideoCanvas {
    background: #0a0d11;
    border: 1px solid #303945;
    border-radius: 10px;
}
QLabel#Title { font-size: 22px; font-weight: 700; }
QLabel#Subtitle, QLabel#Muted { color: #8e98a6; }
QLabel#PanelTitle { font-size: 14px; font-weight: 700; }
QLabel#Metric { color: #9aa6b4; font-size: 12px; }
QLabel#MetricValue { font-size: 15px; font-weight: 700; }
QLabel#ValueChip {
    background: #202833;
    border: 1px solid #364250;
    border-radius: 6px;
    padding: 2px 8px;
    min-width: 42px;
}
QPushButton {
    background: #212833;
    color: #eef2f6;
    border: 1px solid #35404d;
    border-radius: 8px;
    padding: 8px 12px;
}
QPushButton:hover { background: #2a3441; border-color: #4e6175; }
QPushButton:pressed { background: #18202a; }
QPushButton#accent { background: #3a7d67; border-color: #4c9b80; font-weight: 700; }
QPushButton#accent:hover { background: #479276; }
QPushButton#danger { background: #322126; border-color: #5d333d; }
QSpinBox {
    background: #1a2028;
    border: 1px solid #35404d;
    border-radius: 7px;
    padding: 6px;
}
QSlider::groove:horizontal { height: 5px; background: #2b333d; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #4d9d80; border-radius: 3px; }
QSlider::handle:horizontal {
    background: #80c6ab;
    border: 1px solid #9ad9c0;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QGroupBox {
    border: 1px solid #28303a;
    border-radius: 10px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: 700;
}
QGroupBox::title { left: 12px; padding: 0 6px; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { width: 10px; background: transparent; }
QScrollBar::handle:vertical { background: #313b47; border-radius: 5px; min-height: 30px; }
QLabel#Error { color: #ff7b7b; font-weight: 700; }
QCheckBox { spacing: 8px; }
QListWidget, QComboBox {
    background: #1a2028;
    border: 1px solid #35404d;
    border-radius: 7px;
    padding: 4px;
}
"""

LIGHT_STYLE = """
QMainWindow, QWidget {
    background: #eef1f5;
    color: #1d252d;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}
QFrame#Header, QFrame#SidebarCard, QFrame#VideoCard, QFrame#StatusBar {
    background: #ffffff;
    border: 1px solid #d3dae2;
    border-radius: 12px;
}
QFrame#VideoCanvas {
    background: #dfe5ec;
    border: 1px solid #c4ccd6;
    border-radius: 10px;
}
QLabel#Title { font-size: 22px; font-weight: 700; }
QLabel#Subtitle, QLabel#Muted { color: #5f6b78; }
QLabel#PanelTitle { font-size: 14px; font-weight: 700; }
QLabel#Metric { color: #5f6b78; font-size: 12px; }
QLabel#MetricValue { font-size: 15px; font-weight: 700; }
QLabel#Error { color: #c0392b; font-weight: 700; }
QLabel#ValueChip {
    background: #e8edf2;
    border: 1px solid #c4ccd6;
    border-radius: 6px;
    padding: 2px 8px;
    min-width: 42px;
}
QPushButton {
    background: #f7f9fb;
    color: #1d252d;
    border: 1px solid #c4ccd6;
    border-radius: 8px;
    padding: 8px 12px;
}
QPushButton:hover { background: #e8eef4; border-color: #8fa3b5; }
QPushButton:pressed { background: #dbe3ea; }
QPushButton#accent { background: #3a7d67; color: #ffffff; border-color: #4c9b80; font-weight: 700; }
QPushButton#accent:hover { background: #479276; }
QPushButton#danger { background: #f8e3e6; border-color: #e0b4bc; }
QSpinBox, QListWidget, QComboBox {
    background: #ffffff;
    border: 1px solid #c4ccd6;
    border-radius: 7px;
    padding: 6px;
}
QSlider::groove:horizontal { height: 5px; background: #cdd5de; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #4d9d80; border-radius: 3px; }
QSlider::handle:horizontal {
    background: #3a7d67;
    border: 1px solid #2c614f;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QGroupBox {
    border: 1px solid #d3dae2;
    border-radius: 10px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: 700;
}
QGroupBox::title { left: 12px; padding: 0 6px; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { width: 10px; background: transparent; }
QScrollBar::handle:vertical { background: #bcc6d1; border-radius: 5px; min-height: 30px; }
"""


class VideoLabel(QLabel):
    """Scalable image view that preserves the source aspect ratio."""

    def __init__(self):
        super().__init__()
        self._pixmap = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(240, 180)
        # Ignored: the scaled pixmap must never become the minimum size of the view,
        # otherwise the window could not shrink again after a large frame was shown.
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def set_frame(self, image: QImage) -> None:
        self._pixmap = QPixmap.fromImage(image)
        self._refresh_pixmap()

    def clear_frame(self) -> None:
        self._pixmap = QPixmap()
        self.clear()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._pixmap.isNull():
            return
        self.setPixmap(
            self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class ValueSlider(QWidget):
    def __init__(self, title: str, minimum: int, maximum: int, value: int, parent=None):
        super().__init__(parent)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)
        self.title_label = QLabel(title)
        self.title_label.setMinimumWidth(120)
        self.value_label = QLabel(str(value))
        self.value_label.setObjectName("ValueChip")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)
        layout.addWidget(self.title_label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)
        self.slider.valueChanged.connect(self._on_value_changed)

    def _on_value_changed(self, value: int) -> None:
        self.value_label.setText(str(value))

    def value(self) -> int:
        return self.slider.value()

    def set_value(self, value: int) -> None:
        self.slider.setValue(value)


class AppWindow(QMainWindow):
    """Main interactive UI for Lab 07."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(QSize(1120, 740))
        self.resize(1500, 900)

        self.settings = QSettings(ORGANIZATION_NAME, APP_NAME)
        self.video_thread: VideoThread | None = None
        self.selected_file = ""
        self.source_type = "camera"
        self.theme = DEFAULT_THEME
        self.nms_mode = DEFAULT_NMS_MODE
        self.model_path = str(MODEL_PATH)  # the model that is (or will be) loaded
        self._closing = False
        self._restoring = False

        self._build_ui()
        self._restore_settings()
        self._build_shortcuts()
        self._apply_theme()
        self._start_video_thread()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 12)
        root.setSpacing(12)

        header = QFrame()
        header.setObjectName("Header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 14, 18, 14)
        title_box = QVBoxLayout()
        title = QLabel("YOLO Detection Playground")
        title.setObjectName("Title")
        subtitle = QLabel("Interactive confidence thresholding and Non-Maximum Suppression")
        subtitle.setObjectName("Subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_layout.addLayout(title_box, 1)
        self.theme_button = QPushButton("Light Mode")
        self.theme_button.clicked.connect(self._toggle_theme)
        header_layout.addWidget(self.theme_button)
        self.reset_button = QPushButton("Reset")
        self.reset_button.setObjectName("danger")
        self.reset_button.clicked.connect(self._reset_defaults)
        header_layout.addWidget(self.reset_button)
        root.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        video_side = QWidget()
        video_layout = QVBoxLayout(video_side)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_grid = QGridLayout()
        video_grid.setSpacing(10)
        self.candidate_panel = self._create_video_card("Candidate Detections (before NMS)")
        self.final_panel = self._create_video_card("Final Detections (after NMS)")
        video_grid.addWidget(self.candidate_panel, 0, 0)
        video_grid.addWidget(self.final_panel, 0, 1)
        for column in range(2):
            video_grid.setColumnStretch(column, 1)
        video_layout.addLayout(video_grid, 1)

        status = QFrame()
        status.setObjectName("StatusBar")
        status_layout = QHBoxLayout(status)
        status_layout.setContentsMargins(14, 8, 14, 8)
        self.status_label = QLabel("Starting source...")
        self.status_label.setObjectName("Muted")
        self.error_label = QLabel("")
        self.error_label.setObjectName("Error")
        self.settings_label = QLabel(DEFAULT_SETTINGS_MESSAGE)
        self.settings_label.setObjectName("MetricValue")
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setObjectName("MetricValue")
        status_layout.addWidget(self.status_label, 1)
        status_layout.addWidget(self.error_label)
        status_layout.addSpacing(18)
        status_layout.addWidget(self.settings_label)
        status_layout.addSpacing(18)
        status_layout.addWidget(self.fps_label)
        video_layout.addWidget(status)

        splitter.addWidget(video_side)
        splitter.addWidget(self._build_sidebar())
        splitter.setSizes([1140, 380])
        self.splitter = splitter
        root.addWidget(splitter, 1)

        self.setCentralWidget(central)

    def _create_video_card(self, title: str) -> QFrame:
        card = QFrame()
        card.setObjectName("VideoCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        header = QHBoxLayout()
        title_label = QLabel(title)
        title_label.setObjectName("PanelTitle")
        count_label = QLabel("--")
        count_label.setObjectName("ValueChip")
        count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(title_label, 1)
        header.addWidget(count_label)
        layout.addLayout(header)
        canvas = QFrame()
        canvas.setObjectName("VideoCanvas")
        canvas_layout = QVBoxLayout(canvas)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        image = VideoLabel()
        image.setText("Waiting for source...")
        canvas_layout.addWidget(image)
        layout.addWidget(canvas, 1)
        card.image = image
        card.count_label = count_label
        return card

    def _build_sidebar(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(10)

        model_box = QGroupBox("Model")
        model_layout = QHBoxLayout(model_box)
        self.model_label = QLabel(Path(self.model_path).name)
        self.model_label.setObjectName("MetricValue")
        self.model_label.setToolTip(self.model_path)
        self.model_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.model_button = QPushButton()
        self.model_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.model_button.setToolTip("Browse for an ONNX model on this computer")
        self.model_button.setFixedWidth(46)
        self.model_button.clicked.connect(self._browse_model)
        model_layout.addWidget(self.model_label, 1)
        model_layout.addWidget(self.model_button)
        layout.addWidget(model_box)

        source_box = QGroupBox("Input Source")
        source_layout = QVBoxLayout(source_box)
        camera_row = QHBoxLayout()
        camera_row.addWidget(QLabel("Camera index"))
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 9)
        self.camera_spin.valueChanged.connect(self._camera_index_changed)
        camera_row.addWidget(self.camera_spin)
        source_layout.addLayout(camera_row)

        camera_buttons = QHBoxLayout()
        self.webcam_button = QPushButton("Webcam")
        self.webcam_button.clicked.connect(self._use_webcam)
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self._browse_file)
        self.upload_button = QPushButton("Upload")
        self.upload_button.setObjectName("accent")
        self.upload_button.setEnabled(False)
        self.upload_button.clicked.connect(self._upload_selected_file)
        camera_buttons.addWidget(self.webcam_button)
        camera_buttons.addWidget(self.browse_button)
        camera_buttons.addWidget(self.upload_button)
        source_layout.addLayout(camera_buttons)

        self.file_label = QLabel("No offline image/video selected")
        self.file_label.setObjectName("Muted")
        self.file_label.setWordWrap(True)
        source_layout.addWidget(self.file_label)

        action_row = QHBoxLayout()
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self._toggle_pause)
        self.restart_button = QPushButton("Restart")
        self.restart_button.clicked.connect(self._restart_source)
        action_row.addWidget(self.pause_button)
        action_row.addWidget(self.restart_button)
        source_layout.addLayout(action_row)
        layout.addWidget(source_box)

        detection_box = QGroupBox("Detection Configuration")
        detection_layout = QVBoxLayout(detection_box)
        self.confidence_slider = ValueSlider(
            "Confidence ×100", MIN_CONFIDENCE_PERCENT, 95, DEFAULT_CONFIDENCE_PERCENT
        )
        self.nms_slider = ValueSlider("NMS IoU ×100", 5, 95, DEFAULT_NMS_IOU_PERCENT)
        detection_layout.addWidget(self.confidence_slider)
        detection_layout.addWidget(self.nms_slider)
        self.nms_mode_button = QPushButton()
        self.nms_mode_button.clicked.connect(self._toggle_nms_mode)
        detection_layout.addWidget(self.nms_mode_button)
        self.labels_check = QCheckBox("Show labels")
        self.confidence_check = QCheckBox("Show confidence")
        detection_layout.addWidget(self.labels_check)
        detection_layout.addWidget(self.confidence_check)
        layout.addWidget(detection_box)

        class_box = QGroupBox("Class Filter")
        class_layout = QVBoxLayout(class_box)
        self.class_list = QListWidget()
        self.class_list.setMinimumHeight(170)
        for name in COCO_CLASSES:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.class_list.addItem(item)
        class_layout.addWidget(self.class_list)
        class_buttons = QHBoxLayout()
        self.all_classes_button = QPushButton("All")
        self.all_classes_button.clicked.connect(lambda: self._set_all_classes(True))
        self.no_classes_button = QPushButton("None")
        self.no_classes_button.clicked.connect(lambda: self._set_all_classes(False))
        class_buttons.addWidget(self.all_classes_button)
        class_buttons.addWidget(self.no_classes_button)
        class_layout.addLayout(class_buttons)

        stats_box = QGroupBox("Statistics")
        stats_layout = QGridLayout(stats_box)
        self.metric_values: dict[str, QLabel] = {}
        for row, name in enumerate(
            (
                "Candidates", "Final detections", "Mean confidence", "FPS",
                "Frame size", "Model (ms)", "Post-processing (ms)",
            )
        ):
            label = QLabel(name)
            label.setObjectName("Metric")
            value = QLabel("--")
            value.setObjectName("MetricValue")
            stats_layout.addWidget(label, row, 0)
            stats_layout.addWidget(value, row, 1)
            self.metric_values[name] = value
        layout.addWidget(stats_box)  # before the long class list, so it is visible without scrolling
        layout.addWidget(class_box)
        layout.addStretch(1)

        for control in (self.confidence_slider, self.nms_slider):
            control.slider.valueChanged.connect(self._parameters_changed)
        for check in (self.labels_check, self.confidence_check):
            check.toggled.connect(self._parameters_changed)
        self.class_list.itemChanged.connect(self._parameters_changed)

        self._update_mode_buttons()
        scroll.setWidget(content)
        return scroll

    def _build_shortcuts(self) -> None:
        QShortcut(QKeySequence("Space"), self, activated=self._toggle_pause)
        QShortcut(QKeySequence("F11"), self, activated=self._toggle_fullscreen)
        QShortcut(QKeySequence("R"), self, activated=self._reset_defaults)
        QShortcut(QKeySequence("Q"), self, activated=self.close)

    def _selected_classes(self) -> tuple[int, ...] | None:
        selected = tuple(
            index
            for index in range(self.class_list.count())
            if self.class_list.item(index).checkState() == Qt.CheckState.Checked
        )
        return None if len(selected) == self.class_list.count() else selected

    def _current_parameters(self) -> DetectionParameters:
        return DetectionParameters(
            confidence_threshold=self.confidence_slider.value() / 100.0,
            nms_threshold=self.nms_slider.value() / 100.0,
            nms_mode=self.nms_mode,
            class_filter=self._selected_classes(),
            show_labels=self.labels_check.isChecked(),
            show_confidence=self.confidence_check.isChecked(),
        )

    def _start_video_thread(self) -> None:
        self._stop_video_thread()
        self.video_thread = VideoThread(
            camera_index=self.camera_spin.value(),
            source_type=self.source_type,
            file_path=self.selected_file,
            model_path=self.model_path,
            parent=self,
        )
        self.video_thread.set_parameters(self._current_parameters())
        self.video_thread.model_changed.connect(self._on_model_changed)
        self.video_thread.model_failed.connect(self._on_model_failed)
        self.video_thread.frame_ready.connect(self._update_frames)
        self.video_thread.status_changed.connect(self.status_label.setText)
        self.video_thread.processing_error.connect(self._show_error)
        self.video_thread.source_error.connect(self._show_source_error)
        self.video_thread.start()
        self.pause_button.setText("Pause")
        self.status_label.setText("Starting source...")

    def _stop_video_thread(self) -> None:
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread.deleteLater()
            self.video_thread = None

    def _set_metric(self, name: str, text: str) -> None:
        label = self.metric_values[name]
        if label.text() != text:
            label.setText(text)

    def _update_frames(self, result: FrameResult) -> None:
        if self._closing:
            return
        self.candidate_panel.image.set_frame(result.candidate_view)
        self.final_panel.image.set_frame(result.final_view)
        self.candidate_panel.count_label.setText(f"{result.num_candidates} boxes")
        self.final_panel.count_label.setText(f"{result.num_final} objects")
        self._set_metric("Candidates", str(result.num_candidates))
        self._set_metric("Final detections", str(result.num_final))
        self._set_metric("Mean confidence", f"{result.mean_confidence:.2f}" if result.num_final else "--")
        fps_text = f"{result.fps:.1f}" if result.fps > 0 else "--"
        self._set_metric("FPS", fps_text)
        self._set_metric("Frame size", f"{result.frame_width}×{result.frame_height}")
        self._set_metric("Model (ms)", f"{result.model_ms:.1f}")
        self._set_metric("Post-processing (ms)", f"{result.postprocess_ms:.1f}")
        if self.fps_label.text() != f"FPS: {fps_text}":
            self.fps_label.setText(f"FPS: {fps_text}")
        if self.video_thread is not None:
            self.video_thread.frame_consumed()

    def _show_error(self, message: str) -> None:
        self.error_label.setText(f"postprocess_detections error: {message}" if message else "")

    def _show_source_error(self, message: str) -> None:
        self.error_label.setText(f"Source error: {message}" if message else "")

    def _parameters_changed(self, *_args) -> None:
        if self._restoring:
            return
        self._save_settings()
        if self.video_thread is not None:
            self.video_thread.set_parameters(self._current_parameters())

    # ---- model selection
    def _browse_model(self) -> None:
        start_dir = str(Path(self.model_path).parent) if Path(self.model_path).exists() else ""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select ONNX model", start_dir, "ONNX models (*.onnx);;All Files (*)"
        )
        if file_path and self.video_thread is not None:
            self.status_label.setText(f"Loading model {Path(file_path).name}...")
            self.video_thread.set_model(file_path)

    def _on_model_changed(self, path: str, class_names) -> None:
        """Called after the worker thread loaded a model successfully."""
        self.model_path = path
        self.model_label.setText(Path(path).name)
        self.model_label.setToolTip(path)
        self.error_label.setText("")
        self._set_class_names(list(class_names))
        self._save_settings()

    def _on_model_failed(self, message: str) -> None:
        """The previous model stays active when a new one cannot be loaded."""
        self.error_label.setText(f"Model error: {message}")

    def _set_class_names(self, names: list[str]) -> None:
        current = [self.class_list.item(i).text() for i in range(self.class_list.count())]
        if names == current:
            return
        self.class_list.blockSignals(True)
        self.class_list.clear()
        for name in names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.class_list.addItem(item)
        self.class_list.blockSignals(False)
        self._parameters_changed()

    def _toggle_nms_mode(self) -> None:
        self.nms_mode = NMS_ULTRALYTICS if self.nms_mode == NMS_STUDENT else NMS_STUDENT
        self._update_mode_buttons()
        self._parameters_changed()

    def _update_mode_buttons(self) -> None:
        self.nms_mode_button.setText(f"Current NMS: {self.nms_mode}")

    def _set_all_classes(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self.class_list.blockSignals(True)
        for index in range(self.class_list.count()):
            self.class_list.item(index).setCheckState(state)
        self.class_list.blockSignals(False)
        self._parameters_changed()

    def _apply_source(self) -> None:
        """Tell the (already running) video thread which source to use."""
        self.pause_button.setText("Pause")
        if self.video_thread is not None:
            self.video_thread.set_source(self.source_type, self.selected_file, self.camera_spin.value())

    def _use_webcam(self) -> None:
        self.source_type = "camera"
        self._save_settings()
        self.status_label.setText("Starting webcam...")
        self._apply_source()

    def _camera_index_changed(self, _value: int) -> None:
        self._save_settings()
        if self.source_type == "camera" and not self._closing:
            self._apply_source()

    def _browse_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image or Video", "", FILE_FILTER)
        if not file_path:
            return
        self.selected_file = file_path
        self.file_label.setText(f"Selected: {Path(file_path).name}")
        self.upload_button.setEnabled(True)
        self.status_label.setText("File selected. Click Upload to process it.")

    def _upload_selected_file(self) -> None:
        if not self.selected_file:
            self.status_label.setText("No file selected. Click Browse first.")
            return
        self.source_type = "file"
        self._save_settings()
        self.status_label.setText(f"Opening {Path(self.selected_file).name}...")
        self._apply_source()

    def _toggle_pause(self) -> None:
        if self.video_thread is None:
            return
        paused = self.pause_button.text() == "Pause"
        self.video_thread.set_paused(paused)
        self.pause_button.setText("Resume" if paused else "Pause")

    def _restart_source(self) -> None:
        if self.video_thread is not None:
            self.video_thread.restart_source()
            self.pause_button.setText("Pause")
            self.video_thread.set_paused(False)
            self.status_label.setText("Restarting source...")
        else:
            self._start_video_thread()

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _toggle_theme(self) -> None:
        self.theme = "light" if self.theme == "dark" else "dark"
        self._apply_theme()
        self._save_settings()

    def _apply_theme(self) -> None:
        self.setStyleSheet(DARK_STYLE if self.theme == "dark" else LIGHT_STYLE)
        self.theme_button.setText("Light Mode" if self.theme == "dark" else "Dark Mode")

    def _reset_defaults(self) -> None:
        self.camera_spin.setValue(DEFAULT_CAMERA_INDEX)
        self.nms_mode = DEFAULT_NMS_MODE
        self.confidence_slider.set_value(DEFAULT_CONFIDENCE_PERCENT)
        self.nms_slider.set_value(DEFAULT_NMS_IOU_PERCENT)
        self.labels_check.setChecked(DEFAULT_SHOW_LABELS)
        self.confidence_check.setChecked(DEFAULT_SHOW_CONFIDENCE)
        self._set_all_classes(True)
        self._update_mode_buttons()
        if self.video_thread is not None and Path(self.model_path) != Path(MODEL_PATH):
            self.video_thread.set_model(MODEL_PATH)
        self._save_settings()
        if self.video_thread is not None:
            self.video_thread.set_parameters(self._current_parameters())
        self.status_label.setText("Parameters reset to defaults.")

    def _save_settings(self) -> None:
        if not hasattr(self, "labels_check"):
            return
        unchecked = [
            str(index)
            for index in range(self.class_list.count())
            if self.class_list.item(index).checkState() != Qt.CheckState.Checked
        ]
        self.settings.setValue("camera_index", self.camera_spin.value())
        self.settings.setValue("theme", self.theme)
        self.settings.setValue("model_path", self.model_path)
        self.settings.setValue("nms_mode", self.nms_mode)
        self.settings.setValue("confidence_percent", self.confidence_slider.value())
        self.settings.setValue("nms_iou_percent", self.nms_slider.value())
        self.settings.setValue("show_labels", self.labels_check.isChecked())
        self.settings.setValue("show_confidence", self.confidence_check.isChecked())
        self.settings.setValue("unchecked_classes", ",".join(unchecked))
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/splitter", self.splitter.saveState())
        self.settings.sync()
        self.settings_label.setText("Settings: Saved")

    def _restore_settings(self) -> None:
        self._restoring = True
        self.camera_spin.blockSignals(True)
        self.camera_spin.setValue(self.settings.value("camera_index", DEFAULT_CAMERA_INDEX, int))
        self.camera_spin.blockSignals(False)
        self.theme = str(self.settings.value("theme", DEFAULT_THEME))
        saved_model = str(self.settings.value("model_path", str(MODEL_PATH)))
        self.model_path = saved_model if Path(saved_model).is_file() else str(MODEL_PATH)
        self.model_label.setText(Path(self.model_path).name)
        self.model_label.setToolTip(self.model_path)
        self.nms_mode = str(self.settings.value("nms_mode", DEFAULT_NMS_MODE))
        if self.nms_mode not in (NMS_STUDENT, NMS_ULTRALYTICS):
            self.nms_mode = DEFAULT_NMS_MODE
        self.confidence_slider.set_value(
            int(self.settings.value("confidence_percent", DEFAULT_CONFIDENCE_PERCENT))
        )
        self.nms_slider.set_value(int(self.settings.value("nms_iou_percent", DEFAULT_NMS_IOU_PERCENT)))
        self.labels_check.setChecked(self.settings.value("show_labels", DEFAULT_SHOW_LABELS, bool))
        self.confidence_check.setChecked(
            self.settings.value("show_confidence", DEFAULT_SHOW_CONFIDENCE, bool)
        )
        unchecked = str(self.settings.value("unchecked_classes", "") or "")
        self.class_list.blockSignals(True)
        for token in unchecked.split(","):
            if token.strip().isdigit() and int(token) < self.class_list.count():
                self.class_list.item(int(token)).setCheckState(Qt.CheckState.Unchecked)
        self.class_list.blockSignals(False)
        self._update_mode_buttons()
        geometry, splitter_state = self.settings.value("window/geometry"), self.settings.value("window/splitter")
        if geometry:
            self.restoreGeometry(geometry)
        if splitter_state:
            self.splitter.restoreState(splitter_state)
        self._restoring = False
        self.settings_label.setText("Settings: Restored")

    def closeEvent(self, event) -> None:
        if not self._closing:
            self._closing = True
            self._save_settings()
            self._stop_video_thread()
        event.accept()


def create_application() -> QApplication:
    app = QApplication.instance()
    if app is None:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
        app = QApplication([])
    app.setOrganizationName(ORGANIZATION_NAME)
    app.setApplicationName(APP_NAME)
    return app
