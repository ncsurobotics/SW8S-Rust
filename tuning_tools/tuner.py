import sys
import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QSlider,
    QSpinBox,
    QCheckBox,
)

RTSP_URL = "rtsp://192.168.2.5:8554/front.mp4"


class RTSPViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RTSP YUV Viewer (OpenCV)")

        # OpenCV RTSP capture using GStreamer
        self.cap = cv2.VideoCapture(
            f"rtspsrc location={RTSP_URL} latency=0 ! decodebin ! videoconvert ! appsink",
            cv2.CAP_GSTREAMER,
        )
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open RTSP stream: {RTSP_URL}")

        # Default YUV ranges
        self.yuv_min = np.array([0, 0, 0], dtype=np.uint8)
        self.yuv_max = np.array([255, 255, 255], dtype=np.uint8)
        self.show_mask_only = False
        self.overlay_mode = True

        # UI: video + sliders
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setMinimumSize(640, 360)

        controls = self._build_controls()

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(self.video_label, 3)
        layout.addLayout(controls, 1)
        self.setCentralWidget(central)

        # Timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33 FPS

    def _build_controls(self):
        layout = QVBoxLayout()
        form = QFormLayout()

        def add_slider(label, idx, init, minv=0, maxv=255):
            slider = QSlider(Qt.Horizontal)
            slider.setRange(minv, maxv)
            slider.setValue(init)
            spin = QSpinBox()
            spin.setRange(minv, maxv)
            spin.setValue(init)
            slider.valueChanged.connect(lambda v: spin.setValue(v))
            spin.valueChanged.connect(lambda v: slider.setValue(v))
            slider.valueChanged.connect(lambda v: self._on_slider(idx, v))
            spin.valueChanged.connect(lambda v: self._on_slider(idx, v))

            h = QHBoxLayout()
            h.addWidget(slider)
            h.addWidget(spin)
            form.addRow(label, h)

        add_slider("Y min", (0, "min"), 0)
        add_slider("Y max", (0, "max"), 255)
        add_slider("U min", (1, "min"), 0)
        add_slider("U max", (1, "max"), 255)
        add_slider("V min", (2, "min"), 0)
        add_slider("V max", (2, "max"), 255)

        layout.addLayout(form)

        cb_mask = QCheckBox("Show mask only")
        cb_mask.stateChanged.connect(
            lambda s: setattr(self, "show_mask_only", s == Qt.Checked)
        )
        layout.addWidget(cb_mask)

        cb_overlay = QCheckBox("Overlay on original")
        cb_overlay.setChecked(True)
        cb_overlay.stateChanged.connect(
            lambda s: setattr(self, "overlay_mode", s == Qt.Checked)
        )
        layout.addWidget(cb_overlay)

        layout.addStretch(1)
        return layout

    def _on_slider(self, idx_info, value):
        idx, bound = idx_info
        if bound == "min":
            self.yuv_min[idx] = value
        else:
            self.yuv_max[idx] = value

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Convert to YUV
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        mask = cv2.inRange(yuv, self.yuv_min, self.yuv_max)

        if self.show_mask_only:
            disp = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        elif self.overlay_mode:
            disp = cv2.bitwise_and(frame, frame, mask=mask)
        else:
            disp = frame

        # Convert to QImage
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = RTSPViewer()
    win.resize(960, 540)
    win.show()
    sys.exit(app.exec())
