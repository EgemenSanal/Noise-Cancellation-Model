import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.audio.stream import AudioStream
from app.processing.analyzer import AudioAnalyzer


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Real-Time Audio Monitor")
        self.resize(1100, 750)

        self.stream = None
        self.analyzer = AudioAnalyzer()

        self.peak_hold_db = -100.0
        self.peak_hold_decay = 1.0

        self._build_ui()

        self.timer = QTimer(self)
        self.timer.setInterval(20)
        self.timer.timeout.connect(self.update_audio)

        self.load_devices()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        # Device selection
        device_layout = QHBoxLayout()

        device_layout.addWidget(QLabel("Microphone:"))

        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(400)

        device_layout.addWidget(self.device_combo)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.toggle_stream)

        device_layout.addWidget(self.start_button)

        layout.addLayout(device_layout)

        # Audio information
        volume_label = QLabel("Volume")

        self.volume_meter = QProgressBar()
        self.volume_meter.setRange(0, 60)
        self.volume_meter.setValue(0)
        self.volume_meter.setTextVisible(False)
        self.volume_meter.setMinimumHeight(25)

        layout.addWidget(volume_label)
        layout.addWidget(self.volume_meter)

        self.peak_hold_label = QLabel("Peak Hold: -100.00 dBFS")
        layout.addWidget(self.peak_hold_label)

        info_layout = QHBoxLayout()

        self.rms_label = QLabel("RMS: -100.00 dBFS")
        self.peak_label = QLabel("Peak: -100.00 dBFS")
        self.centroid_label = QLabel("Centroid: 0 Hz")
        self.zcr_label = QLabel("ZCR: 0.000")

        info_layout.addWidget(self.rms_label)
        info_layout.addWidget(self.peak_label)
        info_layout.addWidget(self.centroid_label)
        info_layout.addWidget(self.zcr_label)

        layout.addLayout(info_layout)

        # Waveform
        self.waveform_plot = pg.PlotWidget()
        self.waveform_plot.setLabel("left", "Amplitude")
        self.waveform_plot.setLabel("bottom", "Samples")
        self.waveform_plot.setYRange(-1.0, 1.0)
        self.waveform_plot.showGrid(x=True, y=True)

        self.waveform_curve = self.waveform_plot.plot()

        layout.addWidget(self.waveform_plot)

        # Spectrum
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setLabel("left", "Magnitude", units="dB")
        self.spectrum_plot.setLabel("bottom", "Frequency", units="Hz")
        self.spectrum_plot.setXRange(0, 12_000)
        self.spectrum_plot.showGrid(x=True, y=True)

        self.spectrum_curve = self.spectrum_plot.plot()

        layout.addWidget(self.spectrum_plot)
        self.clipping_label = QLabel("Clipping: 0 samples")
        layout.addWidget(self.clipping_label)

        self.noise_floor_label = QLabel("Noise Floor: -100.00 dBFS")
        layout.addWidget(self.noise_floor_label)

        self.dynamic_range_label = QLabel("Dynamic Range: 0.00 dB")
        layout.addWidget(self.dynamic_range_label)

    def load_devices(self):
        self.device_combo.clear()

        devices = AudioStream.get_input_devices()

        for device in devices:
            self.device_combo.addItem(
                device["name"],
                device["index"],
            )

    def toggle_stream(self):
        if self.stream is None:
            self.start_stream()
        else:
            self.stop_stream()

    def start_stream(self):
        device_index = self.device_combo.currentData()

        self.stream = AudioStream(
            sample_rate=48_000,
            channels=1,
            block_duration_ms=20,
            device=device_index,
        )

        try:
            self.stream.start()
        except Exception as exc:
            print(f"Failed to start audio stream: {exc}")
            self.stream = None
            return

        self.start_button.setText("Stop")
        self.device_combo.setEnabled(False)

        self.timer.start()

    def stop_stream(self):
        self.timer.stop()

        if self.stream is not None:
            self.stream.stop()
            self.stream = None

        self.start_button.setText("Start")
        self.device_combo.setEnabled(True)

        self.peak_hold_db = -100.0
        self.peak_hold_label.setText("Peak Hold: -100.00 dBFS")
        self.volume_meter.setValue(0)

    def update_audio(self):
        if self.stream is None:
            return

        audio = self.stream.read()

        if audio is None:
            return

        result = self.analyzer.analyze(
            audio,
            sample_rate=self.stream.sample_rate,
        )

        rms_db = result["rms_db"]
        peak_db = result["peak_db"]

        clipping_count = result["clipping_count"]
        noise_floor = result["noise_floor"]

        dynamic_range = max(
            0.0,
            peak_db - noise_floor
        )
        if clipping_count > 0:
            self.clipping_label.setText(
                f"⚠ CLIPPING: {clipping_count} samples"
            )
        else:
            self.clipping_label.setText(
                "Clipping: 0 samples"
            )

        self.noise_floor_label.setText(
            f"Noise Floor: {noise_floor:.2f} dBFS"
        )

        self.dynamic_range_label.setText(
            f"Dynamic Range: {dynamic_range:.2f} dB"
        )

        self.rms_label.setText(
            f"RMS: {rms_db:.2f} dBFS"
        )

        self.peak_label.setText(
            f"Peak: {peak_db:.2f} dBFS"
        )

        self.centroid_label.setText(
            f"Centroid: {result['spectral_centroid']:.0f} Hz"
        )

        self.zcr_label.setText(
            f"ZCR: {result['zero_crossing_rate']:.3f}"
        )

        # Volume meter
        meter_value = int(np.clip(rms_db + 60, 0, 60))
        self.volume_meter.setValue(meter_value)

        # Peak hold
        if peak_db > self.peak_hold_db:
            self.peak_hold_db = peak_db
        else:
            self.peak_hold_db -= self.peak_hold_decay

        self.peak_hold_db = max(self.peak_hold_db, -100.0)

        self.peak_hold_label.setText(
            f"Peak Hold: {self.peak_hold_db:.2f} dBFS"
        )

        self.centroid_label.setText(
            f"Centroid: {result['spectral_centroid']:.0f} Hz"
        )

        self.zcr_label.setText(
            f"ZCR: {result['zero_crossing_rate']:.3f}"
        )

        waveform = result["waveform"]

        self.waveform_curve.setData(
            np.arange(len(waveform)),
            waveform,
        )

        self.spectrum_curve.setData(
            result["frequencies"],
            result["spectrum_db"],
        )

    def closeEvent(self, event):
        self.stop_stream()
        event.accept()
