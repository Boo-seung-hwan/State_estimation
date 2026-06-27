"""
Manual actuator GUI for the MuJoCo two-wheel balancing robot.

This version avoids Qt OpenGL completely.
It renders MuJoCo offscreen with OSMesa/EGL through mujoco.Renderer,
then displays the RGB frame in a normal Qt QLabel.

Recommended Docker run env:
    -e MUJOCO_GL=osmesa

Run from project root:
    python src/simulation/manual_actuator_gui_osmesa.py
"""

from __future__ import annotations

import os
# Must be set before importing mujoco. OSMesa is safest in Docker/WSL for this GUI.
os.environ.setdefault("MUJOCO_GL", "osmesa")

import pathlib
import sys
import time
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)


MODEL_PATH = pathlib.Path(__file__).resolve().parent / "scene.xml"
MAX_ACTUATOR_CTRL = 200.0  # rad/s target for MuJoCo velocity actuators
SLIDER_SCALE = 1000
CONTROL_HZ = 200
RENDER_HZ = 30
RENDER_WIDTH = 640
RENDER_HEIGHT = 400


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def get_pitch_rad(data: mujoco.MjData) -> float:
    quat_wxyz = data.body("robot_body").xquat
    if quat_wxyz[0] == 0:
        return 0.0
    rot = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return float(rot.as_euler("xyz", degrees=False)[0])


def get_pitch_dot_rad_s(data: mujoco.MjData) -> float:
    return float(data.joint("robot_body_joint").qvel[-3])


def get_wheel_qvels(data: mujoco.MjData) -> tuple[float, float]:
    return float(data.joint("torso_l_wheel").qvel[0]), float(data.joint("torso_r_wheel").qvel[0])


@dataclass
class ManualCommand:
    left_ctrl: float = 0.0
    right_ctrl: float = 0.0
    paused: bool = False


class ImageViewport(QLabel):
    """QLabel-based MuJoCo viewport. No Qt OpenGL context is used."""

    runtime_signal = Signal(float)

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, opt: mujoco.MjvOption) -> None:
        super().__init__()
        self.model = model
        self.data = data
        self.opt = opt
        self.setMinimumSize(640, 400)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("MuJoCo offscreen renderer starting...")
        self.setStyleSheet("background-color: black; color: white;")

        self.renderer = mujoco.Renderer(model, height=RENDER_HEIGHT, width=RENDER_WIDTH)

        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.fixedcamid = -1
        self.cam.lookat = np.array([0.0, 0.0, 0.10], dtype=float)
        self.cam.distance = 1.5
        self.cam.elevation = -18
        self.cam.azimuth = 130

        self._last_pos = None
        self._last_runtime_values: list[float] = []

        self.timer = QTimer(self)
        self.timer.setInterval(int(1000 / RENDER_HZ))
        self.timer.timeout.connect(self.render_frame)
        self.timer.start()

    def mousePressEvent(self, event):
        self._last_pos = event.position()

    def mouseMoveEvent(self, event):
        if self._last_pos is None:
            self._last_pos = event.position()
            return
        pos = event.position()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()

        if event.buttons() & Qt.MouseButton.LeftButton:
            self.cam.azimuth -= 0.35 * dx
            self.cam.elevation -= 0.35 * dy
            self.cam.elevation = clamp(self.cam.elevation, -89.0, 89.0)
        elif event.buttons() & Qt.MouseButton.RightButton:
            # Simple pan approximation in world x/y.
            scale = 0.0015 * self.cam.distance
            self.cam.lookat[0] -= scale * dx
            self.cam.lookat[1] += scale * dy

        self._last_pos = pos

    def wheelEvent(self, event):
        zoom = 1.0 - 0.001 * event.angleDelta().y()
        self.cam.distance = clamp(self.cam.distance * zoom, 0.2, 10.0)

    @Slot()
    def render_frame(self):
        t0 = time.time()
        try:
            self.renderer.update_scene(self.data, camera=self.cam, scene_option=self.opt)
            img = self.renderer.render()  # RGB uint8, shape H x W x 3
        except Exception as exc:
            self.setText(f"Renderer error:\n{exc}")
            return

        img = np.ascontiguousarray(img)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        self.setPixmap(
            pix.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        dt = time.time() - t0
        self._last_runtime_values.append(dt)
        if len(self._last_runtime_values) > 100:
            self._last_runtime_values.pop(0)
        self.runtime_signal.emit(float(np.mean(self._last_runtime_values)))


class ManualActuatorSimThread(QThread):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, parent=None) -> None:
        super().__init__(parent)
        self.model = model
        self.data = data
        self.running = True
        self.command = ManualCommand()
        self.reset_timers()

    @property
    def real_time_s(self) -> float:
        return (time.monotonic_ns() - self.real_time_start_ns) / 1_000_000_000

    def run(self) -> None:
        control_dt_s = 1.0 / CONTROL_HZ
        while self.running:
            if self.command.paused:
                self._write_actuators(0.0, 0.0)
                time.sleep(0.001)
                self.reset_timers(keep_sim_time=True)
                continue

            if self.data.time < self.real_time_s:
                now = time.monotonic_ns()
                if (now - self.last_control_update_ns) / 1_000_000_000 >= control_dt_s:
                    self.last_control_update_ns = now
                    self._write_actuators(self.command.left_ctrl, self.command.right_ctrl)
                mujoco.mj_step(self.model, self.data)
            else:
                time.sleep(0.00001)

    def _write_actuators(self, left: float, right: float) -> None:
        self.data.actuator("motor_l_wheel").ctrl = [clamp(left, -MAX_ACTUATOR_CTRL, MAX_ACTUATOR_CTRL)]
        self.data.actuator("motor_r_wheel").ctrl = [clamp(right, -MAX_ACTUATOR_CTRL, MAX_ACTUATOR_CTRL)]

    def set_left_ctrl(self, value: float) -> None:
        self.command.left_ctrl = clamp(value, -MAX_ACTUATOR_CTRL, MAX_ACTUATOR_CTRL)

    def set_right_ctrl(self, value: float) -> None:
        self.command.right_ctrl = clamp(value, -MAX_ACTUATOR_CTRL, MAX_ACTUATOR_CTRL)

    def set_pause(self, paused: bool) -> None:
        self.command.paused = bool(paused)

    def zero_actuators(self) -> None:
        self.command.left_ctrl = 0.0
        self.command.right_ctrl = 0.0
        self._write_actuators(0.0, 0.0)

    def reset_timers(self, keep_sim_time: bool = False) -> None:
        now = time.monotonic_ns()
        if keep_sim_time:
            self.real_time_start_ns = now - int(self.data.time * 1_000_000_000)
        else:
            self.real_time_start_ns = now
        self.last_control_update_ns = now

    def reset_sim_state(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.zero_actuators()
        self.reset_timers()

    def stop(self):
        self.running = False
        self.wait()


class Window(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Manual Actuator Control - OSMesa QLabel Renderer")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Cannot find MuJoCo model: {MODEL_PATH}")

        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.opt = mujoco.MjvOption()

        self.th = ManualActuatorSimThread(self.model, self.data, self)
        self.th.start()

        self.viewport = ImageViewport(self.model, self.data, self.opt)
        self.viewport.runtime_signal.connect(self.show_runtime)

        top_layout = QHBoxLayout()
        reset_button = QPushButton("Reset")
        reset_button.setMinimumWidth(90)
        reset_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        reset_button.clicked.connect(self.reset_simulation)
        top_layout.addWidget(reset_button)

        zero_button = QPushButton("Zero actuators")
        zero_button.setMinimumWidth(120)
        zero_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        zero_button.clicked.connect(self.zero_actuators)
        top_layout.addWidget(zero_button)

        self.pause_box = QCheckBox("Pause")
        self.pause_box.stateChanged.connect(self.pause_changed)
        top_layout.addWidget(self.pause_box)

        top_layout.addWidget(self.create_control_box(), stretch=2)
        top_layout.addWidget(self.create_monitor_box(), stretch=1)

        root_layout = QVBoxLayout()
        root_layout.addLayout(top_layout)
        root_layout.addWidget(self.viewport, stretch=1)

        root = QWidget()
        root.setLayout(root_layout)
        self.setCentralWidget(root)
        self.resize(1050, 760)

        self.monitor_timer = QTimer(self)
        self.monitor_timer.setInterval(50)
        self.monitor_timer.timeout.connect(self.update_monitor)
        self.monitor_timer.start()

    def create_control_box(self) -> QGroupBox:
        box = QGroupBox("Manual actuator sliders")
        layout = QVBoxLayout()
        label_width = 130

        self.left_slider, self.left_value_label = self._make_slider_row(
            layout, "Left actuator ctrl", label_width, self.left_slider_changed
        )
        self.right_slider, self.right_value_label = self._make_slider_row(
            layout, "Right actuator ctrl", label_width, self.right_slider_changed
        )

        hint = QLabel(
            "Direct mode: slider value is written directly to data.actuator(...).ctrl.\n"
            "Forward motion usually needs opposite signs on the two wheel sliders."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        box.setLayout(layout)
        return box

    def _make_slider_row(self, parent_layout, label_text, label_width, callback):
        row = QHBoxLayout()
        label = QLabel(label_text)
        label.setFixedWidth(label_width)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(int(-MAX_ACTUATOR_CTRL * SLIDER_SCALE))
        slider.setMaximum(int(MAX_ACTUATOR_CTRL * SLIDER_SCALE))
        slider.setValue(0)
        slider.valueChanged.connect(callback)
        value_label = QLabel("+0.000 rad/s")
        value_label.setFixedWidth(110)
        row.addWidget(label)
        row.addWidget(slider)
        row.addWidget(value_label)
        parent_layout.addLayout(row)
        return slider, value_label

    def create_monitor_box(self) -> QGroupBox:
        box = QGroupBox("State monitor")
        layout = QVBoxLayout()
        self.pitch_label = QLabel("pitch: -")
        self.pitch_dot_label = QLabel("pitch_dot: -")
        self.wheel_label = QLabel("wheel qvel L/R: -")
        self.pos_label = QLabel("root pos xyz: -")
        for label in [self.pitch_label, self.pitch_dot_label, self.wheel_label, self.pos_label]:
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(label)
        box.setLayout(layout)
        return box

    @Slot(float)
    def show_runtime(self, render_dt: float):
        self.statusBar().showMessage(
            f"render avg: {render_dt:.4f}s | sim time: {self.data.time:.2f}s | MUJOCO_GL={os.environ.get('MUJOCO_GL')}"
        )

    @Slot(int)
    def left_slider_changed(self, value: int) -> None:
        ctrl = value / SLIDER_SCALE
        self.left_value_label.setText(f"{ctrl:+.3f} rad/s")
        self.th.set_left_ctrl(ctrl)

    @Slot(int)
    def right_slider_changed(self, value: int) -> None:
        ctrl = value / SLIDER_SCALE
        self.right_value_label.setText(f"{ctrl:+.3f} rad/s")
        self.th.set_right_ctrl(ctrl)

    @Slot()
    def reset_simulation(self) -> None:
        self.left_slider.setValue(0)
        self.right_slider.setValue(0)
        self.th.reset_sim_state()

    @Slot()
    def zero_actuators(self) -> None:
        self.left_slider.setValue(0)
        self.right_slider.setValue(0)
        self.th.zero_actuators()

    @Slot(int)
    def pause_changed(self, state: int) -> None:
        self.th.set_pause(state == Qt.CheckState.Checked.value)

    @Slot()
    def update_monitor(self) -> None:
        pitch = get_pitch_rad(self.data)
        pitch_dot = get_pitch_dot_rad_s(self.data)
        left_qvel, right_qvel = get_wheel_qvels(self.data)
        pos = self.data.qpos[0:3]
        self.pitch_label.setText(f"pitch: {pitch:+.3f} rad / {np.degrees(pitch):+.2f} deg")
        self.pitch_dot_label.setText(f"pitch_dot: {pitch_dot:+.3f} rad/s")
        self.wheel_label.setText(f"wheel qvel L/R: {left_qvel:+.3f} / {right_qvel:+.3f} rad/s")
        self.pos_label.setText(f"root pos xyz: {pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}")

    def closeEvent(self, event):
        self.th.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())
