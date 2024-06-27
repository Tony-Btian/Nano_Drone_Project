import sys
import torch
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QObject, QTimer
from PySide6.QtWidgets import QMessageBox
from camera_processor import CameraProcessor
from model_loader import ModelLoader
from message_display import MessageDisplay

class Functionality(QObject):
    def __init__(self, ui_builder):
        super().__init__()
        self.ui_builder = ui_builder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.midas = None
        self.midas_transforms = None
        self.camera_processor = None
        self.model_loader = None
        self.loading_animation_timer = QTimer()
        self.loading_animation_step = 0
        self.model_loading_in_progress = False  # Add this flag

        self.message_display = MessageDisplay(self.ui_builder.textEdit)

        self._connect_signals()
        self._initialize_controls()

    def _connect_signals(self):
        self.ui_builder.pushButton_loadmodel.clicked.connect(self._on_load_model)
        self.ui_builder.pushButton_launch.clicked.connect(self._on_launch)
        self.ui_builder.pushButton_stop.clicked.connect(self._on_stop)
        self.ui_builder.pushButton_connect.clicked.connect(self._on_connect)
        self.ui_builder.pushButton_disconnect.clicked.connect(self._on_disconnect)
        self.loading_animation_timer.timeout.connect(self._loading_animation)

    def _initialize_controls(self):
        self._set_controls_enabled({
            'launch': False,
            'stop': False,
            'load_model': True,
            'connect': True,
            'disconnect': True,
            'radio_buttons': True
        })

    def _set_controls_enabled(self, states):
        self.ui_builder.pushButton_launch.setEnabled(states['launch'])
        self.ui_builder.pushButton_stop.setEnabled(states['stop'])
        self.ui_builder.pushButton_loadmodel.setEnabled(states['load_model'])
        self.ui_builder.pushButton_connect.setEnabled(states['connect'])
        self.ui_builder.pushButton_disconnect.setEnabled(states['disconnect'])
        radio_buttons = [
            self.ui_builder.radioButton_yolov1,
            self.ui_builder.radioButton_yolov2,
            self.ui_builder.radioButton_yolov3,
            self.ui_builder.radioButton_yolov4,
            self.ui_builder.radioButton_yolov5,
            self.ui_builder.radioButton_yolov6,
            self.ui_builder.radioButton_yolov7,
            self.ui_builder.radioButton_yolov8,
            self.ui_builder.radioButton_midas_small,
            self.ui_builder.radioButton_midas_hybrid,
            self.ui_builder.radioButton_midas_large
        ]
        for button in radio_buttons:
            button.setEnabled(states['radio_buttons'])

    def _on_load_model(self):
        if self.model_loading_in_progress:  # Check if model loading is already in progress
            return
        self.model_loading_in_progress = True  # Set the flag to true

        self._set_controls_enabled({
            'launch': False,
            'stop': False,
            'load_model': False,
            'connect': False,
            'disconnect': False,
            'radio_buttons': False
        })
        self.message_display.set_text("Loading MiDaS model, please wait...\n")
        model_name = self._get_selected_model_name()
        if self.model_loader is not None and self.model_loader.isRunning():
            self.model_loader.quit()
            self.model_loader.wait()
        self.model_loader = ModelLoader(self.device, model_name)
        self.model_loader.update_output.connect(self._update_text_edit)
        self.model_loader.model_loaded.connect(self._set_model)
        self.model_loader.start()
        self.loading_animation_timer.start(500)

    def _get_selected_model_name(self):
        if self.ui_builder.radioButton_midas_small.isChecked():
            return 'MiDaS_small'
        elif self.ui_builder.radioButton_midas_hybrid.isChecked():
            return 'MiDaS_hybrid'
        else:
            return 'DPT_Large'

    def _update_text_edit(self, text):
        self.message_display.append_text(text)

    def _set_model(self, midas, midas_transforms):
        self.midas = midas
        self.midas_transforms = midas_transforms
        self.loading_animation_timer.stop()
        final_message = "\nMiDaS model loaded successfully." if self.midas else "\nFailed to load MiDaS model."
        self.message_display.append_text(final_message)
        self.model_loading_in_progress = False  # Reset the flag

        self._set_controls_enabled({
            'launch': True,
            'stop': True,
            'load_model': True,
            'connect': True,
            'disconnect': True,
            'radio_buttons': True
        })

    def _loading_animation(self):
        self.message_display.loading_animation(self.loading_animation_step)
        self.loading_animation_step += 1

    def _on_connect(self):
        print('连接按钮被点击了！')

    def _on_disconnect(self):
        print('断开按钮被点击了！')

    def _on_launch(self):
        if not self.midas:
            self.show_error_message("Model not loaded. Please load a model first.")
            return

        if self.camera_processor is None or not self.camera_processor.isRunning():
            self.camera_processor = CameraProcessor(self.midas, self.midas_transforms)
            self.camera_processor.frameCaptured.connect(self.update_frame)
            self.camera_processor.processingStopped.connect(self.on_processing_stopped)
            self.camera_processor.errorOccurred.connect(self.show_error_message)
            self.camera_processor.start()

    def _on_stop(self):
        if self.camera_processor is not None and self.camera_processor.isRunning():
            self.camera_processor.stop()

    def on_processing_stopped(self):
        self.camera_processor = None

    def show_error_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText(message)
        msg_box.setWindowTitle("错误")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec()

    def update_frame(self, original_frame, midas_frame):
        self._display_image(self.ui_builder.video_original, original_frame)
        self._display_image(self.ui_builder.video_yolo, original_frame)
        self._display_image(self.ui_builder.video_midas, midas_frame)

    def _display_image(self, label, frame):
        if frame is None:
            label.clear()
            return
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        label.setPixmap(QPixmap.fromImage(q_img).scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def __del__(self):
        if self.model_loader is not None:
            self.model_loader.quit()
            self.model_loader.wait()
