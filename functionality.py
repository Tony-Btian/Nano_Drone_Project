from PySide6.QtGui import QImage, QPixmap, QTextCursor
from PySide6.QtCore import Qt, QObject, QThread, Signal, QTimer
from PySide6.QtWidgets import QMessageBox
from camera_processor import CameraProcessor
import torch
import sys
from io import StringIO

class ModelLoader(QThread):
    update_output = Signal(str)
    model_loaded = Signal(object, object)

    def __init__(self, device):
        super().__init__()
        self.device = device

    def run(self):
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        try:
            if torch.cuda.is_available():
                print(f"CUDA device count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
            else:
                print("CUDA is not available. Switching to CPU.")

            midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Large', pretrained=True).to(self.device)
            midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            print("MiDaS model loaded successfully.")
            self.model_loaded.emit(midas, midas_transforms)
        except Exception as e:
            print(f"Error loading MiDaS model: {e}")
            self.model_loaded.emit(None, None)
        finally:
            self.update_output.emit(mystdout.getvalue())
            sys.stdout = old_stdout

class Functionality(QObject):
    def __init__(self, ui_builder):
        super().__init__()
        self.ui_builder = ui_builder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.midas = None
        self.midas_transforms = None
        self.camera_processor = None
        self.loading_animation_timer = QTimer()
        self.loading_animation_timer.timeout.connect(self._loading_animation)
        self.loading_animation_step = 0
        self._initialize_midas()

    def _initialize_midas(self):
        self._set_controls_enabled(False)
        self.ui_builder.textEdit.setPlainText("Loading MiDaS model, please wait...\n")
        self.model_loader = ModelLoader(self.device)
        self.model_loader.update_output.connect(self._update_text_edit)
        self.model_loader.model_loaded.connect(self._set_model)
        self.model_loader.start()
        self.loading_animation_timer.start(500)

    def _update_text_edit(self, text):
        current_text = self.ui_builder.textEdit.toPlainText()
        self.ui_builder.textEdit.setPlainText(current_text + text)

    def _set_model(self, midas, midas_transforms):
        self.midas = midas
        self.midas_transforms = midas_transforms
        self.loading_animation_timer.stop()
        final_message = "\nMiDaS model loaded successfully." if self.midas else "\nFailed to load MiDaS model."
        self.ui_builder.textEdit.append(final_message)
        self._set_controls_enabled(True)

    def _set_controls_enabled(self, enabled):
        self.ui_builder.pushButton_connect.setEnabled(enabled)
        self.ui_builder.pushButton_disconnect.setEnabled(enabled)
        self.ui_builder.pushButton_launch.setEnabled(enabled)
        self.ui_builder.pushButton_stop.setEnabled(enabled)
        self.ui_builder.checkBox_config_1.setEnabled(enabled)
        self.ui_builder.checkBox_config_2.setEnabled(enabled)
        self.ui_builder.checkBox_config_3.setEnabled(enabled)

    def _loading_animation(self):
        animation_steps = ['.', '..', '...', '....', '.....']
        step = animation_steps[self.loading_animation_step % len(animation_steps)]
        self.loading_animation_step += 1
        loading_text = f"Loading MiDaS model, please wait{step}"
        self.ui_builder.textEdit.setPlainText(loading_text)

    def _on_connect(self):
        print('连接按钮被点击了！')

    def _on_disconnect(self):
        print('断开按钮被点击了！')

    def _on_launch(self):
        if self.camera_processor is None or not self.camera_processor.isRunning():
            self.camera_processor = CameraProcessor(self.midas, self.midas_transforms)
            self.camera_processor.frameCaptured.connect(self.update_frame)
            self.camera_processor.processingStopped.connect(self.on_processing_stopped)
            self.camera_processor.errorOccurred.connect(self.show_error_message)
            self.camera_processor.start()
        print('启动按钮被点击了！')

    def _on_stop(self):
        if self.camera_processor is not None and self.camera_processor.isRunning():
            self.camera_processor.stop()
        print('停止按钮被点击了！')

    def on_processing_stopped(self):
        print('处理已停止')
        self.camera_processor = None

    def show_error_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText(message)
        msg_box.setWindowTitle("错误")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec()

    def update_frame(self, original_frame, midas_frame):
        self.display_image(self.ui_builder.video_original, original_frame)
        self.display_image(self.ui_builder.video_midas, midas_frame)

    def display_image(self, label, frame):
        if frame is None:
            label.clear()
            return
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        label.setPixmap(QPixmap.fromImage(q_img).scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio))
