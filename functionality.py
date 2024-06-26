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
        self.yolo_model = None
        self.camera_processor = None
        self.model_loader = None
        self.loading_animation_timer = QTimer()
        self.loading_animation_step = 0
        self.model_loading_in_progress = False
        self.current_model_name = None

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
        midas_model_name = self._get_selected_midas_model_name()
        yolo_model_name = self._get_selected_yolo_model_name()

        if self.model_loading_in_progress:
            return

        if midas_model_name == self.current_model_name and yolo_model_name == self.current_model_name:
            self.message_display.append_text("Loaded model is the same as the current model. No action taken.\n")
            return

        self.model_loading_in_progress = True

        self._set_controls_enabled({
            'launch': False,
            'stop': False,
            'load_model': False,
            'connect': False,
            'disconnect': False,
            'radio_buttons': False
        })
        self.message_display.set_text(f"Loading models {midas_model_name} and {yolo_model_name}, please wait...\n")

        self._release_current_model()

        self.model_loader = ModelLoader(self.device, midas_model_name, yolo_model_name)
        self.model_loader.update_output.connect(self._update_text_edit)
        self.model_loader.models_loaded.connect(self._set_models)
        self.model_loader.start()
        self.loading_animation_timer.start(500)

    def _get_selected_midas_model_name(self):
        if self.ui_builder.radioButton_midas_small.isChecked():
            return 'MiDaS_small'
        elif self.ui_builder.radioButton_midas_hybrid.isChecked():
            return 'DPT_Hybrid'
        elif self.ui_builder.radioButton_midas_large.isChecked():
            return 'DPT_Large'
        else:
            return None

    def _get_selected_yolo_model_name(self):
        if self.ui_builder.radioButton_yolov1.isChecked():
            return 'yolov5s'  # Replace with actual YOLOv1 model name if needed
        elif self.ui_builder.radioButton_yolov2.isChecked():
            return 'yolov5s'  # Replace with actual YOLOv2 model name if needed
        elif self.ui_builder.radioButton_yolov3.isChecked():
            return 'yolov5s'  # Replace with actual YOLOv3 model name if needed
        elif self.ui_builder.radioButton_yolov4.isChecked():
            return 'yolov5s'  # Replace with actual YOLOv4 model name if needed
        elif self.ui_builder.radioButton_yolov5.isChecked():
            return 'yolov5s'
        elif self.ui_builder.radioButton_yolov6.isChecked():
            return 'yolov5s'  # Replace with actual YOLOv6 model name if needed
        elif self.ui_builder.radioButton_yolov7.isChecked():
            return 'yolov5s'  # Replace with actual YOLOv7 model name if needed
        elif self.ui_builder.radioButton_yolov8.isChecked():
            return 'yolov5s'  # Replace with actual YOLOv8 model name if needed
        else:
            return None

    def _update_text_edit(self, text):
        self.message_display.append_text(text)

    def _set_models(self, midas, midas_transforms, yolo_model):
        self.midas = midas
        self.midas_transforms = midas_transforms
        self.yolo_model = yolo_model
        
        self.current_model_name = self._get_selected_midas_model_name() if self.midas else None
        self.loading_animation_timer.stop()
        final_message = "\nModels loaded successfully." if self.midas and self.yolo_model else "\nFailed to load models."
        self.message_display.append_text(final_message)
        self.model_loading_in_progress = False

        self._set_controls_enabled({
            'launch': True if self.midas and self.yolo_model else False,
            'stop': False,
            'load_model': True,
            'connect': True,
            'disconnect': True,
            'radio_buttons': True
        })

    def _release_current_model(self):
        if self.camera_processor is not None and self.camera_processor.isRunning():
            self.camera_processor.stop()
            self.camera_processor = None

        if self.midas is not None:
            del self.midas
            self.midas = None

        if self.midas_transforms is not None:
            del self.midas_transforms
            self.midas_transforms = None

        if self.yolo_model is not None:
            del self.yolo_model
            self.yolo_model = None

        torch.cuda.empty_cache()  # Clear CUDA memory if used

    def _loading_animation(self):
        self.message_display.loading_animation(self.loading_animation_step)
        self.loading_animation_step += 1

    def _on_connect(self):
        print('连接按钮被点击了！')

    def _on_disconnect(self):
        print('断开按钮被点击了！')

    def _on_launch(self):
        if not self.midas or not self.yolo_model:
            self.show_error_message("Models not loaded. Please load the models first.")
            return

        if self.camera_processor is None or not self.camera_processor.isRunning():
            self.camera_processor = CameraProcessor(self.midas, self.midas_transforms, self.yolo_model)
            self.camera_processor.frameCaptured.connect(self.update_frame)
            self.camera_processor.processingStopped.connect(self.on_processing_stopped)
            self.camera_processor.errorOccurred.connect(self.show_error_message)
            self.camera_processor.start()
            self._set_controls_enabled({'launch': False, 'stop': True, 'load_model': False, 'connect': True, 'disconnect': True, 'radio_buttons': False})

    def _on_stop(self):
        if self.camera_processor is not None and self.camera_processor.isRunning():
            self.camera_processor.stop()
            self._set_controls_enabled({'launch': True, 'stop': False, 'load_model': True, 'connect': True, 'disconnect': True, 'radio_buttons': True})

    def on_processing_stopped(self):
        self.camera_processor = None
        self._set_controls_enabled({'launch': True, 'stop': False, 'load_model': True, 'connect': True, 'disconnect': True, 'radio_buttons': True})

    def show_error_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText(message)
        msg_box.setWindowTitle("错误")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec()

    def update_frame(self, original_frame, yolo_frame, midas_frame):
        self._display_image(self.ui_builder.video_original, original_frame)
        self._display_image(self.ui_builder.video_yolo, yolo_frame)
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
        self._release_current_model()
        if self.model_loader is not None:
            self.model_loader.quit()
            self.model_loader.wait()
