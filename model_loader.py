import sys
import torch
from io import StringIO
from PySide6.QtCore import QThread, Signal

class ModelLoader(QThread):
    update_output = Signal(str)
    models_loaded = Signal(object, object, object)  # Emit MiDaS model, MiDaS transforms, and YOLO model

    def __init__(self, device, midas_model_name, yolo_model_name):
        super().__init__()
        self.device = device
        self.midas_model_name = midas_model_name
        self.yolo_model_name = yolo_model_name

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

            # Load MiDaS model
            midas = torch.hub.load('intel-isl/MiDaS', self.midas_model_name, pretrained=True).to(self.device)
            midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            print(f"MiDaS model {self.midas_model_name} loaded successfully.")

            # Load YOLO model
            yolo_model = torch.hub.load('ultralytics/yolov5', self.yolo_model_name, pretrained=True).to(self.device)
            print(f"YOLO model {self.yolo_model_name} loaded successfully.")
            
            self.models_loaded.emit(midas, midas_transforms, yolo_model)
        except Exception as e:
            print(f"Error loading models: {e}")
            self.models_loaded.emit(None, None, None)
        finally:
            self.update_output.emit(mystdout.getvalue())
            sys.stdout = old_stdout
