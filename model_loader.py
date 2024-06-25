import sys
import torch
from io import StringIO
from PySide6.QtCore import QThread, Signal

class ModelLoader(QThread):
    update_output = Signal(str)
    model_loaded = Signal(object, object)

    def __init__(self, device, model_name):
        super().__init__()
        self.device = device
        self.model_name = model_name

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

            midas = torch.hub.load('intel-isl/MiDaS', self.model_name, pretrained=True).to(self.device)
            midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            print(f"MiDaS model {self.model_name} loaded successfully.")
            self.model_loaded.emit(midas, midas_transforms)
        except Exception as e:
            print(f"Error loading MiDaS model: {e}")
            self.model_loaded.emit(None, None)
        finally:
            self.update_output.emit(mystdout.getvalue())
            sys.stdout = old_stdout
