import cv2
import torch
import numpy as np
from PySide6.QtCore import QThread, Signal
from torchvision.ops import nms

class CameraProcessor(QThread):
    frameCaptured = Signal(np.ndarray, np.ndarray, np.ndarray)  # Signal to send frames to UI
    processingStopped = Signal()  # Signal to notify that processing has stopped
    errorOccurred = Signal(str)  # Signal to notify that an error has occurred

    def __init__(self, midas, midas_transforms, yolo_model, parent=None):
        super().__init__(parent)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.midas = midas
        self.midas_transforms = midas_transforms
        self.yolo_model = yolo_model
        self.cap = cv2.VideoCapture(0)  # Initialize camera
        self._run_flag = False

    def estimate_depth(self, frame):
        if not self.midas or not self.midas_transforms:
            raise RuntimeError("MiDaS model or transforms are not loaded properly.")
        
        transform = self.midas_transforms.dpt_transform
        frame = np.array(frame)
        input_batch = transform(frame).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode='bicubic',
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

    def detect_objects(self, frame):
        self.yolo_model.to(self.device)
        self.yolo_model.eval()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float().to(self.device) / 255.0
        img = img.unsqueeze(0)

        with torch.no_grad():
            results = self.yolo_model(img)[0]

        # Perform NMS on CPU
        results = results.cpu()
        boxes = results[:, :4]
        scores = results[:, 4]
        indices = nms(boxes, scores, 0.5).cpu()

        return results[indices].numpy()

    def draw_boxes(self, frame, results):
        for result in results:
            x1, y1, x2, y2, conf, cls = result
            label = f'{int(cls)}: {conf:.2f}'
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        return frame

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.errorOccurred.emit("摄像头连接出错")
            return None
        return frame
    
    def release(self):
        self.cap.release()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def run(self):
        self._run_flag = True
        while self._run_flag:
            frame = self.get_frame()
            if frame is not None:
                midas_frame = self.estimate_depth(frame)
                yolo_results = self.detect_objects(frame)
                frame_with_boxes = self.draw_boxes(frame, yolo_results)
                self.frameCaptured.emit(frame, frame_with_boxes, midas_frame)
            else:
                self._run_flag = False
        self.release()
        self.processingStopped.emit()  # Emit signal to notify processing has stopped

    def stop(self):
        self._run_flag = False
