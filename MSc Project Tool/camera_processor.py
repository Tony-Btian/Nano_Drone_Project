import cv2
import numpy as np
import torch

class CameraProcessor:
    def __init__(self):
        # 检查CUDA是否可用
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 验证CUDA可用性
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA is not available. Switching to CPU.")


        # 加载Midas深度估计模型
        try:
            self.midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Large', pretrained=True).to(self.device)
            self.midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            print("MiDaS model loaded successfully.")
        except Exception as e:
            print(f"Error loading MiDaS model: {e}")           
            self.midas = None
            self.midas_transforms = None

    def estimate_depth(self, frame):
        # 将图像转换为Midas模型的输入格式
        if self.midas is None or self.midas_transforms is None:
            raise RuntimeError("MiDaS model or transforms are not loaded properly.")
        
        transform = self.midas_transforms.dpt_transform
        input_batch = transform(frame).to(self.device)
        
        # 进行深度估计
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
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
        return depth_map