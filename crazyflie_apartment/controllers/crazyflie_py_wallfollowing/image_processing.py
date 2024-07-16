import cv2
import torch
from ultralytics import YOLO

class depth_estimation_and_object_recognition():
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


        # 加载YOLOv5模型
        try:
            self.yolo_model = YOLO('yolov8s.pt')
        except Exception as e:
            print(f"Error loading YOLOv5 model: {e}")


        # 加载Midas深度估计模型
        try:
            self.midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', pretrained=True).to(self.device)
            self.midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            print("MiDaS model loaded successfully.")
        except Exception as e:
            print(f"Error loading MiDaS model: {e}")
    
    
    def objects_detect(self, image_array):
        object_detection = self.yolo_model(image_array)
        yolo_display = object_detection[0].plot()
        return yolo_display
    
    def estimate_depth(self, camera_image):
        # 将图像转换为Midas模型的输入格式
        transform = self.midas_transforms.dpt_transform
        input_batch = transform(camera_image).to(self.device)

        # 进行深度估计
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size = camera_image.shape[:2],
                mode = 'bicubic',
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
        return depth_map
    
    # 图像滤波器
    def filter_depth_image(depth_image, method='gaussian'):
        if method == 'gaussian':
            return cv2.GaussianBlur(depth_image, (5, 5), 0)
        elif method == 'median':
            return cv2.medianBlur(depth_image, 5)
        elif method == 'bilateral':
            return cv2.bilateralFilter(depth_image, 9, 75, 75)
        elif method == 'mean':
            return cv2.blur(depth_image, (5, 5))
        else:
            raise ValueError("Unsupported filtering method")
        
    