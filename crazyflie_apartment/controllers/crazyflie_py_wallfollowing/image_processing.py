import cv2
import torch
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN # type: ignore

class depth_estimation_and_object_recognition():
    def __init__(self):
        # Check if CUDA is Available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Verify CUDA Availability
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA is not available. Switching to CPU.")

        # Loading the YOLOv5 Model
        try:
            self.yolo_model = YOLO('yolov8s.pt')
        except Exception as e:
            print(f"Error loading YOLOv5 model: {e}")

        # Loading the Midas Depth Estimation Model
        try:
            self.midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', pretrained=True).to(self.device)
            self.midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            print("MiDaS model loaded successfully.")
        except Exception as e:
            print(f"Error loading MiDaS model: {e}")
    

    # -------------- Object Recognition and Depth Estimation ---------------- #
    # Object Detection 物体检测
    def objects_detect(self, image_array):
        object_detection = self.yolo_model(image_array)
        yolo_display = object_detection[0].plot()
        return yolo_display
    

    # Depth Estimation 深度估计
    def estimate_depth(self, camera_image):
        # Converting images to Midas model input format
        transform = self.midas_transforms.dpt_transform
        input_batch = transform(camera_image).to(self.device)

        # Perform depth estimation
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size = camera_image.shape[:2],
                mode = 'bicubic',
                align_corners=False,
            ).squeeze()
        
        depth_value = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_value, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
        return depth_value, depth_map
    
    
    # Sobel Operator for Edge Detection
    def sobel_edge_detection(self, depth_image):
        # Convert to grayscale image
        gray = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        
        # Calculate the gradient in the x-direction and y-direction
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate the gradient magnitude
        grad = cv2.magnitude(grad_x, grad_y)
        
        # Normalized and converted to 8-bit images
        grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)
        grad = np.uint8(grad)
        
        return grad
    
    # Canny Operator for Edge Detection
    # def canny_dege_detection(self, depth_image):
    

    # Convert a depth map to a grid map
    def depth_to_grid(self, depth_map, threshold, grid_size):
        """
        将深度图转换为网格地图。
        
        :param depth_map: MiDas 输出的深度图
        :param threshold: 深度阈值，小于该值视为障碍物
        :param grid_size: 网格单元的大小
        :return: 生成的网格地图
        """
        rows, cols = depth_map.shape
        grid_rows = rows // grid_size
        grid_cols = cols // grid_size
        grid_map = np.zeros((grid_rows, grid_cols), dtype=int)
        
        for i in range(grid_rows):
            for j in range(grid_cols):
                grid_cell = depth_map[i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size]
                if np.any(grid_cell < threshold):
                    grid_map[i, j] = 1  # 标记为障碍物
        
        return grid_map
    

    # -------------- Handle Obstacle Detection ---------------- #
    def depth_to_point_cloud(self, depth_image, K):
        # 假设内参矩阵K
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        height, width = depth_image.shape

        i, j = np.indices((height, width))
        z = depth_image
        x = (j - cx) * z / fx
        y = (i - cy) * z / fy

        point_cloud = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        return point_cloud
    

    # Clustering-based Obstacle Detection
    def detect_obstacles_clustering(self, point_cloud, eps=0.5, min_samples=10):
        # 确保点云是二维数组
        if len(point_cloud.shape) == 1:
            raise ValueError("Point cloud must be a 2D array with shape (N, 3)")
        # 应用DBSCAN聚类算法
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(point_cloud.reshape(-1, 3))
        labels = clustering.labels_
        unique_labels = set(labels)
        obstacles = []

        for label in unique_labels:
            if label == -1:  # 噪声点
                continue
            obstacle_points = point_cloud[labels == label]
            centroid = np.mean(obstacle_points, axis=0)
            obstacles.append(centroid)
            
        return obstacles


    # Compute the Gradient of the Depth Map
    def detect_obstacles_gradient(self, depth_map, grad_threshold):
        # 计算深度图的梯度
        grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=5)
        grad = np.sqrt(grad_x**2 + grad_y**2)
        
        # 梯度大于阈值的区域被认为是障碍物
        obstacle_mask = grad > grad_threshold
        return obstacle_mask
    

    # Calculate the Location and Size of Obstacles
    def compute_obstacle_properties(slef, obstacle_mask):
        # 检查 obstacle_mask 的维度
        print(f"obstacle_mask shape: {obstacle_mask.shape}")

        # 如果 obstacle_mask 是高维的，调整为二维
        if obstacle_mask.ndim > 2:
            obstacle_mask = np.any(obstacle_mask, axis=tuple(range(2, obstacle_mask.ndim)))

        y_indices, x_indices = np.nonzero(obstacle_mask)

        if len(y_indices) == 0 or len(x_indices) == 0:
            raise ValueError("No obstacles found in the given mask.")

        position = (int(np.mean(y_indices)), int(np.mean(x_indices)))
        size = (int(np.ptp(y_indices)), int(np.ptp(x_indices)))

        return position, size


    # -------------- Graphics Processing Tools ---------------- #
    # Image filters
    def filter_depth_image(self, depth_image, method='gaussian'):
        if method == 'gaussian':
            return cv2.GaussianBlur(depth_image, (5, 5), 1)
        elif method == 'median':
            return cv2.medianBlur(depth_image, 5)
        elif method == 'bilateral':
            return cv2.bilateralFilter(depth_image, 9, 75, 75)
        elif method == 'mean':
            return cv2.blur(depth_image, (5, 5))
        else:
            raise ValueError("Unsupported filtering method")
        

    # Image Format Harmonization
    def ensure_same_format(self, images):
        # Ensure all images have the same type and number of rows
        reference_shape = images[0].shape
        reference_type = images[0].dtype
        formatted_images = []

        for img in images:
            # Convert images to the same type
            if img.dtype != reference_type:
                img = img.astype(reference_type)

            if len(img.shape) == 2:  # If it is a grayscale image, convert to BGR
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if img.shape[0] != reference_shape[0]:  # Resize to match the number of rows
                img = cv2.resize(img, (reference_shape[1], reference_shape[0]))
    
            formatted_images.append(img)
        
        return formatted_images