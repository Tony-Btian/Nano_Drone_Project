# -*- coding: utf-8 -*-
#
#  ...........       ____  _ __
#  |  ,-^-,  |      / __ )(_) /_______________ _____  ___
#  | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  | / ,..´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#     +.......   /_____/_/\__/\___/_/   \__,_/ /___/\___/

# MIT License

# Copyright (c) 2023 Bitcraze


"""
file: crazyflie_py_wallfollowing.py

Controls the crazyflie and implements a wall following method in webots in Python

Author:   Kimberly McGuire (Bitcraze AB)
"""

import cv2
import numpy as np
import torch

from controller import Robot
from controller import Keyboard

from math import cos, sin

from pid_controller import pid_velocity_fixed_height_controller
from wall_following import WallFollowing
from ultralytics import YOLO

FLYING_ATTITUDE = 1

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 验证CUDA可用性
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Switching to CPU.")


# 加载YOLOv5模型
try:
    yolo_model = YOLO('yolov8s.pt')
except Exception as e:
    print(f"Error loading YOLOv5 model: {e}")


# 加载Midas深度估计模型
try:
    midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', pretrained=True).to(device)
    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    print("MiDaS model loaded successfully.")
except Exception as e:
    print(f"Error loading MiDaS model: {e}")


def objects_detect(image_array):
    object_detection = yolo_model(image_array)
    yolo_display = object_detection[0].plot()
    return yolo_display


def estimate_depth(camera_image):
    # 将图像转换为Midas模型的输入格式
    transform = midas_transforms.dpt_transform
    input_batch = transform(camera_image).to(device)

    # 进行深度估计
    with torch.no_grad():
        prediction = midas(input_batch)
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


def sobel_edge_detection(depth_image):
    # 转换为灰度图像
    gray = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # 计算x方向和y方向的梯度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度幅值
    grad = cv2.magnitude(grad_x, grad_y)

    # 归一化并转换为8位图像
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)
    grad = np.uint8(grad)
    
    return grad


if __name__ == '__main__':

    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # Initialize motors
    m1_motor = robot.getDevice("m1_motor")
    m1_motor.setPosition(float('inf'))
    m1_motor.setVelocity(-1)
    m2_motor = robot.getDevice("m2_motor")
    m2_motor.setPosition(float('inf'))
    m2_motor.setVelocity(1)
    m3_motor = robot.getDevice("m3_motor")
    m3_motor.setPosition(float('inf'))
    m3_motor.setVelocity(-1)
    m4_motor = robot.getDevice("m4_motor")
    m4_motor.setPosition(float('inf'))
    m4_motor.setVelocity(1)

    # Initialize Sensors
    imu = robot.getDevice("inertial_unit")
    imu.enable(timestep)
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    range_front = robot.getDevice("range_front")
    range_front.enable(timestep)
    range_left = robot.getDevice("range_left")
    range_left.enable(timestep)
    range_back = robot.getDevice("range_back")
    range_back.enable(timestep)
    range_right = robot.getDevice("range_right")
    range_right.enable(timestep)

    # Get keyboard
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # Initialize variables
    past_x_global = 0
    past_y_global = 0
    past_time = 0
    first_time = True

    # 准备棋盘格尺寸
    chessboard_size = (9, 6)
    square_size = 1.0  # 每个方块的实际大小

    # 准备世界坐标
    objp = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # 存储棋盘格角点
    obj_points = []  # 3D 点
    img_points = []  # 2D 图像点

    # Crazyflie velocity PID controller
    PID_crazyflie = pid_velocity_fixed_height_controller()
    PID_update_last_time = robot.getTime()
    sensor_read_last_time = robot.getTime()

    height_desired = FLYING_ATTITUDE

    wall_following = WallFollowing(angle_value_buffer=0.01, reference_distance_from_wall=0.5,
                                   max_forward_speed=0.3, init_state=WallFollowing.StateWallFollowing.FORWARD)

    autonomous_mode = False
    image_process_mode = False
    yolo_process_mode = False
    depth_process_mode = False

    print("\n")

    print("====== Controls =======\n\n")

    print(" The Crazyflie can be controlled from your keyboard!\n")
    print(" All controllable movement is in body coordinates\n")
    print("- Use the up, back, right and left button to move in the horizontal plane\n")
    print("- Use Q and E to rotate around yaw\n")
    print("- Use W and S to go up and down\n")
    print("- Press A to start autonomous mode\n")
    print("- Press D to disable autonomous mode\n ")


    # Main loop:
    while robot.step(timestep) != -1:

        dt = robot.getTime() - past_time
        actual_state = {}

        if first_time:
            past_x_global = gps.getValues()[0]
            past_y_global = gps.getValues()[1]
            past_time = robot.getTime()
            first_time = False

        # Get sensor data
        roll = imu.getRollPitchYaw()[0]
        pitch = imu.getRollPitchYaw()[1]
        yaw = imu.getRollPitchYaw()[2]
        yaw_rate = gyro.getValues()[2]
        x_global = gps.getValues()[0]
        v_x_global = (x_global - past_x_global)/dt
        y_global = gps.getValues()[1]
        v_y_global = (y_global - past_y_global)/dt
        altitude = gps.getValues()[2]

        # Get body fixed velocities
        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)
        v_x = v_x_global * cos_yaw + v_y_global * sin_yaw
        v_y = - v_x_global * sin_yaw + v_y_global * cos_yaw

        # Initialize values
        desired_state = [0, 0, 0, 0]
        forward_desired = 0
        sideways_desired = 0
        yaw_desired = 0
        height_diff_desired = 0

        key = keyboard.getKey()
        while key > 0:
            if key == Keyboard.UP:
                forward_desired += 0.5
            elif key == Keyboard.DOWN:
                forward_desired -= 0.5
            elif key == Keyboard.RIGHT:
                sideways_desired -= 0.5
            elif key == Keyboard.LEFT:
                sideways_desired += 0.5
            elif key == ord('Q'):
                yaw_desired = + 1
            elif key == ord('E'):
                yaw_desired = - 1
            elif key == ord('W'):
                height_diff_desired = 0.1
            elif key == ord('S'):
                height_diff_desired = - 0.1
            elif key == ord('A'):
                if autonomous_mode is False:
                    autonomous_mode = True
                    print("Autonomous mode: ON")
            elif key == ord('D'):
                if autonomous_mode is True:
                    autonomous_mode = False
                    print("Autonomous mode: OFF")
            elif key == ord('N'):
                if image_process_mode is False:
                    image_process_mode = True
                    print("Image process mode: ON")
            elif key == ord('M'):
                if image_process_mode is True:
                    image_process_mode = False
                    print("Image process mode: OFF")
            key = keyboard.getKey()

        height_desired += height_diff_desired * dt

        # Converting images to NumPy arrays
        camera_data = camera.getImage()

        if image_process_mode:
            # 打印图像原始数据类型和大小
            width = camera.getWidth()
            height = camera.getHeight()
            image_array = np.frombuffer(camera_data, np.uint8).reshape((height, width, 4))
        
            # 去掉alpha通道，确保图像为RGB格式
            image_array = image_array[:, :, :3]
        
            # 确保图像数组的形状为(height, width, 3)
            if image_array.shape[2] == 3:

                # 显示图像
                # display_image = image_array

                # 使用 YOLO 处理图像
                yolo_display = objects_detect(image_array)

                # 使用 MiDas 处理图像
                depth_display = estimate_depth(image_array)

                # 对深度图进行处理
                # 确认深度图格式和大小
                print("Depth image dtype:", depth_display.dtype)
                print("Depth image shape:", depth_display.shape)
                print("Depth image min value:", np.min(depth_display))
                print("Depth image max value:", np.max(depth_display))
                print("Depth image mean value:", np.mean(depth_display))

                gray_resized = cv2.resize(depth_display, (image_array.shape[1], image_array.shape[0]))
                filtered_image = filter_depth_image(gray_resized, method='gaussian')


                # 创建双视图图像
                frame_top_row = cv2.hconcat([image_array, filtered_image])
                frame_bottom_row = cv2.hconcat([yolo_display, depth_display])
                quad_frame = cv2.vconcat([frame_top_row, frame_bottom_row])

                # 显示图像
                cv2.imshow('Camera Image', quad_frame)

                # 处理键盘事件
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    break

            else:
                print("Image array does not have 3 channels after removing alpha channel")
        
        # get range in meters
        range_front_value = range_front.getValue() / 1000
        range_right_value = range_right.getValue() / 1000
        range_left_value = range_left.getValue() / 1000

        # Choose a wall following direction
        # if you choose direction left, use the right range value
        # if you choose direction right, use the left range value
        direction = WallFollowing.WallFollowingDirection.LEFT
        range_side_value = range_right_value

        # Get the velocity commands from the wall following state machine
        cmd_vel_x, cmd_vel_y, cmd_ang_w, state_wf = wall_following.wall_follower(
            range_front_value, range_side_value, yaw, direction, robot.getTime())

        if autonomous_mode:
            sideways_desired = cmd_vel_y
            forward_desired = cmd_vel_x
            yaw_desired = cmd_ang_w

        # PID velocity controller with fixed height
        motor_power = PID_crazyflie.pid(dt, forward_desired, sideways_desired,
                                        yaw_desired, height_desired,
                                        roll, pitch, yaw_rate,
                                        altitude, v_x, v_y)

        m1_motor.setVelocity(-motor_power[0])
        m2_motor.setVelocity(motor_power[1])
        m3_motor.setVelocity(-motor_power[2])
        m4_motor.setVelocity(motor_power[3])

        past_time = robot.getTime()
        past_x_global = x_global
        past_y_global = y_global
    cv2.destroyAllWindows()
