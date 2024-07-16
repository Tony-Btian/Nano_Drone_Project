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

from controller import Robot
from controller import Keyboard
from math import cos, sin
from pid_controller import pid_velocity_fixed_height_controller
from wall_following import WallFollowing
from ultralytics import YOLO
from image_processing import depth_estimation_and_object_recognition

FLYING_ATTITUDE = 1

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

    # Image Processor
    Image_Processor = depth_estimation_and_object_recognition()

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
                yolo_display = Image_Processor.objects_detect(image_array)

                # 使用 MiDas 处理图像
                depth_display = Image_Processor.estimate_depth(image_array)

                # 对深度图进行处理
                # 确认深度图格式和大小
                # print("Depth image dtype:", depth_display.dtype)
                # print("Depth image shape:", depth_display.shape)
                # print("Depth image min value:", np.min(depth_display))
                # print("Depth image max value:", np.max(depth_display))
                # print("Depth image mean value:", np.mean(depth_display))

                gray_resized = cv2.resize(depth_display, (image_array.shape[1], image_array.shape[0]))
                # filtered_image = filter_depth_image(gray_resized, method='gaussian')

                # 边缘检测
                edges_image = Image_Processor.sobel_edge_detection(depth_display)  # Sobel 算子
                # edges_image = canny_edge_detection(depth_display)  # Canny 算子

                # Canny边缘检测

                # 障碍物检测
                depth_image = np.random.uniform(0, 2, (480, 640)).astype(np.float32)  # 替换为实际的深度图像
                edge_image = np.random.randint(0, 256, (480, 640), dtype=np.uint8)

                obstacle_image, contours = Image_Processor.detect_obstacles(depth_image, edge_image)


                # 根据需要转换和调整图像
                images = [image_array, depth_display, edges_image, obstacle_image]
                formatted_images = Image_Processor.ensure_same_format(images)

                # 创建三视图图像
                tripple_viewer = cv2.hconcat(formatted_images)

                # 显示图像
                cv2.imshow('Camera Image', tripple_viewer)

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
