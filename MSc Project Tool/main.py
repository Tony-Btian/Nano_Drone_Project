import tkinter as tk
from tkinter import ttk
import threading
import cv2
from camera_processor import CameraProcessor
from PIL import Image, ImageTk

class MonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Monitor UI")
        self.root.geometry("800x600")  # 设置默认窗口大小为800x600

        # 配置根窗口的行和列，使其可以调整大小
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_rowconfigure(2, weight=0)
        self.root.grid_rowconfigure(3, weight=0)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # 创建两个监视窗口
        self.monitor1 = tk.LabelFrame(root, text="Monitor 1")
        self.monitor1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.monitor2 = tk.LabelFrame(root, text="Monitor 2")
        self.monitor2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # 在监视窗口内放置标签（模拟监视内容）
        self.label1 = tk.Label(self.monitor1)
        self.label1.pack(expand=True, fill='both')
        self.label2 = tk.Label(self.monitor2)
        self.label2.pack(expand=True, fill='both')

        # 创建第一个groupbox
        self.checkbox_group1 = tk.LabelFrame(root, text="Options Group 1")
        self.checkbox_group1.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # 创建第一个groupbox中的勾选选项
        self.option1_var = tk.BooleanVar()
        self.option2_var = tk.BooleanVar()
        self.option3_var = tk.BooleanVar()

        self.option1 = tk.Checkbutton(self.checkbox_group1, text="Option 1", variable=self.option1_var)
        self.option1.pack(anchor='w', padx=5, pady=2)
        self.option2 = tk.Checkbutton(self.checkbox_group1, text="Option 2", variable=self.option2_var)
        self.option2.pack(anchor='w', padx=5, pady=2)
        self.option3 = tk.Checkbutton(self.checkbox_group1, text="Option 3", variable=self.option3_var)
        self.option3.pack(anchor='w', padx=5, pady=2)

        # 创建第二个groupbox
        self.checkbox_group2 = tk.LabelFrame(root, text="Options Group 2")
        self.checkbox_group2.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        # 创建第二个groupbox中的勾选选项
        self.option4_var = tk.BooleanVar()
        self.option5_var = tk.BooleanVar()
        self.option6_var = tk.BooleanVar()

        self.option4 = tk.Checkbutton(self.checkbox_group2, text="Option 4", variable=self.option4_var)
        self.option4.pack(anchor='w', padx=5, pady=2)
        self.option5 = tk.Checkbutton(self.checkbox_group2, text="Option 5", variable=self.option5_var)
        self.option5.pack(anchor='w', padx=5, pady=2)
        self.option6 = tk.Checkbutton(self.checkbox_group2, text="Option 6", variable=self.option6_var)
        self.option6.pack(anchor='w', padx=5, pady=2)

        # 创建配置按钮
        self.config_button = tk.Button(root, text="Config", command=self.config)
        self.config_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        # 创建启动和停止按钮
        self.start_button = tk.Button(root, text="Launch", command=self.start)
        self.start_button.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

        self.stop_button = tk.Button(root, text="Stop", command=self.stop)
        self.stop_button.grid(row=3, column=1, padx=10, pady=10, sticky="ew")

        self.camera_processor = CameraProcessor()
        self.running = False


    def config(self):
        print("Config button clicked")
        print("Option 1:", self.option1_var.get())
        print("Option 2:", self.option2_var.get())
        print("Option 3:", self.option3_var.get())
        print("Option 4:", self.option4_var.get())
        print("Option 5:", self.option5_var.get())
        print("Option 6:", self.option6_var.get())


    def start(self):
        if not self.running:
            self.running = True
            self.camera_thread = threading.Thread(target=self.camera_loop)
            self.camera_thread.start()
            print("Start button clicked")


    def stop(self):
        self.running = False
        print("Stop button clicked")


    def camera_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Unable to open camera")
            return
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Unable to read frame")
                break

            processed_frame = self.camera_processor.estimate_depth(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            frame_image = ImageTk.PhotoImage(Image.fromarray(frame))
            processed_frame_image = ImageTk.PhotoImage(Image.fromarray(processed_frame))

            self.label1.config(image=frame_image)
            self.label1.image = frame_image

            self.label2.config(image=processed_frame_image)
            self.label2.image = processed_frame_image

            self.root.update_idletasks()
            self.root.after(10)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = MonitorApp(root)
    root.mainloop()
