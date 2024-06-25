import sys
import psutil
import time
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QPen

class CpuGpuMonitor(QThread):
    update_usage = Signal(list, list)

    def __init__(self):
        super().__init__()
        self.cpu_data = []
        self.gpu_data = []
        self.max_data_points = 100
        self._running = True

    def run(self):
        while self._running:
            cpu_usage = psutil.cpu_percent(interval=1)
            try:
                import nvidia_smi
                nvidia_smi.nvmlInit()
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
                gpu_usage = nvidia_smi.nvmlDeviceGetUtilizationRates(handle).gpu
            except ImportError:
                gpu_usage = 0

            if len(self.cpu_data) >= self.max_data_points:
                self.cpu_data.pop(0)
            if len(self.gpu_data) >= self.max_data_points:
                self.gpu_data.pop(0)

            self.cpu_data.append(cpu_usage)
            self.gpu_data.append(gpu_usage)

            self.update_usage.emit(self.cpu_data, self.gpu_data)

            time.sleep(1)

    def stop(self):
        self._running = False

class CpuGpuWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(800, 400)
        self.setAutoFillBackground(True)
        self.set_background_color(QColor(255, 255, 255))
        self.cpu_data = []
        self.gpu_data = []

    def set_background_color(self, color):
        palette = self.palette()
        palette.setColor(self.backgroundRole(), color)
        self.setPalette(palette)

    def update_data(self, cpu_data, gpu_data):
        self.cpu_data = cpu_data
        self.gpu_data = gpu_data
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        pen_cpu = QPen(QColor(255, 0, 0), 2)
        pen_gpu = QPen(QColor(0, 0, 255), 2)
        
        painter.setPen(pen_cpu)
        self.draw_lines(painter, self.cpu_data)
        
        painter.setPen(pen_gpu)
        self.draw_lines(painter, self.gpu_data)

    def draw_lines(self, painter, data):
        if not data:
            return
        width = self.width()
        height = self.height()
        max_data_value = 100

        x_scale = width / len(data)
        y_scale = height / max_data_value

        path = []
        for i, value in enumerate(data):
            x = i * x_scale
            y = height - (value * y_scale)
            path.append((x, y))

        for i in range(len(path) - 1):
            painter.drawLine(path[i][0], path[i][1], path[i + 1][0], path[i + 1][1])
