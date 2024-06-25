# This Python file uses the following encoding: utf-8
import sys
from PySide6.QtWidgets import QApplication, QWidget
from ui_form import Ui_Widget
from functionality import Functionality

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py

class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self)
        self.functionality = Functionality(self.ui)
        self._connect_signals()

    def _connect_signals(self):
        self.ui.pushButton_connect.clicked.connect(self.functionality._on_connect)
        self.ui.pushButton_disconnect.clicked.connect(self.functionality._on_disconnect)
        self.ui.pushButton_launch.clicked.connect(self.functionality._on_launch)
        self.ui.pushButton_stop.clicked.connect(self.functionality._on_stop)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.show()
    widget.ui.textEdit.append("Application started. Initializing models...")
    sys.exit(app.exec())