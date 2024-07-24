# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.7.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QRadioButton, QSizePolicy, QSpacerItem, QTabWidget,
    QTextEdit, QVBoxLayout, QWidget)

class Ui_Widget(object):
    def setupUi(self, Widget):
        if not Widget.objectName():
            Widget.setObjectName(u"Widget")
        Widget.resize(1110, 632)
        self.gridLayout_2 = QGridLayout(Widget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.lineEdit_addr = QLineEdit(Widget)
        self.lineEdit_addr.setObjectName(u"lineEdit_addr")

        self.horizontalLayout_2.addWidget(self.lineEdit_addr)

        self.pushButton_connect = QPushButton(Widget)
        self.pushButton_connect.setObjectName(u"pushButton_connect")

        self.horizontalLayout_2.addWidget(self.pushButton_connect)

        self.pushButton_disconnect = QPushButton(Widget)
        self.pushButton_disconnect.setObjectName(u"pushButton_disconnect")

        self.horizontalLayout_2.addWidget(self.pushButton_disconnect)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)

        self.horizontalLayout_2.setStretch(3, 2)

        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.tabWidget = QTabWidget(Widget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tab_video_display = QWidget()
        self.tab_video_display.setObjectName(u"tab_video_display")
        self.verticalLayout_6 = QVBoxLayout(self.tab_video_display)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.groupBox_original_display = QGroupBox(self.tab_video_display)
        self.groupBox_original_display.setObjectName(u"groupBox_original_display")
        self.groupBox_original_display.setMinimumSize(QSize(350, 350))
        self.gridLayout_4 = QGridLayout(self.groupBox_original_display)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.video_original = QLabel(self.groupBox_original_display)
        self.video_original.setObjectName(u"video_original")

        self.gridLayout_4.addWidget(self.video_original, 0, 0, 1, 1)


        self.horizontalLayout.addWidget(self.groupBox_original_display)

        self.groupBox_yolo_display = QGroupBox(self.tab_video_display)
        self.groupBox_yolo_display.setObjectName(u"groupBox_yolo_display")
        self.groupBox_yolo_display.setMinimumSize(QSize(350, 350))
        self.gridLayout_5 = QGridLayout(self.groupBox_yolo_display)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.video_yolo = QLabel(self.groupBox_yolo_display)
        self.video_yolo.setObjectName(u"video_yolo")

        self.gridLayout_5.addWidget(self.video_yolo, 0, 0, 1, 1)


        self.horizontalLayout.addWidget(self.groupBox_yolo_display)

        self.groupBox_midas_display = QGroupBox(self.tab_video_display)
        self.groupBox_midas_display.setObjectName(u"groupBox_midas_display")
        self.groupBox_midas_display.setMinimumSize(QSize(350, 350))
        self.gridLayout_6 = QGridLayout(self.groupBox_midas_display)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.video_midas = QLabel(self.groupBox_midas_display)
        self.video_midas.setObjectName(u"video_midas")

        self.gridLayout_6.addWidget(self.video_midas, 0, 0, 1, 1)


        self.horizontalLayout.addWidget(self.groupBox_midas_display)


        self.verticalLayout_6.addLayout(self.horizontalLayout)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.groupBox_yolo_config = QGroupBox(self.tab_video_display)
        self.groupBox_yolo_config.setObjectName(u"groupBox_yolo_config")
        self.gridLayout = QGridLayout(self.groupBox_yolo_config)
        self.gridLayout.setObjectName(u"gridLayout")
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.radioButton_yolov1 = QRadioButton(self.groupBox_yolo_config)
        self.radioButton_yolov1.setObjectName(u"radioButton_yolov1")
        self.radioButton_yolov1.setAutoExclusive(True)

        self.verticalLayout_3.addWidget(self.radioButton_yolov1)

        self.radioButton_yolov2 = QRadioButton(self.groupBox_yolo_config)
        self.radioButton_yolov2.setObjectName(u"radioButton_yolov2")
        self.radioButton_yolov2.setAutoExclusive(True)

        self.verticalLayout_3.addWidget(self.radioButton_yolov2)

        self.radioButton_yolov3 = QRadioButton(self.groupBox_yolo_config)
        self.radioButton_yolov3.setObjectName(u"radioButton_yolov3")
        self.radioButton_yolov3.setAutoExclusive(True)

        self.verticalLayout_3.addWidget(self.radioButton_yolov3)

        self.radioButton_yolov4 = QRadioButton(self.groupBox_yolo_config)
        self.radioButton_yolov4.setObjectName(u"radioButton_yolov4")
        self.radioButton_yolov4.setAutoExclusive(True)

        self.verticalLayout_3.addWidget(self.radioButton_yolov4)


        self.horizontalLayout_4.addLayout(self.verticalLayout_3)

        self.line = QFrame(self.groupBox_yolo_config)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.Shape.VLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.horizontalLayout_4.addWidget(self.line)

        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.radioButton_yolov5 = QRadioButton(self.groupBox_yolo_config)
        self.radioButton_yolov5.setObjectName(u"radioButton_yolov5")
        self.radioButton_yolov5.setChecked(True)
        self.radioButton_yolov5.setAutoExclusive(True)

        self.verticalLayout_4.addWidget(self.radioButton_yolov5)

        self.radioButton_yolov6 = QRadioButton(self.groupBox_yolo_config)
        self.radioButton_yolov6.setObjectName(u"radioButton_yolov6")
        self.radioButton_yolov6.setAutoExclusive(True)

        self.verticalLayout_4.addWidget(self.radioButton_yolov6)

        self.radioButton_yolov7 = QRadioButton(self.groupBox_yolo_config)
        self.radioButton_yolov7.setObjectName(u"radioButton_yolov7")
        self.radioButton_yolov7.setAutoExclusive(True)

        self.verticalLayout_4.addWidget(self.radioButton_yolov7)

        self.radioButton_yolov8 = QRadioButton(self.groupBox_yolo_config)
        self.radioButton_yolov8.setObjectName(u"radioButton_yolov8")
        self.radioButton_yolov8.setAutoExclusive(True)

        self.verticalLayout_4.addWidget(self.radioButton_yolov8)


        self.horizontalLayout_4.addLayout(self.verticalLayout_4)


        self.gridLayout.addLayout(self.horizontalLayout_4, 0, 0, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(17, 13, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer_2, 1, 0, 1, 1)


        self.horizontalLayout_5.addWidget(self.groupBox_yolo_config)

        self.groupBox_midas_config = QGroupBox(self.tab_video_display)
        self.groupBox_midas_config.setObjectName(u"groupBox_midas_config")
        self.verticalLayout = QVBoxLayout(self.groupBox_midas_config)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.radioButton_midas_small = QRadioButton(self.groupBox_midas_config)
        self.radioButton_midas_small.setObjectName(u"radioButton_midas_small")

        self.verticalLayout.addWidget(self.radioButton_midas_small)

        self.radioButton_midas_hybrid = QRadioButton(self.groupBox_midas_config)
        self.radioButton_midas_hybrid.setObjectName(u"radioButton_midas_hybrid")

        self.verticalLayout.addWidget(self.radioButton_midas_hybrid)

        self.radioButton_midas_large = QRadioButton(self.groupBox_midas_config)
        self.radioButton_midas_large.setObjectName(u"radioButton_midas_large")
        self.radioButton_midas_large.setChecked(True)

        self.verticalLayout.addWidget(self.radioButton_midas_large)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)


        self.horizontalLayout_5.addWidget(self.groupBox_midas_config)


        self.verticalLayout_5.addLayout(self.horizontalLayout_5)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.pushButton_loadmodel = QPushButton(self.tab_video_display)
        self.pushButton_loadmodel.setObjectName(u"pushButton_loadmodel")

        self.horizontalLayout_3.addWidget(self.pushButton_loadmodel)

        self.pushButton_launch = QPushButton(self.tab_video_display)
        self.pushButton_launch.setObjectName(u"pushButton_launch")

        self.horizontalLayout_3.addWidget(self.pushButton_launch)

        self.pushButton_stop = QPushButton(self.tab_video_display)
        self.pushButton_stop.setObjectName(u"pushButton_stop")

        self.horizontalLayout_3.addWidget(self.pushButton_stop)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer)


        self.verticalLayout_5.addLayout(self.horizontalLayout_3)


        self.horizontalLayout_6.addLayout(self.verticalLayout_5)

        self.textEdit = QTextEdit(self.tab_video_display)
        self.textEdit.setObjectName(u"textEdit")

        self.horizontalLayout_6.addWidget(self.textEdit)


        self.verticalLayout_6.addLayout(self.horizontalLayout_6)

        self.tabWidget.addTab(self.tab_video_display, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.tabWidget.addTab(self.tab_2, "")

        self.verticalLayout_2.addWidget(self.tabWidget)


        self.gridLayout_2.addLayout(self.verticalLayout_2, 0, 0, 1, 1)


        self.retranslateUi(Widget)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(Widget)
    # setupUi

    def retranslateUi(self, Widget):
        Widget.setWindowTitle(QCoreApplication.translate("Widget", u"Widget", None))
        self.pushButton_connect.setText(QCoreApplication.translate("Widget", u"Connect", None))
        self.pushButton_disconnect.setText(QCoreApplication.translate("Widget", u"Disconnect", None))
        self.groupBox_original_display.setTitle(QCoreApplication.translate("Widget", u"Original", None))
        self.video_original.setText("")
        self.groupBox_yolo_display.setTitle(QCoreApplication.translate("Widget", u"YOLO Display", None))
        self.video_yolo.setText("")
        self.groupBox_midas_display.setTitle(QCoreApplication.translate("Widget", u"MiDas Display", None))
        self.video_midas.setText("")
        self.groupBox_yolo_config.setTitle(QCoreApplication.translate("Widget", u"YOLO", None))
        self.radioButton_yolov1.setText(QCoreApplication.translate("Widget", u"YOLOv1", None))
        self.radioButton_yolov2.setText(QCoreApplication.translate("Widget", u"YOLOv2", None))
        self.radioButton_yolov3.setText(QCoreApplication.translate("Widget", u"YOLOv3", None))
        self.radioButton_yolov4.setText(QCoreApplication.translate("Widget", u"YOLOv4", None))
        self.radioButton_yolov5.setText(QCoreApplication.translate("Widget", u"YOLOv5", None))
        self.radioButton_yolov6.setText(QCoreApplication.translate("Widget", u"YOLOv6", None))
        self.radioButton_yolov7.setText(QCoreApplication.translate("Widget", u"YOLOv7", None))
        self.radioButton_yolov8.setText(QCoreApplication.translate("Widget", u"YOLOv8", None))
        self.groupBox_midas_config.setTitle(QCoreApplication.translate("Widget", u"MiDaS", None))
        self.radioButton_midas_small.setText(QCoreApplication.translate("Widget", u"MiDaSv2.1-Small", None))
        self.radioButton_midas_hybrid.setText(QCoreApplication.translate("Widget", u"MiDaSv3-Hybrid", None))
        self.radioButton_midas_large.setText(QCoreApplication.translate("Widget", u"MiDaSv3-Large", None))
        self.pushButton_loadmodel.setText(QCoreApplication.translate("Widget", u"Load Model", None))
        self.pushButton_launch.setText(QCoreApplication.translate("Widget", u"Launch", None))
        self.pushButton_stop.setText(QCoreApplication.translate("Widget", u"Stop", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_video_display), QCoreApplication.translate("Widget", u"Display Window", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("Widget", u"Tab 2", None))
    # retranslateUi

