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
from PySide6.QtWidgets import (QApplication, QCheckBox, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QSpacerItem, QTabWidget, QTextEdit,
    QVBoxLayout, QWidget)

class Ui_Widget(object):
    def setupUi(self, Widget):
        if not Widget.objectName():
            Widget.setObjectName(u"Widget")
        Widget.resize(1110, 687)
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
        self.gridLayout = QGridLayout(self.tab_video_display)
        self.gridLayout.setObjectName(u"gridLayout")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
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


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.groupBox_config = QGroupBox(self.tab_video_display)
        self.groupBox_config.setObjectName(u"groupBox_config")
        self.gridLayout_3 = QGridLayout(self.groupBox_config)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.checkBox_config_1 = QCheckBox(self.groupBox_config)
        self.checkBox_config_1.setObjectName(u"checkBox_config_1")

        self.horizontalLayout_4.addWidget(self.checkBox_config_1)

        self.checkBox_config_2 = QCheckBox(self.groupBox_config)
        self.checkBox_config_2.setObjectName(u"checkBox_config_2")

        self.horizontalLayout_4.addWidget(self.checkBox_config_2)

        self.checkBox_config_3 = QCheckBox(self.groupBox_config)
        self.checkBox_config_3.setObjectName(u"checkBox_config_3")

        self.horizontalLayout_4.addWidget(self.checkBox_config_3)


        self.gridLayout_3.addLayout(self.horizontalLayout_4, 0, 0, 1, 1)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_3.addItem(self.horizontalSpacer_3, 0, 1, 1, 1)


        self.verticalLayout.addWidget(self.groupBox_config)

        self.textEdit = QTextEdit(self.tab_video_display)
        self.textEdit.setObjectName(u"textEdit")

        self.verticalLayout.addWidget(self.textEdit)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.pushButton_launch = QPushButton(self.tab_video_display)
        self.pushButton_launch.setObjectName(u"pushButton_launch")

        self.horizontalLayout_3.addWidget(self.pushButton_launch)

        self.pushButton_stop = QPushButton(self.tab_video_display)
        self.pushButton_stop.setObjectName(u"pushButton_stop")

        self.horizontalLayout_3.addWidget(self.pushButton_stop)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer)


        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.verticalLayout.setStretch(0, 1)

        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)

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
        self.groupBox_config.setTitle(QCoreApplication.translate("Widget", u"Config", None))
        self.checkBox_config_1.setText(QCoreApplication.translate("Widget", u"CheckBox", None))
        self.checkBox_config_2.setText(QCoreApplication.translate("Widget", u"CheckBox", None))
        self.checkBox_config_3.setText(QCoreApplication.translate("Widget", u"CheckBox", None))
        self.pushButton_launch.setText(QCoreApplication.translate("Widget", u"Launch", None))
        self.pushButton_stop.setText(QCoreApplication.translate("Widget", u"Stop", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_video_display), QCoreApplication.translate("Widget", u"Display Window", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("Widget", u"Tab 2", None))
    # retranslateUi

