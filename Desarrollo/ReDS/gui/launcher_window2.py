# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'launcher_window_v2.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(690, 554)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMaximumSize(QtCore.QSize(690, 16777215))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("C:/Users/USUARIO/.designer/assets/icons/g868.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMaximumSize(QtCore.QSize(690, 16777215))
        self.centralwidget.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.logoLabel = QtWidgets.QLabel(self.centralwidget)
        self.logoLabel.setText("")
        self.logoLabel.setPixmap(QtGui.QPixmap("../assets/logos/logo_a.png"))
        self.logoLabel.setScaledContents(False)
        self.logoLabel.setObjectName("logoLabel")
        self.verticalLayout_2.addWidget(self.logoLabel)
        self.text1Label = QtWidgets.QLabel(self.centralwidget)
        self.text1Label.setTextFormat(QtCore.Qt.RichText)
        self.text1Label.setObjectName("text1Label")
        self.verticalLayout_2.addWidget(self.text1Label)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_2.addWidget(self.line)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setTextFormat(QtCore.Qt.RichText)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_3.addWidget(self.label_4)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setTextFormat(QtCore.Qt.RichText)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.moduleAPushButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.moduleAPushButton.sizePolicy().hasHeightForWidth())
        self.moduleAPushButton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.moduleAPushButton.setFont(font)
        self.moduleAPushButton.setAutoFillBackground(False)
        self.moduleAPushButton.setStyleSheet("background-color: rgb(212, 212, 212);")
        self.moduleAPushButton.setFlat(False)
        self.moduleAPushButton.setObjectName("moduleAPushButton")
        self.horizontalLayout.addWidget(self.moduleAPushButton)
        self.moduleBPushButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.moduleBPushButton.setFont(font)
        self.moduleBPushButton.setStyleSheet("background-color: rgb(212, 212, 212);")
        self.moduleBPushButton.setObjectName("moduleBPushButton")
        self.horizontalLayout.addWidget(self.moduleBPushButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout_2.addWidget(self.line_2)
        self.lowerLogoLabel = QtWidgets.QLabel(self.centralwidget)
        self.lowerLogoLabel.setText("")
        self.lowerLogoLabel.setPixmap(QtGui.QPixmap("../assets/logos/all_logos3.png"))
        self.lowerLogoLabel.setScaledContents(True)
        self.lowerLogoLabel.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.lowerLogoLabel.setObjectName("lowerLogoLabel")
        self.verticalLayout_2.addWidget(self.lowerLogoLabel)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ReDS"))
        self.text1Label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt;\">Reconstrucción de Datos Sísmicos - ReDS</span></p><p></br></p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt;\">ReDS - RR</span></p><p align=\"center\"><span style=\" font-size:14pt;\">Reconstrucción de Receptores</span></p></body></html>"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt;\">ReDS - RD</span></p><p align=\"center\"><span style=\" font-size:14pt;\">Reconstrucción de Disparos</span></p></body></html>"))
        self.moduleAPushButton.setText(_translate("MainWindow", "Iniciar RR"))
        self.moduleBPushButton.setText(_translate("MainWindow", "Iniciar RD"))
import assets_rc
