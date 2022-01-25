from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QDesktopWidget
from pathlib import Path
import gui.resources.res
from about_window import Ui_AboutWindow
from results_window import UI_Results_Window
import numpy as np
from gui.alerts import *

'''
Parameters from Ipynb notebook
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
from skimage.metrics import structural_similarity as ssim
from skimage import data, img_as_float
from skimage.metrics import mean_squared_error
import math

from Algorithms.Function import *
from Algorithms import Function
import scipy


# help(Function.soft_threshold)


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        self.logger = ''
        self.MainWindow = MainWindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(500, 800)  # 349, 617
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.inputGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.inputGroupBox.setFont(font)
        self.inputGroupBox.setObjectName("inputGroupBox")
        self.inputVerticalLayout = QtWidgets.QVBoxLayout(self.inputGroupBox)
        self.inputVerticalLayout.setObjectName("inputVerticalLayout")
        self.loadDataHorizontalLayout = QtWidgets.QHBoxLayout()
        self.loadDataHorizontalLayout.setObjectName("loadDataHorizontalLayout")
        self.seismicDataLabel = QtWidgets.QLabel(self.inputGroupBox)
        self.seismicDataLabel.setObjectName("seismicDataLabel")
        self.loadDataHorizontalLayout.addWidget(self.seismicDataLabel)
        self.lineEdit = QtWidgets.QLineEdit(self.inputGroupBox)
        self.lineEdit.setObjectName("lineEdit")
        self.loadDataHorizontalLayout.addWidget(self.lineEdit)
        self.loadPushButton = QtWidgets.QPushButton(self.inputGroupBox)
        self.loadPushButton.setObjectName("loadPushButton")
        self.loadDataHorizontalLayout.addWidget(self.loadPushButton)
        self.inputVerticalLayout.addLayout(self.loadDataHorizontalLayout)
        self.algorithmHorizontalLayout = QtWidgets.QHBoxLayout()
        self.algorithmHorizontalLayout.setObjectName("algorithmHorizontalLayout")
        self.algorithmLabel = QtWidgets.QLabel(self.inputGroupBox)
        self.algorithmLabel.setObjectName("algorithmLabel")
        self.algorithmHorizontalLayout.addWidget(self.algorithmLabel)
        self.algorithmComboBox = QtWidgets.QComboBox(self.inputGroupBox)
        self.algorithmComboBox.setObjectName("algorithmComboBox")
        self.algorithmComboBox.addItem("")
        self.algorithmComboBox.addItem("")
        self.algorithmComboBox.addItem("")
        # self.algorithmComboBox.addItem("")  # ADMM not working
        self.algorithmHorizontalLayout.addWidget(self.algorithmComboBox)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.algorithmHorizontalLayout.addItem(spacerItem)
        self.inputVerticalLayout.addLayout(self.algorithmHorizontalLayout)
        self.verticalLayout_2.addWidget(self.inputGroupBox)
        self.paramsGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.paramsGroupBox.setObjectName("paramsGroupBox")
        self.paramVerticalLayout = QtWidgets.QVBoxLayout(self.paramsGroupBox)
        self.paramVerticalLayout.setObjectName("paramVerticalLayout")

        self.param0HorizontalLayout = QtWidgets.QHBoxLayout()
        self.param0HorizontalLayout.setObjectName("param0HorizontalLayout")
        self.param0Label = QtWidgets.QLabel(self.paramsGroupBox)
        self.param0Label.setObjectName("param0Label")
        self.param0HorizontalLayout.addWidget(self.param0Label)
        self.param0LineEdit = QtWidgets.QLineEdit(self.paramsGroupBox)
        self.param0LineEdit.setObjectName("param0LineEdit")
        self.param0HorizontalLayout.addWidget(self.param0LineEdit)
        self.paramVerticalLayout.addLayout(self.param0HorizontalLayout)

        self.param1HorizontalLayout = QtWidgets.QHBoxLayout()
        self.param1HorizontalLayout.setObjectName("param1HorizontalLayout")
        self.param1Label = QtWidgets.QLabel(self.paramsGroupBox)
        self.param1Label.setObjectName("param1Label")
        self.param1HorizontalLayout.addWidget(self.param1Label)
        self.param1LineEdit = QtWidgets.QLineEdit(self.paramsGroupBox)
        self.param1LineEdit.setObjectName("param1LineEdit")
        self.param1HorizontalLayout.addWidget(self.param1LineEdit)
        self.paramVerticalLayout.addLayout(self.param1HorizontalLayout)

        self.param2HorizontalLayout = QtWidgets.QHBoxLayout()
        self.param2HorizontalLayout.setObjectName("param2HorizontalLayout")
        self.param2Label = QtWidgets.QLabel(self.paramsGroupBox)
        self.param2Label.setObjectName("param2Label")
        self.param2HorizontalLayout.addWidget(self.param2Label)
        self.param2LineEdit = QtWidgets.QLineEdit(self.paramsGroupBox)
        self.param2LineEdit.setObjectName("param2LineEdit")
        self.param2HorizontalLayout.addWidget(self.param2LineEdit)
        self.paramVerticalLayout.addLayout(self.param2HorizontalLayout)
        self.param3HorizontalLayout = QtWidgets.QHBoxLayout()
        self.param3HorizontalLayout.setObjectName("param3HorizontalLayout")
        self.param3Label = QtWidgets.QLabel(self.paramsGroupBox)
        self.param3Label.setObjectName("param3Label")
        self.param3HorizontalLayout.addWidget(self.param3Label)
        self.param3LineEdit = QtWidgets.QLineEdit(self.paramsGroupBox)
        self.param3LineEdit.setObjectName("param3LineEdit")
        self.param3HorizontalLayout.addWidget(self.param3LineEdit)
        self.paramVerticalLayout.addLayout(self.param3HorizontalLayout)
        self.param4HorizontalLayout = QtWidgets.QHBoxLayout()
        self.param4HorizontalLayout.setObjectName("param4HorizontalLayout")
        self.param4Label = QtWidgets.QLabel(self.paramsGroupBox)
        self.param4Label.setEnabled(True)
        self.param4Label.setObjectName("param4Label")
        self.param4HorizontalLayout.addWidget(self.param4Label)
        self.param4LineEdit = QtWidgets.QLineEdit(self.paramsGroupBox)
        self.param4LineEdit.setObjectName("param4LineEdit")
        self.param4HorizontalLayout.addWidget(self.param4LineEdit)
        self.paramVerticalLayout.addLayout(self.param4HorizontalLayout)
        self.verticalLayout_2.addWidget(self.paramsGroupBox)
        self.outputGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.outputGroupBox.setObjectName("outputGroupBox")
        self.verticalLayout_1 = QtWidgets.QVBoxLayout(self.outputGroupBox)
        self.verticalLayout_1.setObjectName("verticalLayout_1")
        self.outputVerticalLayout = QtWidgets.QVBoxLayout()
        self.outputVerticalLayout.setObjectName("outputVerticalLayout")
        self.saveAsHorizontalLayout = QtWidgets.QHBoxLayout()
        self.saveAsHorizontalLayout.setObjectName("saveAsHorizontalLayout")
        self.saveAsLabel = QtWidgets.QLabel(self.outputGroupBox)
        self.saveAsLabel.setObjectName("saveAsLabel")
        self.saveAsHorizontalLayout.addWidget(self.saveAsLabel)
        self.saveAsLineEdit = QtWidgets.QLineEdit(self.outputGroupBox)
        self.saveAsLineEdit.setObjectName("saveAsLineEdit")
        self.saveAsHorizontalLayout.addWidget(self.saveAsLineEdit)
        self.outputVerticalLayout.addLayout(self.saveAsHorizontalLayout)
        self.progressBar = QtWidgets.QProgressBar(self.outputGroupBox)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.outputVerticalLayout.addWidget(self.progressBar)
        self.loggerHorizontalLayout = QtWidgets.QHBoxLayout()
        self.loggerHorizontalLayout.setObjectName("loggerHorizontalLayout")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.loggerHorizontalLayout.addItem(spacerItem1)
        self.loggerLabel = QtWidgets.QLabel(self.outputGroupBox)
        self.loggerLabel.setObjectName("loggerLabel")
        self.loggerHorizontalLayout.addWidget(self.loggerLabel)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.loggerHorizontalLayout.addItem(spacerItem2)
        self.outputVerticalLayout.addLayout(self.loggerHorizontalLayout)
        self.textEdit = QtWidgets.QTextEdit(self.outputGroupBox)
        self.textEdit.setObjectName("textEdit")
        self.outputVerticalLayout.addWidget(self.textEdit)
        self.verticalLayout_1.addLayout(self.outputVerticalLayout)
        self.verticalLayout_2.addWidget(self.outputGroupBox)
        self.startHorizontalLayout = QtWidgets.QHBoxLayout()
        self.startHorizontalLayout.setObjectName("startHorizontalLayout")
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.startHorizontalLayout.addItem(spacerItem3)
        self.startPushButton = QtWidgets.QPushButton(self.centralwidget)
        self.startPushButton.setObjectName("startPushButton")
        self.startHorizontalLayout.addWidget(self.startPushButton)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.startHorizontalLayout.addItem(spacerItem4)
        self.verticalLayout_2.addLayout(self.startHorizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        self.mainToolBar.setEnabled(True)
        self.mainToolBar.setMovable(False)
        self.mainToolBar.setFloatable(False)
        self.mainToolBar.setObjectName("mainToolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.AboutAction = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/info_icon/info_icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.AboutAction.setIcon(icon)
        self.AboutAction.setObjectName("AboutAction")

        self.showResultsAction = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/info_icon/noun-checked-results.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.showResultsAction.setIcon(icon1)
        self.showResultsAction.setObjectName("showResultsAction")

        self.mainToolBar.addAction(self.AboutAction)
        self.mainToolBar.addAction(self.showResultsAction)

        self.AboutAction.triggered.connect(self.show_about_window)
        self.showResultsAction.triggered.connect(self.show_results_window)
        self.loadPushButton.clicked.connect(self.browseFiles)
        self.active_folder = ''
        self.onlydouble = QtGui.QDoubleValidator(decimals=10)

        self.onlyInt = QtGui.QIntValidator()
        self.param4LineEdit.setValidator(self.onlyInt)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.progressBar.setValue(0)

        self.startPushButton.clicked.connect(self.run)
        self.textEdit.setReadOnly(True)

        self.algorithmComboBox.currentTextChanged.connect(self.on_algorithm_changed)

        self.update_parameters_info(self.algorithmComboBox.currentText())

    def browseFiles(self):
        self.fname = QFileDialog.getOpenFileName(self.MainWindow, 'Open File', filter='numpy file (*.npy)')
        self.lineEdit.setText(self.fname[0].split("/")[-1].strip("/.npy"))

    def show_about_window(self):
        self.about_window = QtWidgets.QWidget()
        self.ui_about_window = Ui_AboutWindow()
        self.ui_about_window.setupUi(self.about_window)
        self.about_window.show()

    def show_results_window(self):
        self.show_results_window = QtWidgets.QWidget()
        self.ui_results_window = UI_Results_Window()
        self.ui_results_window.setupUi(self.show_results_window)
        self.show_results_window.show()

    def on_algorithm_changed(self, value):
        self.update_parameters_info(value)
        self.textEdit.setText('')

    def update_parameters_info(self, algorithm):
        self.param0Label.setVisible(True)
        self.param0LineEdit.setVisible(True)
        self.param1Label.setVisible(True)
        self.param1LineEdit.setVisible(True)
        self.param2Label.setVisible(True)
        self.param2LineEdit.setVisible(True)
        self.param3Label.setVisible(True)
        self.param3LineEdit.setVisible(True)
        self.param4Label.setVisible(True)
        self.param4LineEdit.setVisible(True)

        if algorithm == "FISTA":
            self.param1Label.setText("Tau")
            self.param1LineEdit.setText(str(0.1))
            self.param2Label.setText("Mu")
            self.param2LineEdit.setText(str(0.3))
            self.param1LineEdit.setValidator(self.onlyInt)
            self.param1LineEdit.setValidator(self.onlydouble)
            self.param2LineEdit.setValidator(self.onlydouble)

            self.param3Label.setVisible(False)
            self.param3LineEdit.setVisible(False)
            self.param4Label.setVisible(False)
            self.param4LineEdit.setVisible(False)
            # self.param3LineEdit.setValidator(self.onlydouble)

        elif algorithm == "GAP":
            self.param1Label.setText("Tau")
            self.param1LineEdit.setText(str(1.0))
            self.param1LineEdit.setValidator(self.onlydouble)
            self.param2Label.setVisible(False)
            self.param2LineEdit.setVisible(False)
            self.param3Label.setVisible(False)
            self.param3LineEdit.setVisible(False)
            self.param4Label.setVisible(False)
            self.param4LineEdit.setVisible(False)

        elif algorithm == "TWIST":
            self.param1Label.setText("Tau")
            self.param1LineEdit.setText(str(0.9))
            self.param2Label.setText("Alpha")
            self.param2LineEdit.setText(str(1.2))
            self.param3Label.setText("Beta")
            self.param3LineEdit.setText(str(1.998))

            self.param1LineEdit.setValidator(self.onlydouble)
            self.param2LineEdit.setValidator(self.onlydouble)
            self.param3LineEdit.setValidator(self.onlydouble)

            self.param4Label.setVisible(False)
            self.param4LineEdit.setVisible(False)

        elif algorithm == "ADMM":
            self.param1Label.setText("Rho")
            self.param1LineEdit.setText(str(0.5))
            self.param2Label.setText("Gamma")
            self.param2LineEdit.setText(str(1.0))
            self.param3Label.setText("Lambda")
            self.param3LineEdit.setText(str(0.0078))

            self.param1LineEdit.setValidator(self.onlydouble)
            self.param2LineEdit.setValidator(self.onlydouble)
            self.param3LineEdit.setValidator(self.onlydouble)

            self.param4Label.setVisible(False)
            self.param4LineEdit.setVisible(False)
        else:
            raise Exception("Invalid Algorithm Name")

        self.param0Label.setMinimumWidth(80)
        self.param1Label.setMinimumWidth(80)
        self.param2Label.setMinimumWidth(80)
        self.param3Label.setMinimumWidth(80)
        self.param4Label.setMinimumWidth(80)

    def check_params_algorithm(self, algorithm):

        if self.param0LineEdit.text().strip() == '':
            showWarning("Para iniciar, debe completar todos los parámetros.")
        elif algorithm == "FISTA":
            if self.param1LineEdit.text().strip() == '' or self.param2LineEdit.text().strip() == '':
                showWarning("Para iniciar, debe completar todos los parámetros.")
        elif algorithm == "GAP":
            if self.param1LineEdit.text().strip() == '':
                showWarning("Para iniciar, debe completar todos los parámetros.")
        elif algorithm == "TWIST":
            if (self.param1LineEdit.text().strip() == '' or self.param2LineEdit.text().strip() == '' or
                    self.param3LineEdit.text().strip() == ''):
                showWarning("Para iniciar, debe completar todos los parámetros.")
        elif algorithm == "ADMM":
            if (self.param1LineEdit.text().strip() == '' or self.param2LineEdit.text().strip() == '' or
                    self.param3LineEdit.text().strip() == ''):
                showWarning("Para iniciar, debe completar todos los parámetros.")
        else:
            raise Exception("Invalid Algorithm Name")

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Proyecto 9836"))
        self.inputGroupBox.setTitle(_translate("MainWindow", "Entradas"))
        self.seismicDataLabel.setText(_translate("MainWindow", "Dato Sísmico"))
        self.loadPushButton.setText(_translate("MainWindow", "Cargar"))
        self.algorithmLabel.setText(_translate("MainWindow", "Algoritmo      "))
        self.algorithmComboBox.setItemText(0, _translate("MainWindow", "FISTA"))
        self.algorithmComboBox.setItemText(1, _translate("MainWindow", "GAP"))
        self.algorithmComboBox.setItemText(2, _translate("MainWindow", "TWIST"))
        # self.algorithmComboBox.setItemText(3, _translate("MainWindow", "ADMM")) # not working
        self.paramsGroupBox.setTitle(_translate("MainWindow", "Parámetros"))
        self.param0Label.setText(_translate("MainWindow", "Max Iter"))
        self.param0LineEdit.setText(_translate("MainWindow", "10"))
        self.param0LineEdit.setValidator(self.onlyInt)
        self.param1Label.setText(_translate("MainWindow", "Muestreo en X"))
        self.param2Label.setText(_translate("MainWindow", "Muestreo en Y"))
        self.param3Label.setText(_translate("MainWindow", "Muestreo en Z"))
        self.param4Label.setText(_translate("MainWindow", "No. de  Shots  "))
        self.outputGroupBox.setTitle(_translate("MainWindow", "Salidas"))
        self.saveAsLabel.setText(_translate("MainWindow", "Guardar como"))
        self.loggerLabel.setText(_translate("MainWindow", "Logger"))
        self.startPushButton.setText(_translate("MainWindow", "Iniciar"))
        self.mainToolBar.setWindowTitle(_translate("MainWindow", "mainToolBar"))
        self.AboutAction.setText(_translate("MainWindow", "Acerca de"))
        self.AboutAction.setToolTip(
            _translate("MainWindow", "<html><head/><body><p>Acerca de este proyecto</p></body></html>"))
        self.showResultsAction.setText(_translate("MainWindow", "show results"))
        self.showResultsAction.setToolTip(_translate("MainWindow", "Ver y generar reportes de resultados"))

    def run(self):
        # checks
        if not hasattr(self, 'fname') or self.fname is None:
            showWarning("Para iniciar, debe cargar el dato sísmico dando click al boton 'Cargar'")
            return

        self.check_params_algorithm(self.algorithmComboBox.currentText())

        if self.saveAsLineEdit.text().strip() == '':
            showWarning("Por favor seleccione un nombre de archivo para guardar los resultados del algoritmo.")
            return

        try:
            # I took this values from ipynb notebook
            self.progressBar.setValue(0)

            self.maxiter = int(self.param0LineEdit.text())
            x = np.load(self.fname[0])
            x = x.T
            x = x / np.abs(x).max()

            '''
            ---------------  SAMPLING --------------------
            '''
            sr_rand = 0.5  # 1-compression
            y_rand, pattern_rand, pattern_index = random_sampling(x, sr_rand)
            H = pattern_index

            # Sampling pattern
            H0 = np.tile(pattern_rand.reshape(1, -1), (x.shape[0], 1))

            # save sampling data
            self.sampling_dict = {
                "x_ori": x,
                "sr_rand": sr_rand,
                "y_rand": y_rand,
                "pattern_rand": pattern_rand,
                "pattern_index": pattern_index,
                "H": H,
                "H0": H0
            }
            ''' ---------- Visualization of SAMPLING----------
            '''
            '''
            fig, axs = plt.subplots(1, 3, dpi=250, figsize=(20, 4))
            fig.suptitle('Inputs')

            axs[0].imshow(x, cmap='seismic', aspect='auto')
            axs[0].set_title('Complete data')
            axs[0].set_ylabel('Time (s)')
            axs[0].set_xlabel('Receivers')

            axs[1].imshow(H0, cmap='gray', aspect='auto')
            axs[1].set_title('Sampling Pattern')

            axs[2].imshow(y_rand, cmap='seismic', aspect='auto')
            axs[2].set_title('Incomplete Data')
            axs[2].set_ylabel('Time (s)')
            axs[2].set_xlabel('Receivers')
            '''

            self.textEdit.clear()
            self.logger = f'Running: {self.algorithmComboBox.currentText()} \n'
            self.logger += 'iterations \t ||x-xold|| \t PSNR \n'

            self.thread = QtCore.QThread()

            # Algorithms object

            Alg = Algorithms(x, H, 'DCT2D', 'IDCT2D')  # Assuming using DCT2D ad IDCT2D for all algorithms

            if self.algorithmComboBox.currentText() == "FISTA":
                parameters = {
                    "lmb": float(self.param1LineEdit.text()),  # Tau
                    "mu": float(self.param2LineEdit.text()),  # Mu
                    "max_itr": self.maxiter
                }
                func = Alg.FISTA
            elif self.algorithmComboBox.currentText() == "GAP":
                parameters = {
                    "lmb": float(self.param1LineEdit.text()),  # Tau
                    "max_itr": self.maxiter
                }
                func = Alg.GAP
            elif self.algorithmComboBox.currentText() == "TWIST":
                parameters = {
                    "lmb": float(self.param1LineEdit.text()),  # Tau
                    "alpha": float(self.param2LineEdit.text()),  # Alpha
                    "beta": float(self.param3LineEdit.text()),  # Beta
                    "max_itr": self.maxiter
                }
                func = Alg.TwIST
            elif self.algorithmComboBox.currentText() == "ADMM":
                parameters = {
                    "rho": float(self.param1LineEdit.text()),  # Rho
                    "gamma": float(self.param2LineEdit.text()),  # Gamma
                    "lamnda": float(self.param3LineEdit.text()),  # Lambda
                    "max_itr": self.maxiter
                }
                func = Alg.ADMM
            else:
                showCritical(
                    "No se encontró el algoritmo. Por favor intente nuevamente o utilice un algoritmo diferente.")
                self.progressBar.setValue(0)
                return
            # elif self.algorithmComboBox.currentText() == ""

            self.worker = Worker(func, parameters, self.maxiter)

            self.worker.moveToThread(self.thread)

            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.progress.connect(self.reportProgress)

            self.thread.start()

            self.worker.finished.connect(self.save_results)  # save results

            # Final resets
            self.startPushButton.setEnabled(False)
            self.thread.finished.connect(self.reset_values)

        except BaseException as err:
            msg = f"Unexpected {err=}, {type(err)=}"
            showCritical("Ocurrió un error inesperado al procesar el dato sísmico. Por favor, intente nuevamente o "
                         "utilice un dato diferente.", details=msg)
            self.progressBar.setValue(0)
            return

    def reportProgress(self, iter_val, err, psnr):
        self.progressBar.setValue(int((iter_val / self.maxiter) * 100))
        self.logger += f'iter: {iter_val} \t  error: {err} \t  PSNR: {psnr} \n'
        self.textEdit.setText(self.logger)
        self.textEdit.moveCursor(QtGui.QTextCursor.End)
        # self.stepLabel.setText(f"Long-Running Step: {n}")

    def reset_values(self):
        self.startPushButton.setEnabled(True)
        self.progressBar.setValue(0)
        # self.logger = ''
        self.maxiter = 0
        self.worker = None
        self.thread = None
        ee.progress = None
        self.sampling_dict = None

    def save_results(self, res_dict):
        pwd = os.getcwd()
        save_path = pwd + "/Results"
        Path(save_path).mkdir(parents=True, exist_ok=True)
        filepath = save_path + "/" + self.saveAsLineEdit.text() + ".npz"
        np.savez(filepath, x_result=res_dict['result'], hist=res_dict['hist'], sampling=self.sampling_dict)
        print("Results saved [Ok]")


# class Worker(QtCore.QObject):
#     finished = QtCore.pyqtSignal(dict)
#     progress = QtCore.pyqtSignal(int, str, str)
#
#     def __init__(self, function, parameters):
#         super().__init__()
#         self.function = function
#         self.parameters = parameters
#
#     def run(self):
#         ee.progress = self.progress
#
#         @ee.on("algorithm_update")
#         def handler(iter, err, psnr):
#             ee.progress.emit(iter, err, psnr)
#
#         # Alg.FISTA(tau, mu, self.maxiter)
#         x_result, hist = self.function(**self.parameters)
#
#         self.finished.emit({'result': x_result, 'hist': hist})


class Worker(QtCore.QObject):
    finished = QtCore.pyqtSignal(dict)
    progress = QtCore.pyqtSignal(int, str, str)

    def __init__(self, function, parameters, maxiter):
        super().__init__()
        self.function = function
        self.parameters = parameters
        self.maxiter = maxiter

    def run(self):
        generator = self.function(**self.parameters)
        for itr, err, psnr in generator:
            self.progress.emit(itr, err, psnr)

            if itr == self.maxiter:
                break

        # get last yield
        x_result, hist = next(generator)

        self.finished.emit({'result': x_result, 'hist': hist})


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    qtRectangle = MainWindow.frameGeometry()
    centerPoint = QDesktopWidget().availableGeometry().center()
    qtRectangle.moveCenter(centerPoint)
    MainWindow.move(qtRectangle.topLeft())
    enterPoint = QDesktopWidget().availableGeometry().center()
    qtRectangle.moveCenter(centerPoint)
    MainWindow.move(qtRectangle.topLeft())

    MainWindow.show()
    sys.exit(app.exec_())
