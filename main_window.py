from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QDesktopWidget
from pathlib import Path
import gui.resources.res
from about_window import Ui_AboutWindow
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
help(Function.soft_threshold)

class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        self.logger = ''
        self.MainWindow = MainWindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(500, 800)  # 349, 617
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout_2.addWidget(self.lineEdit)
        self.loadDataBtn = QtWidgets.QPushButton(self.groupBox)
        self.loadDataBtn.setObjectName("loadDataBtn")
        self.horizontalLayout_2.addWidget(self.loadDataBtn)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_9.addWidget(self.label_7)
        self.comboBox = QtWidgets.QComboBox(self.groupBox)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        # self.comboBox.addItem("")  # ADMM not working
        self.horizontalLayout_9.addWidget(self.comboBox)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem)
        self.verticalLayout_3.addLayout(self.horizontalLayout_9)
        self.verticalLayout_5.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.horizontalLayout_3.addWidget(self.lineEdit_3)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_4 = QtWidgets.QLabel(self.groupBox_2)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.horizontalLayout_4.addWidget(self.lineEdit_4)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_5 = QtWidgets.QLabel(self.groupBox_2)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_5.addWidget(self.label_5)
        self.lineEdit_5 = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.horizontalLayout_5.addWidget(self.lineEdit_5)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_6 = QtWidgets.QLabel(self.groupBox_2)
        self.label_6.setEnabled(True)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_6.addWidget(self.label_6)
        self.lineEdit_6 = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.horizontalLayout_6.addWidget(self.lineEdit_6)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        self.verticalLayout_5.addWidget(self.groupBox_2)
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_3 = QtWidgets.QLabel(self.groupBox_3)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_7.addWidget(self.label_3)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.horizontalLayout_7.addWidget(self.lineEdit_2)
        self.verticalLayout_2.addLayout(self.horizontalLayout_7)
        self.progressBar = QtWidgets.QProgressBar(self.groupBox_3)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout_2.addWidget(self.progressBar)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem1)
        self.label_10 = QtWidgets.QLabel(self.groupBox_3)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_10.addWidget(self.label_10)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem2)
        self.verticalLayout_2.addLayout(self.horizontalLayout_10)
        self.textEdit = QtWidgets.QTextEdit(self.groupBox_3)
        self.textEdit.setObjectName("textEdit")
        self.verticalLayout_2.addWidget(self.textEdit)
        self.verticalLayout_4.addLayout(self.verticalLayout_2)
        self.verticalLayout_5.addWidget(self.groupBox_3)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.startButton = QtWidgets.QPushButton(self.centralwidget)
        self.startButton.setObjectName("startButton")
        self.horizontalLayout.addWidget(self.startButton)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem4)
        self.verticalLayout_5.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setEnabled(True)
        self.toolBar.setMovable(False)
        self.toolBar.setFloatable(False)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionAcerca_de = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/info_icon/info_icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionAcerca_de.setIcon(icon)
        self.actionAcerca_de.setObjectName("actionAcerca_de")
        self.toolBar.addAction(self.actionAcerca_de)
        self.actionAcerca_de.triggered.connect(self.show_about_window)
        self.loadDataBtn.clicked.connect(self.browseFiles)
        self.onlydouble = QtGui.QDoubleValidator(decimals=10)

        self.onlyInt = QtGui.QIntValidator()
        self.lineEdit_6.setValidator(self.onlyInt)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.progressBar.setValue(0)

        self.startButton.clicked.connect(self.run)
        self.textEdit.setReadOnly(True)

        self.comboBox.currentTextChanged.connect(self.on_algorithm_changed)

        self.update_parameters_info(self.comboBox.currentText())

    def browseFiles(self):
        self.fname = QFileDialog.getOpenFileName(self.MainWindow, filter='*.npy')
        self.lineEdit.setText(self.fname[0].split("/")[-1].strip("/.npy"))

    def show_about_window(self):
        self.about_window = QtWidgets.QWidget()
        self.ui_about_window = Ui_AboutWindow()
        self.ui_about_window.setupUi(self.about_window)
        self.about_window.show()

    def on_algorithm_changed(self, value):
        self.update_parameters_info(value)
        self.textEdit.setText('')

    def update_parameters_info(self, algorithm):
        self.label_2.setVisible(True)
        self.lineEdit_3.setVisible(True)
        self.label_4.setVisible(True)
        self.lineEdit_4.setVisible(True)
        self.label_5.setVisible(True)
        self.label_6.setVisible(True)
        self.lineEdit_5.setVisible(True)
        self.lineEdit_6.setVisible(True)
        if algorithm == "FISTA":
            self.label_2.setText("Tau")
            self.lineEdit_3.setText(str(0.1))
            self.label_4.setText("Mu")
            self.lineEdit_4.setText(str(0.3))
            self.lineEdit_3.setValidator(self.onlydouble)
            self.lineEdit_4.setValidator(self.onlydouble)

            self.label_5.setVisible(False)
            self.label_6.setVisible(False)
            self.lineEdit_5.setVisible(False)
            self.lineEdit_6.setVisible(False)
            # self.lineEdit_5.setValidator(self.onlydouble)

        elif algorithm == "GAP":
            self.label_2.setText("Tau")
            self.lineEdit_3.setText(str(1.0))

            self.lineEdit_3.setValidator(self.onlydouble)

            self.label_4.setVisible(False)
            self.lineEdit_4.setVisible(False)
            self.label_5.setVisible(False)
            self.label_6.setVisible(False)
            self.lineEdit_5.setVisible(False)
            self.lineEdit_6.setVisible(False)

        elif algorithm == "TWIST":

            self.label_2.setText("Tau")
            self.lineEdit_3.setText(str(0.9))
            self.label_4.setText("Alpha")
            self.lineEdit_4.setText(str(1.2))
            self.label_5.setText("Beta")
            self.lineEdit_5.setText(str(1.998))

            self.lineEdit_3.setValidator(self.onlydouble)
            self.lineEdit_4.setValidator(self.onlydouble)
            self.lineEdit_5.setValidator(self.onlydouble)

            self.label_6.setVisible(False)
            self.lineEdit_6.setVisible(False)

        elif algorithm == "ADMM":
            self.label_2.setText("Rho")
            self.lineEdit_3.setText(str(0.5))
            self.label_4.setText("Gamma")
            self.lineEdit_4.setText(str(1.0))
            self.label_5.setText("Lambda")
            self.lineEdit_5.setText(str(0.0078))

            self.lineEdit_3.setValidator(self.onlydouble)
            self.lineEdit_4.setValidator(self.onlydouble)
            self.lineEdit_5.setValidator(self.onlydouble)

            self.label_6.setVisible(False)
            self.lineEdit_6.setVisible(False)
        else:
            raise Exception("Invalid Algorithm Name")

        self.label_2.setMinimumWidth(80)
        self.label_4.setMinimumWidth(80)
        self.label_5.setMinimumWidth(80)
        self.label_6.setMinimumWidth(80)

    def check_params_algorithm(self, algorithm):

        if algorithm == "FISTA":
            if self.lineEdit_3.text().strip() == '' or self.lineEdit_4.text().strip() == '':
                showWarning("Para iniciar, debe completar todos los parámetros.")
        elif algorithm == "GAP":
            if self.lineEdit_3.text().strip() == '':
                showWarning("Para iniciar, debe completar todos los parámetros.")
        elif algorithm == "TWIST":
            if (self.lineEdit_3.text().strip() == '' or self.lineEdit_4.text().strip() == '' or
                    self.lineEdit_5.text().strip() == ''):
                showWarning("Para iniciar, debe completar todos los parámetros.")
        elif algorithm == "ADMM":
            if (self.lineEdit_3.text().strip() == '' or self.lineEdit_4.text().strip() == '' or
                    self.lineEdit_5.text().strip() == ''):
                showWarning("Para iniciar, debe completar todos los parámetros.")
        else:
            raise Exception("Invalid Algorithm Name")


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "Entradas"))
        self.label.setText(_translate("MainWindow", "Dato Sísmico"))
        self.loadDataBtn.setText(_translate("MainWindow", "Cargar"))
        self.label_7.setText(_translate("MainWindow", "Algoritmo      "))
        self.comboBox.setItemText(0, _translate("MainWindow", "FISTA"))
        self.comboBox.setItemText(1, _translate("MainWindow", "GAP"))
        self.comboBox.setItemText(2, _translate("MainWindow", "TWIST"))
        # self.comboBox.setItemText(3, _translate("MainWindow", "ADMM")) # not working
        self.groupBox_2.setTitle(_translate("MainWindow", "Parámetros"))
        self.label_2.setText(_translate("MainWindow", "Muestreo en X"))
        self.label_4.setText(_translate("MainWindow", "Muestreo en Y"))
        self.label_5.setText(_translate("MainWindow", "Muestreo en Z"))
        self.label_6.setText(_translate("MainWindow", "No. de  Shots  "))
        self.groupBox_3.setTitle(_translate("MainWindow", "Salidas"))
        self.label_3.setText(_translate("MainWindow", "Guardar como"))
        self.label_10.setText(_translate("MainWindow", "Logger"))
        self.startButton.setText(_translate("MainWindow", "Iniciar"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionAcerca_de.setText(_translate("MainWindow", "Acerca de"))
        self.actionAcerca_de.setToolTip(
            _translate("MainWindow", "<html><head/><body><p>Acerca de este proyecto</p></body></html>"))

    def run(self):
        # checks
        if not hasattr(self, 'fname') or self.fname is None:
            showWarning("Para iniciar, debe cargar el dato sísmico dando click al boton 'Cargar'")
            return

        self.check_params_algorithm(self.comboBox.currentText())

        if self.lineEdit_2.text().strip() == '':
            showWarning("Por favor seleccione un nombre de archivo para guardar los resultados del algoritmo.")
            return

        try:
            # I took this values from ipynb notebook
            self.progressBar.setValue(0)

            self.maxiter = 1000
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

            self.logger = f'Running: {self.comboBox.currentText()} \n'
            self.logger += 'iterations \t ||x-xold|| \t PSNR \n'

            self.thread = QtCore.QThread()

            # Algorithms object

            Alg = Algorithms(x, H, 'DCT2D', 'IDCT2D')  # Assuming using DCT2D ad IDCT2D for all algorithms

            if self.comboBox.currentText() == "FISTA":
                parameters = {
                    "lmb": float(self.lineEdit_3.text()),  # Tau
                    "mu": float(self.lineEdit_4.text()),  # Mu
                    "max_itr": self.maxiter
                }
                func = Alg.FISTA
            elif self.comboBox.currentText() == "GAP":
                parameters = {
                    "lmb": float(self.lineEdit_3.text()),  # Tau
                    "max_itr": self.maxiter
                }
                func = Alg.GAP
            elif self.comboBox.currentText() == "TWIST":
                parameters = {
                    "lmb": float(self.lineEdit_3.text()),  # Tau
                    "alpha": float(self.lineEdit_4.text()),  # Alpha
                    "beta": float(self.lineEdit_5.text()),  # Beta
                    "max_itr": self.maxiter
                }
                func = Alg.TwIST
            elif self.comboBox.currentText() == "ADMM":
                parameters = {
                    "rho": float(self.lineEdit_3.text()),  # Rho
                    "gamma": float(self.lineEdit_4.text()),  # Gamma
                    "lamnda": float(self.lineEdit_5.text()),  # Lambda
                    "max_itr": self.maxiter
                }
                func = Alg.ADMM
            # elif self.comboBox.currentText() == ""

            self.worker = Worker(func, parameters)

            self.worker.moveToThread(self.thread)

            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.progress.connect(self.reportProgress)

            self.thread.start()

            self.worker.finished.connect(self.save_results)  # save results

            # Final resets
            self.startButton.setEnabled(False)
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
        self.startButton.setEnabled(True)
        self.progressBar.setValue(0)
        # self.logger = ''
        self.maxiter = 0
        self.worker = None
        self.thread = None
        ee.progress = None
        self.sampling_dict = None

    def save_results(self, res_dict):
        pwd = os.getcwd()
        save_path = pwd+"/Results"
        Path(save_path).mkdir(parents=True, exist_ok=True)
        filepath = save_path + "/" + self.lineEdit_2.text() + ".npz"
        np.savez(filepath, x_result=res_dict['result'], hist=res_dict['hist'], sampling=self.sampling_dict)
        print("Results saved [Ok]")


class Worker(QtCore.QObject):
    finished = QtCore.pyqtSignal(dict)
    progress = QtCore.pyqtSignal(int, str, str)

    def __init__(self, function, parameters):
        super().__init__()
        self.function = function
        self.parameters = parameters

    def run(self):

        ee.progress = self.progress

        @ee.on("algorithm_update")
        def handler(iter, err, psnr):
            ee.progress.emit(iter, err, psnr)


        # Alg.FISTA(tau, mu, self.maxiter)
        x_result, hist = self.function(**self.parameters)

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
