# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui/show_results.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import os
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from Algorithms.Function import PSNR
from skimage.metrics import structural_similarity as ssim

from gui.alerts import showCritical


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=500, height=400, dpi=500):
        self.figure = Figure(figsize=(width, height))
        super(MplCanvas, self).__init__(self.figure)

    def update_figure(self, data):
        try:
            self.figure.clear()

            x_result = data['x_result']
            sampling = {item[0]: item[1] for item in data['sampling']}

            x = sampling['x_ori']
            y_rand = sampling['y_rand']
            pattern_rand = sampling['pattern_rand']

            temp = np.asarray(range(0, pattern_rand.shape[0]))
            pattern_rand_b2 = np.asarray(pattern_rand, dtype=bool) == 0
            H_elim = temp[pattern_rand_b2]

            case = str(data['alg_name'])
            self.figure.suptitle(f'Resultos del algoritmo {case}')
            axs = self.figure.subplots(2, 2)

            axs[0, 0].imshow(x, cmap='seismic', aspect='auto')
            axs[0, 0].set_title('Referencia')

            ytemp = y_rand.copy()
            ytemp[:, H_elim] = 0
            axs[1, 0].imshow(ytemp, cmap='seismic', aspect='auto')
            axs[1, 0].set_title('Medidas')

            # axs[1, 0].sharex(axs[0, 0])
            metric = PSNR(x[:, H_elim], x_result[:, H_elim])
            metric_ssim = ssim(x[:, H_elim], x_result[:, H_elim], win_size=3)
            axs[0, 1].imshow(x_result, cmap='seismic', aspect='auto')
            axs[0, 1].set_title(f'Reconstruido \n PSNR: {metric:0.2f} dB, SSIM:{metric_ssim:0.2f}')

            index = 5
            axs[1, 1].plot(x[:, H_elim[index]], 'r', label='Referencia')
            axs[1, 1].plot(x_result[:, H_elim[index]], 'b', label='Recuperado')
            axs[1, 1].legend(loc='best')
            axs[1, 1].set_title('Traza ' + str("{:.0f}".format(H_elim[index])))

            self.draw()

        except BaseException as err:
            msg = f"Unexpected {err=}, {type(err)=}"
            showCritical("Ocurrió un error inesperado al procesar el dato sísmico. Por favor, intente nuevamente o "
                         "utilice un dato diferente.", details=msg)
            return


class UIResultsWindow(QtWidgets.QMainWindow):
    def setupUi(self, FormWidget):
        self.ResultsWindow = FormWidget
        FormWidget.setObjectName("FormWidget")
        FormWidget.resize(621, 656)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(FormWidget)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.reportGenGroupBox = QtWidgets.QGroupBox(FormWidget)
        self.reportGenGroupBox.setObjectName("reportGenGroupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.reportGenGroupBox)
        self.verticalLayout.setObjectName("verticalLayout")
        self.fileResultsHorizontalLayout = QtWidgets.QHBoxLayout()
        self.fileResultsHorizontalLayout.setObjectName("fileResultsHorizontalLayout")
        self.fileResultsLabel = QtWidgets.QLabel(self.reportGenGroupBox)
        self.fileResultsLabel.setObjectName("fileResultsLabel")
        self.fileResultsHorizontalLayout.addWidget(self.fileResultsLabel)
        self.fileResultsLineEdit = QtWidgets.QLineEdit(self.reportGenGroupBox)
        self.fileResultsLineEdit.setObjectName("fileResultsLineEdit")
        self.fileResultsHorizontalLayout.addWidget(self.fileResultsLineEdit)
        self.loadDataPushButton = QtWidgets.QPushButton(self.reportGenGroupBox)
        self.loadDataPushButton.setObjectName("loadDataPushButton")
        self.fileResultsHorizontalLayout.addWidget(self.loadDataPushButton)
        self.verticalLayout.addLayout(self.fileResultsHorizontalLayout)
        # self.loadDataprogressBar = QtWidgets.QProgressBar(self.reportGenGroupBox)
        # self.loadDataprogressBar.setProperty("value", 0)
        # self.loadDataprogressBar.setObjectName("loadDataprogressBar")
        # self.verticalLayout.addWidget(self.loadDataprogressBar)
        self.verticalLayout_3.addWidget(self.reportGenGroupBox)
        self.showResultsGroupBox = QtWidgets.QGroupBox(FormWidget)
        self.showResultsGroupBox.setObjectName("showResultsGroupBox")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.showResultsGroupBox)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.showResultsGroupBox.setMinimumSize(700, 700)
        self.verticalLayout_3.addWidget(self.showResultsGroupBox)

        self.canvas = MplCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.verticalLayout_2.addWidget(self.toolbar)
        self.verticalLayout_2.addWidget(self.canvas)

        self.dReportHBoxLayout = QtWidgets.QHBoxLayout()
        self.dReportHBoxLayout.setObjectName("dReportHBoxLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.dReportHBoxLayout.addItem(spacerItem)
        self.downloadReportPushButton = QtWidgets.QPushButton(FormWidget)
        self.downloadReportPushButton.setObjectName("downloadReportPushButton")
        self.dReportHBoxLayout.addWidget(self.downloadReportPushButton)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.dReportHBoxLayout.addItem(spacerItem1)
        self.verticalLayout_3.addLayout(self.dReportHBoxLayout)

        self.current_directory = ''

        # events
        self.loadDataPushButton.clicked.connect(self.browse_files)
        self.downloadReportPushButton.clicked.connect(self.download_report)

        self.retranslateUi(FormWidget)
        QtCore.QMetaObject.connectSlotsByName(FormWidget)

    def browse_files(self):
        kwargs = {}
        if 'SNAP' in os.environ:
            kwargs['options'] = QFileDialog.DontUseNativeDialog
        self.fname = QtWidgets.QFileDialog.getOpenFileName(self.ResultsWindow, 'Open File', self.current_directory,
                                                           filter='numpy file (*.npz)', **kwargs)
        self.fileResultsLineEdit.setText(self.fname[0].split("/")[-1].strip("/.npz"))

        if self.fname[0] == '':
            return

        self.current_directory = self.fname[0]
        data = np.load(self.fname[0], allow_pickle=True)
        self.canvas.update_figure(data)

    def download_report(self):
        # selecting file path
        kwargs = {}
        if 'SNAP' in os.environ:
            kwargs['options'] = QFileDialog.DontUseNativeDialog
        fname = QFileDialog.getSaveFileName(self.ResultsWindow, "Save Image", self.current_directory,
                                            filter="PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ", **kwargs)

        # if file path is blank return back
        if fname == '':
            return
        self.current_directory = fname[0]

        # saving canvas at desired path
        self.canvas.figure.savefig(fname[0])

    def retranslateUi(self, FormWidget):
        _translate = QtCore.QCoreApplication.translate
        FormWidget.setWindowTitle(_translate("FormWidget", "Visualización de Resultados"))
        self.reportGenGroupBox.setTitle(_translate("FormWidget", "Generación de reportes"))
        self.fileResultsLabel.setText(_translate("FormWidget", "Archivo de Resultados"))
        self.loadDataPushButton.setText(_translate("FormWidget", "Cargar"))
        self.showResultsGroupBox.setTitle(_translate("FormWidget", "Visualización de resultados"))
        self.downloadReportPushButton.setText(_translate("FormWidget", "Descargar Reporte"))

# H0 = A['sampling'][()]['H0']