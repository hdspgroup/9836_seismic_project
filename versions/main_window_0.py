import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.ticker import MaxNLocator, MultipleLocator, NullLocator, FixedLocator, AutoLocator

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QDesktopWidget, QTableWidgetItem

from about_window import UIAboutWindow
from equation_window import UIEquationWindow
from results_window import UIResultsWindow
from gui.alerts import *

from Algorithms.Function import *

'''
Parameters from Ipynb notebook
'''


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.iteracion = []
        self.error = []
        self.psnr = []

        self.figure = plt.figure()
        self.axes_1 = self.figure.add_subplot(111)
        self.axes_2 = self.axes_1.twinx()

        super(MplCanvas, self).__init__(self.figure)

    def update_values(self, iteracion, error, psnr):
        self.iteracion.append(iteracion)
        self.error.append(error)
        self.psnr.append(psnr)

    def reset_values(self):
        self.iteracion = []
        self.error = []
        self.psnr = []

    def update_figure(self):

        try:
            self.axes_1.cla()
            self.axes_2.cla()

            self.figure.suptitle(f'Resultados del experimento')

            color = 'tab:red'
            self.axes_1.set_xlabel('iteraciones')
            self.axes_1.set_ylabel('error', color=color)
            self.axes_1.plot(self.error, color=color)
            self.axes_1.tick_params(axis='y', labelcolor=color, length=10)
            self.axes_1.yaxis.set_major_locator(MaxNLocator(8))
            self.axes_1.invert_yaxis()
            self.axes_1.set_xlim(self.axes_1.get_xlim())

            color = 'tab:blue'
            self.axes_2.set_ylabel('psnr', color=color)
            self.axes_2.plot(self.psnr, color=color)
            self.axes_2.tick_params(axis='y', labelcolor=color, length=10)
            self.axes_2.yaxis.set_major_locator(MaxNLocator(8))
            self.axes_2.set_xlim(self.axes_2.get_xlim())

            self.draw()

        except BaseException as err:
            msg = f"Unexpected {err=}, {type(err)=}"
            showCritical("Ocurrió un error inesperado al procesar el dato sísmico. Por favor, intente nuevamente o "
                         "utilice un dato diferente.", details=msg)
            return


class UiMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(UiMainWindow, self).__init__()
        self.setupUi()

    def closeEvent(self, event):
        message_box = QMessageBox(self)
        message_box.pos()
        message_box.setIcon(QMessageBox.Question)
        message_box.setWindowTitle('Cerrar aplicación')
        message_box.setText('¿Estás segur@ que quieres cerrar la aplicación?')
        message_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        yesButton = message_box.button(QMessageBox.Yes)
        yesButton.setText('Si')
        buttonN = message_box.button(QMessageBox.No)
        buttonN.setText('No')
        message_box.exec_()

        if message_box.clickedButton() == yesButton:
            event.accept()
            print('Window closed')
            sys.exit(0)
        else:
            event.ignore()

    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(1047, 615)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.inputGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.inputGroupBox.setObjectName("inputGroupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.inputGroupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.algorithmHLayout = QtWidgets.QHBoxLayout()
        self.algorithmHLayout.setObjectName("algorithmHLayout")
        self.algorithmComboBox = QtWidgets.QComboBox(self.inputGroupBox)
        self.algorithmComboBox.setObjectName("algorithmComboBox")
        self.algorithmComboBox.addItem("")
        self.algorithmComboBox.addItem("")
        self.algorithmComboBox.addItem("")
        self.algorithmComboBox.addItem("")
        self.algorithmHLayout.addWidget(self.algorithmComboBox)
        self.algorithmPushButton = QtWidgets.QPushButton(self.inputGroupBox)
        self.algorithmPushButton.setEnabled(True)
        self.algorithmPushButton.setAutoFillBackground(False)
        self.algorithmPushButton.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("assets/icons/view.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.algorithmPushButton.setIcon(icon)
        self.algorithmPushButton.setObjectName("algorithmPushButton")
        self.algorithmHLayout.addWidget(self.algorithmPushButton)
        spacerItem = QtWidgets.QSpacerItem(13, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.algorithmHLayout.addItem(spacerItem)
        self.gridLayout.addLayout(self.algorithmHLayout, 1, 1, 1, 1)
        self.inputLabel = QtWidgets.QLabel(self.inputGroupBox)
        self.inputLabel.setObjectName("inputLabel")
        self.gridLayout.addWidget(self.inputLabel, 0, 0, 1, 1)
        self.loadPushButton = QtWidgets.QPushButton(self.inputGroupBox)
        self.loadPushButton.setObjectName("loadPushButton")
        self.gridLayout.addWidget(self.loadPushButton, 0, 2, 1, 1)
        self.inputLineEdit = QtWidgets.QLineEdit(self.inputGroupBox)
        self.inputLineEdit.setObjectName("inputLineEdit")
        self.gridLayout.addWidget(self.inputLineEdit, 0, 1, 1, 1)
        self.algorithmLabelHLayout = QtWidgets.QHBoxLayout()
        self.algorithmLabelHLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.algorithmLabelHLayout.setObjectName("algorithmLabelHLayout")
        self.algorithmLabel = QtWidgets.QLabel(self.inputGroupBox)
        self.algorithmLabel.setObjectName("algorithmLabel")
        self.algorithmLabelHLayout.addWidget(self.algorithmLabel)
        spacerItem1 = QtWidgets.QSpacerItem(13, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.algorithmLabelHLayout.addItem(spacerItem1)
        self.gridLayout.addLayout(self.algorithmLabelHLayout, 1, 0, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(78, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem2, 1, 2, 1, 1)
        self.gridLayout.setColumnStretch(1, 10)
        self.verticalLayout.addWidget(self.inputGroupBox)
        self.paramGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.paramGroupBox.setObjectName("paramGroupBox")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.paramGroupBox)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.paramHLayout = QtWidgets.QHBoxLayout()
        self.paramHLayout.setObjectName("paramHLayout")
        self.maxiterLabel = QtWidgets.QLabel(self.paramGroupBox)
        self.maxiterLabel.setObjectName("maxiterLabel")
        self.paramHLayout.addWidget(self.maxiterLabel)
        self.maxiterSpinBox = QtWidgets.QSpinBox(self.paramGroupBox)
        self.maxiterSpinBox.setMinimum(1)
        self.maxiterSpinBox.setMaximum(9999)
        self.maxiterSpinBox.setProperty("value", 100)
        self.maxiterSpinBox.setObjectName("maxiterSpinBox")
        self.paramHLayout.addWidget(self.maxiterSpinBox)
        self.param1Label = QtWidgets.QLabel(self.paramGroupBox)
        self.param1Label.setObjectName("param1Label")
        self.paramHLayout.addWidget(self.param1Label)
        self.param1LineEdit = QtWidgets.QLineEdit(self.paramGroupBox)
        self.param1LineEdit.setObjectName("param1LineEdit")
        self.paramHLayout.addWidget(self.param1LineEdit)
        self.param2Label = QtWidgets.QLabel(self.paramGroupBox)
        self.param2Label.setObjectName("param2Label")
        self.paramHLayout.addWidget(self.param2Label)
        self.param2LineEdit = QtWidgets.QLineEdit(self.paramGroupBox)
        self.param2LineEdit.setObjectName("param2LineEdit")
        self.paramHLayout.addWidget(self.param2LineEdit)
        self.param3Label = QtWidgets.QLabel(self.paramGroupBox)
        self.param3Label.setObjectName("param3Label")
        self.paramHLayout.addWidget(self.param3Label)
        self.param3LineEdit = QtWidgets.QLineEdit(self.paramGroupBox)
        self.param3LineEdit.setObjectName("param3LineEdit")
        self.paramHLayout.addWidget(self.param3LineEdit)
        self.horizontalLayout_4.addLayout(self.paramHLayout)
        self.verticalLayout.addWidget(self.paramGroupBox)
        self.samplingGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.samplingGroupBox.setObjectName("samplingGroupBox")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.samplingGroupBox)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.samplingHLayout = QtWidgets.QHBoxLayout()
        self.samplingHLayout.setObjectName("samplingHLayout")
        self.samplingTypeLabel = QtWidgets.QLabel(self.samplingGroupBox)
        self.samplingTypeLabel.setObjectName("samplingTypeLabel")
        self.samplingHLayout.addWidget(self.samplingTypeLabel)
        self.samplingTypeComboBox = QtWidgets.QComboBox(self.samplingGroupBox)
        self.samplingTypeComboBox.setObjectName("samplingTypeComboBox")
        self.samplingTypeComboBox.addItem("")
        self.samplingTypeComboBox.addItem("")
        self.samplingTypeComboBox.addItem("")
        self.samplingTypeComboBox.addItem("")
        self.samplingHLayout.addWidget(self.samplingTypeComboBox)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.samplingHLayout.addItem(spacerItem3)
        self.compressLabel = QtWidgets.QLabel(self.samplingGroupBox)
        self.compressLabel.setObjectName("compressLabel")
        self.samplingHLayout.addWidget(self.compressLabel)
        self.compressSpinBox = QtWidgets.QSpinBox(self.samplingGroupBox)
        self.compressSpinBox.setPrefix("")
        self.compressSpinBox.setMinimum(1)
        self.compressSpinBox.setMaximum(99)
        self.compressSpinBox.setProperty("value", 50)
        self.compressSpinBox.setObjectName("compressSpinBox")
        self.samplingHLayout.addWidget(self.compressSpinBox)
        self.verticalLayout_2.addLayout(self.samplingHLayout)
        self.samplingHLine = QtWidgets.QFrame(self.samplingGroupBox)
        self.samplingHLine.setFrameShape(QtWidgets.QFrame.HLine)
        self.samplingHLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.samplingHLine.setObjectName("samplingHLine")
        self.verticalLayout_2.addWidget(self.samplingHLine)
        self.elementHLayout = QtWidgets.QHBoxLayout()
        self.elementHLayout.setObjectName("elementHLayout")
        self.elementLabel = QtWidgets.QLabel(self.samplingGroupBox)
        self.elementLabel.setObjectName("elementLabel")
        self.elementHLayout.addWidget(self.elementLabel)
        self.elementLineEdit = QtWidgets.QLineEdit(self.samplingGroupBox)
        self.elementLineEdit.setObjectName("elementLineEdit")
        self.elementHLayout.addWidget(self.elementLineEdit)
        self.verticalLayout_2.addLayout(self.elementHLayout)
        self.jitterHLayout = QtWidgets.QHBoxLayout()
        self.jitterHLayout.setObjectName("jitterHLayout")
        self.jitterHspacer1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding,
                                                    QtWidgets.QSizePolicy.Minimum)
        self.jitterHLayout.addItem(self.jitterHspacer1)
        self.jitterBlockLabel = QtWidgets.QLabel(self.samplingGroupBox)
        self.jitterBlockLabel.setObjectName("jitterBlockLabel")
        self.jitterHLayout.addWidget(self.jitterBlockLabel)
        self.jitterBlockSpinBox = QtWidgets.QSpinBox(self.samplingGroupBox)
        self.jitterBlockSpinBox.setSuffix("")
        self.jitterBlockSpinBox.setPrefix("")
        self.jitterBlockSpinBox.setMinimum(1)
        self.jitterBlockSpinBox.setMaximum(999)
        self.jitterBlockSpinBox.setObjectName("jitterBlockSpinBox")
        self.jitterHLayout.addWidget(self.jitterBlockSpinBox)
        self.jitterHspacer2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding,
                                                    QtWidgets.QSizePolicy.Minimum)
        self.jitterHLayout.addItem(self.jitterHspacer2)
        self.verticalLayout_2.addLayout(self.jitterHLayout)
        self.verticalLayout.addWidget(self.samplingGroupBox)
        self.runGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.runGroupBox.setObjectName("runGroupBox")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.runGroupBox)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.saveAsHLayout = QtWidgets.QHBoxLayout()
        self.saveAsHLayout.setObjectName("saveAsHLayout")
        self.saveAsLabel = QtWidgets.QLabel(self.runGroupBox)
        self.saveAsLabel.setObjectName("saveAsLabel")
        self.saveAsHLayout.addWidget(self.saveAsLabel)
        self.saveAsLineEdit = QtWidgets.QLineEdit(self.runGroupBox)
        self.saveAsLineEdit.setObjectName("saveAsLineEdit")
        self.saveAsHLayout.addWidget(self.saveAsLineEdit)
        self.saveAsPushButton = QtWidgets.QPushButton(self.runGroupBox)
        self.saveAsPushButton.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("assets/icons/save.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.saveAsPushButton.setIcon(icon1)
        self.saveAsPushButton.setObjectName("saveAsPushButton")
        self.saveAsHLayout.addWidget(self.saveAsPushButton)
        self.verticalLayout_4.addLayout(self.saveAsHLayout)
        self.outputTableWidget = QtWidgets.QTableWidget(self.runGroupBox)
        self.outputTableWidget.setAutoFillBackground(False)
        self.outputTableWidget.setShowGrid(True)
        self.outputTableWidget.setObjectName("outputTableWidget")
        self.outputTableWidget.setColumnCount(2)
        self.outputTableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.outputTableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.outputTableWidget.setHorizontalHeaderItem(1, item)
        self.outputTableWidget.horizontalHeader().setSortIndicatorShown(False)
        self.outputTableWidget.horizontalHeader().setStretchLastSection(True)
        self.verticalLayout_4.addWidget(self.outputTableWidget)
        self.startHLayout = QtWidgets.QHBoxLayout()
        self.startHLayout.setObjectName("startHLayout")
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.startHLayout.addItem(spacerItem8)
        self.startPushButton = QtWidgets.QPushButton(self.runGroupBox)
        self.startPushButton.setObjectName("startPushButton")
        self.startHLayout.addWidget(self.startPushButton)
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.startHLayout.addItem(spacerItem9)
        self.verticalLayout_4.addLayout(self.startHLayout)
        self.verticalLayout.addWidget(self.runGroupBox)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.resultGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.resultGroupBox.setObjectName("resultGroupBox")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.resultGroupBox)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_2.addWidget(self.resultGroupBox)
        self.horizontalLayout_2.setStretch(0, 30)
        self.horizontalLayout_2.setStretch(1, 50)
        self.setCentralWidget(self.centralwidget)
        self.toolBar = QtWidgets.QToolBar(self)
        self.toolBar.setObjectName("toolBar")
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.aboutOfAction = QtWidgets.QAction(self)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("assets/icons/info.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.aboutOfAction.setIcon(icon2)
        self.aboutOfAction.setObjectName("aboutOfAction")
        self.reportAction = QtWidgets.QAction(self)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("assets/icons/report.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.reportAction.setIcon(icon3)
        self.reportAction.setObjectName("reportAction")
        self.toolBar.addAction(self.aboutOfAction)
        self.toolBar.addAction(self.reportAction)

        self.canvas = MplCanvas()
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.verticalLayout_5.addWidget(self.toolbar)
        self.verticalLayout_5.addWidget(self.canvas)

        self.resultProgressBar = QtWidgets.QProgressBar(self.resultGroupBox)
        self.resultProgressBar.setProperty("value", 0)
        self.resultProgressBar.setObjectName("resultProgressBar")
        self.verticalLayout_5.addWidget(self.resultProgressBar)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

        # actions

        self.current_directory = ''
        self.algorithmPushButton.clicked.connect(self.show_equation_window)

        self.aboutOfAction.triggered.connect(self.show_about_window)
        self.reportAction.triggered.connect(self.show_results_window)
        self.loadPushButton.clicked.connect(self.browse_files)
        self.saveAsPushButton.clicked.connect(self.save_files)
        self.onlydouble = QtGui.QDoubleValidator(decimals=10)
        self.onlyInt = QtGui.QIntValidator()
        self.param3LineEdit.setValidator(self.onlyInt)
        self.resultProgressBar.setValue(0)

        self.startPushButton.clicked.connect(self.run)
        self.inputLineEdit.textChanged.connect(self.input_text_changed)
        self.saveAsLineEdit.textChanged.connect(self.save_as_changed)

        self.algorithmComboBox.currentTextChanged.connect(self.on_algorithm_changed)
        self.samplingTypeComboBox.currentTextChanged.connect(self.on_sampling_type_changed)

        self.currentAlgorithmName = ''
        self.update_parameters_info(self.algorithmComboBox.currentText().lower())
        self.update_sampling_type_info(self.samplingTypeComboBox.currentText().lower())

    def input_text_changed(self, event):
        self.fname = (event,)

    def save_as_changed(self, event):
        self.sname = (event,)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "9836 Proyecto de Sísmica"))
        self.inputGroupBox.setTitle(_translate("MainWindow", "Entradas"))
        self.algorithmComboBox.setItemText(0, _translate("MainWindow", "FISTA"))
        self.algorithmComboBox.setItemText(1, _translate("MainWindow", "GAP"))
        self.algorithmComboBox.setItemText(2, _translate("MainWindow", "TwIST"))
        self.algorithmComboBox.setItemText(3, _translate("MainWindow", "ADMM"))
        self.algorithmPushButton.setToolTip(_translate("MainWindow", "Ver ecuación"))
        self.inputLabel.setText(_translate("MainWindow", "Dato sísmico"))
        self.loadPushButton.setText(_translate("MainWindow", "Cargar"))
        self.algorithmLabel.setText(_translate("MainWindow", "Algoritmo"))
        self.paramGroupBox.setTitle(_translate("MainWindow", "Parámetros"))
        self.maxiterLabel.setText(_translate("MainWindow", "Max iter"))
        self.param1Label.setText(_translate("MainWindow", "p1"))
        self.param2Label.setText(_translate("MainWindow", "p2"))
        self.param3Label.setText(_translate("MainWindow", "p3"))
        self.samplingGroupBox.setTitle(_translate("MainWindow", "Submuestreo"))
        self.samplingTypeLabel.setText(_translate("MainWindow", "Tipo"))
        self.samplingTypeComboBox.setItemText(0, _translate("MainWindow", "Aleatorio"))
        self.samplingTypeComboBox.setItemText(1, _translate("MainWindow", "Uniforme"))
        self.samplingTypeComboBox.setItemText(2, _translate("MainWindow", "Jitter"))
        self.samplingTypeComboBox.setItemText(3, _translate("MainWindow", "Lista"))
        self.compressLabel.setText(_translate("MainWindow", "Nivel de compresión"))
        self.compressSpinBox.setSuffix(_translate("MainWindow", "%"))
        self.elementLabel.setText(_translate("MainWindow", "Elementos"))
        self.elementLineEdit.setToolTip(_translate("MainWindow",
                                                   "Ingrese la lista en formato: a,b,c,...\n"
                                                   "Las columnas empiezan con el índice 0\n"
                                                   "Separados por coma sin espacios\n"
                                                   "Solo números enteros\n"
                                                   "Mínimo 7 números"))
        self.jitterBlockLabel.setText(_translate("MainWindow", "Número de bloques"))
        self.runGroupBox.setTitle(_translate("MainWindow", "Correr experimento"))
        self.saveAsLabel.setText(_translate("MainWindow", "Guardar como"))
        item = self.outputTableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Error"))
        item = self.outputTableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "PSNR"))
        self.startPushButton.setText(_translate("MainWindow", "Iniciar"))
        self.resultGroupBox.setTitle(_translate("MainWindow", "Resultados"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.aboutOfAction.setText(_translate("MainWindow", "about"))
        self.aboutOfAction.setToolTip(
            _translate("MainWindow", "<html><head/><body><p>Acerca de este proyecto</p></body></html>"))
        self.reportAction.setText(_translate("MainWindow", "report"))

    def browse_files(self):
        kwargs = {}
        if 'SNAP' in os.environ:
            kwargs['options'] = QFileDialog.DontUseNativeDialog

        self.fname = QFileDialog.getOpenFileName(self, 'Open File', self.current_directory, filter='numpy file (*.npy)',
                                                 **kwargs)
        self.inputLineEdit.setText(self.fname[0])  # .split("/")[-1].strip("/.npy"))

        if self.fname[0] == '':
            return

        self.current_directory = self.fname[0]

    def save_files(self):
        kwargs = {}
        if 'SNAP' in os.environ:
            kwargs['options'] = QFileDialog.DontUseNativeDialog

        self.sname = QFileDialog.getSaveFileName(self, 'Save File', self.current_directory, filter='numpy file (*.npz)',
                                                 **kwargs)
        if self.sname[0] == '':
            self.saveAsLineEdit.setText(self.sname[0])
            return

        self.sname = list(self.sname)
        self.sname[0] = f"{self.sname[0]}.npz"
        self.sname = tuple(self.sname)

        self.saveAsLineEdit.setText(self.sname[0])
        self.current_directory = self.sname[0]

    def show_about_window(self):
        self.about_window = QtWidgets.QWidget()
        self.ui_about_window = UIAboutWindow()
        self.ui_about_window.setupUi(self.about_window)
        self.about_window.show()

    def show_results_window(self):
        self.show_results_window = QtWidgets.QWidget()
        self.ui_results_window = UIResultsWindow()
        self.ui_results_window.setupUi(self.show_results_window)
        self.show_results_window.show()

    def show_equation_window(self):
        self.equation_window = QtWidgets.QWidget()
        self.ui_equation_window = UIEquationWindow()
        self.ui_equation_window.setupUi(self.equation_window, self.algorithmComboBox.currentText())
        self.equation_window.show()

    # def activate_seed(self, event):
    #     self.seedSpinBox.setEnabled(event)

    def on_algorithm_changed(self, value):
        self.update_parameters_info(value.lower())
        # self.textEdit.setText('')

    def on_sampling_type_changed(self, value):
        self.update_sampling_type_info(value.lower())

    def update_parameters_info(self, algorithm):
        self.param1Label.setVisible(True)
        self.param1LineEdit.setVisible(True)
        self.param2Label.setVisible(True)
        self.param2LineEdit.setVisible(True)
        self.param3Label.setVisible(True)
        self.param3LineEdit.setVisible(True)

        if algorithm == "fista":
            self.param1Label.setText("Tau")
            self.param1LineEdit.setText(str(0.1))
            self.param2Label.setText("Mu")
            self.param2LineEdit.setText(str(0.3))
            self.param1LineEdit.setValidator(self.onlyInt)
            self.param1LineEdit.setValidator(self.onlydouble)
            self.param2LineEdit.setValidator(self.onlydouble)
            self.param3Label.setVisible(False)
            self.param3LineEdit.setVisible(False)

        elif algorithm == "gap":
            self.param1Label.setText("Tau")
            self.param1LineEdit.setText(str(1.0))
            self.param1LineEdit.setValidator(self.onlydouble)
            self.param2Label.setVisible(False)
            self.param2LineEdit.setVisible(False)
            self.param3Label.setVisible(False)
            self.param3LineEdit.setVisible(False)

        elif algorithm == "twist":
            self.param1Label.setText("Tau")
            self.param1LineEdit.setText(str(0.9))
            self.param2Label.setText("Alpha")
            self.param2LineEdit.setText(str(1.2))
            self.param3Label.setText("Beta")
            self.param3LineEdit.setText(str(1.998))

            self.param1LineEdit.setValidator(self.onlydouble)
            self.param2LineEdit.setValidator(self.onlydouble)
            self.param3LineEdit.setValidator(self.onlydouble)

        elif algorithm == "admm":
            self.param1Label.setText("Rho")
            self.param1LineEdit.setText(str(0.5))
            self.param2Label.setText("Gamma")
            self.param2LineEdit.setText(str(1.0))
            self.param3Label.setText("Lambda")
            self.param3LineEdit.setText(str(0.0078))

            self.param1LineEdit.setValidator(self.onlydouble)
            self.param2LineEdit.setValidator(self.onlydouble)
            self.param3LineEdit.setValidator(self.onlydouble)
        else:
            raise Exception("Invalid Algorithm Name")

    def update_sampling_type_info(self, sampling):
        self.samplingHLine.setVisible(False)

        self.elementLabel.setVisible(False)
        self.elementLineEdit.setVisible(False)

        self.jitterBlockLabel.setVisible(False)
        self.jitterBlockSpinBox.setVisible(False)
        # self.jitterTypeLabel.setVisible(False)
        # self.jitterComboBox.setVisible(False)

        self.jitterHspacer1.changeSize(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.jitterHspacer2.changeSize(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        self.jitterHLayout.invalidate()

        if sampling == 'aleatorio':
            self.samplingHLine.setVisible(False)
            self.compressLabel.setVisible(True)
            self.compressSpinBox.setVisible(True)

            self.compressSpinBox.setMinimum(7)
            self.compressSpinBox.setMaximum(99)

        elif sampling == 'uniforme':
            self.samplingHLine.setVisible(False)
            self.compressLabel.setVisible(True)
            self.compressSpinBox.setVisible(True)

            self.compressSpinBox.setMinimum(7)
            self.compressSpinBox.setMaximum(50)

        elif sampling == 'jitter':
            self.samplingHLine.setVisible(True)

            self.jitterBlockLabel.setVisible(True)
            self.jitterBlockSpinBox.setVisible(True)

            self.jitterHspacer1.changeSize(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
            self.jitterHspacer2.changeSize(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

            self.compressLabel.setVisible(True)
            self.compressSpinBox.setVisible(True)

            self.compressSpinBox.setMinimum(7)
            self.compressSpinBox.setMaximum(99)

        else:  # lista
            self.samplingHLine.setVisible(True)
            self.elementLabel.setVisible(True)
            self.elementLineEdit.setVisible(True)
            self.compressLabel.setVisible(False)
            self.compressSpinBox.setVisible(False)

    def check_params_algorithm(self, algorithm):

        if self.maxiterSpinBox.text().strip() == '':
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

    def run(self):
        # checks
        if not hasattr(self, 'fname') or self.fname is None:
            showWarning("Para iniciar, debe cargar el dato sísmico dando click al boton 'Cargar'")
            return

        algorithm_case = self.algorithmComboBox.currentText().upper()
        self.check_params_algorithm(algorithm_case)

        if self.saveAsLineEdit.text().strip() == '':
            showWarning("Por favor seleccione un nombre de archivo para guardar los resultados del algoritmo.")
            return

        try:
            # I took this values from ipynb notebook
            self.canvas.reset_values()
            self.resultProgressBar.setValue(0)
            self.outputTableWidget.setRowCount(0)

            self.maxiter = int(self.maxiterSpinBox.text())
            x = np.load(self.fname[0])
            x = x.T
            x = x / np.abs(x).max()

            '''
            ---------------  SAMPLING --------------------
            '''

            compresson_ratio = float(self.compressSpinBox.text().split('%')[0]) / 100
            sampling = Sampling(x)

            if self.samplingTypeComboBox.currentText().lower() == 'aleatorio':
                self.sampling_dict, H = sampling.random_sampling(compresson_ratio)

            elif self.samplingTypeComboBox.currentText().lower() == 'uniforme':
                self.sampling_dict, H = sampling.uniform_sampling(compresson_ratio)

            elif self.samplingTypeComboBox.currentText().lower() == 'jitter':
                n_bloque = int(self.jitterBlockSpinBox.text())
                self.sampling_dict, H = sampling.jitter_sampling(n_bloque, compresson_ratio)

            elif self.samplingTypeComboBox.currentText().lower() == 'lista':
                try:
                    lista = [int(number) for number in self.elementLineEdit.text().replace(' ', '').split(',')]

                    if len(lista) < 7:
                        showWarning("La cantidad mínima de elementos debe ser 7.")
                        return

                    if len(np.unique(lista)) < 7:
                        showWarning("La cantidad mínima de elementos repetidos debe ser 7.\n"
                                    "Preferiblemente no ingrese números repetidos.")
                        return

                    if x.shape[1] <= np.max(lista):
                        showWarning(f"El número de columnas de la muestra ({x.shape[1]}) es inferior al mayor número "
                                    f"ingresado en la lista de elementos ({np.max(lista)}).\n"
                                    "Por favor verifique la lista.")
                        return

                    if any(number < 0 for number in lista):
                        showWarning("No ingrese elementos menores que cero a la lista de elementos.")
                        return


                except:
                    showWarning("Expresión invalida en los elementos de la lista.\n"
                                "Verifique que ingreso correctamente los datos.")
                    return
                self.sampling_dict, H = sampling.list_sampling(lista)
            else:
                showWarning("Hubo un error con el submuestreo de la muestra sísmica, por favor revise el código.")
                return

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
            self.thread = QtCore.QThread()

            # Algorithms object

            Alg = Algorithms(x, H, 'DCT2D', 'IDCT2D')  # Assuming using DCT2D ad IDCT2D for all algorithms

            if algorithm_case == "FISTA":
                parameters = {
                    "lmb": float(self.param1LineEdit.text()),  # Tau
                    "mu": float(self.param2LineEdit.text()),  # Mu
                    "max_itr": self.maxiter
                }
                func = Alg.FISTA
            elif algorithm_case == "GAP":
                parameters = {
                    "lmb": float(self.param1LineEdit.text()),  # Tau
                    "max_itr": self.maxiter
                }
                func = Alg.GAP
            elif algorithm_case == "TWIST":
                parameters = {
                    "lmb": float(self.param1LineEdit.text()),  # Tau
                    "alpha": float(self.param2LineEdit.text()),  # Alpha
                    "beta": float(self.param3LineEdit.text()),  # Beta
                    "max_itr": self.maxiter
                }
                func = Alg.TwIST
            elif algorithm_case == "ADMM":
                parameters = {
                    "rho": float(self.param1LineEdit.text()),  # Rho
                    "gamma": float(self.param2LineEdit.text()),  # Gamma
                    "lmb": float(self.param3LineEdit.text()),  # Lambda
                    "max_itr": self.maxiter
                }
                func = Alg.ADMM
            else:
                showCritical(
                    "No se encontró el algoritmo. Por favor intente nuevamente o utilice un algoritmo diferente.")
                self.resultProgressBar.setValue(0)
                return
            # elif self.algorithmComboBox.currentText() == ""
            self.currentAlgorithmName = self.algorithmComboBox.currentText()

            self.worker = Worker(func, parameters, self.maxiter)

            self.worker.moveToThread(self.thread)

            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.progress.connect(self.report_progress)

            self.thread.start()

            self.worker.finished.connect(self.save_results)  # save results

            # Final resets
            self.startPushButton.setEnabled(False)
            self.thread.finished.connect(self.reset_values)

        except BaseException as err:
            msg = f"Unexpected {err=}, {type(err)=}"
            showCritical("Ocurrió un error inesperado al procesar el dato sísmico. Por favor, intente nuevamente o "
                         "utilice un dato diferente.", details=msg)
            self.resultProgressBar.setValue(0)
            return

    def report_progress(self, iter_val, err, psnr):
        self.resultProgressBar.setValue(int((iter_val / self.maxiter) * 100))

        # update table

        rowPosition = self.outputTableWidget.rowCount()
        self.outputTableWidget.insertRow(rowPosition)

        self.outputTableWidget.setItem(rowPosition, 0, QTableWidgetItem(f"{err}"))
        self.outputTableWidget.setItem(rowPosition, 1, QTableWidgetItem(f"{psnr}"))
        self.outputTableWidget.scrollToBottom()

        # update figure
        self.canvas.update_values(iter_val, err, psnr)

        if iter_val % (self.maxiter // 10) == 0 or iter_val == self.maxiter:
            self.canvas.update_figure()

    def reset_values(self):
        self.startPushButton.setEnabled(True)
        self.resultProgressBar.setValue(0)
        self.maxiter = 1
        self.worker = None
        self.thread = None
        ee.progress = None
        self.sampling_dict = None

    def save_results(self, res_dict):
        filepath = self.sname[0]
        np.savez(filepath, x_result=res_dict['result'], hist=res_dict['hist'], sampling=self.sampling_dict,
                 alg_name=self.currentAlgorithmName)
        print("Results saved [Ok]")


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
    MainWindow = UiMainWindow()

    qtRectangle = MainWindow.frameGeometry()
    centerPoint = QDesktopWidget().availableGeometry().center()
    qtRectangle.moveCenter(centerPoint)
    MainWindow.move(qtRectangle.topLeft())
    enterPoint = QDesktopWidget().availableGeometry().center()
    qtRectangle.moveCenter(centerPoint)
    MainWindow.move(qtRectangle.topLeft())

    MainWindow.show()
    sys.exit(app.exec_())
