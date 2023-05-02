from Desarrollo.ReDS.gui.main_window_v9 import Ui_mainWindow
import os
import sys
from Algorithms.Function import Sampling, Algorithms
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QCoreApplication
from workers import Worker, TuningWorker, ComparisonWorker, TabWorker
import numpy as np
from pathlib import Path
import segyio
from shutil import copyfile
import pandas as pd
from PyQt5.Qt import Qt
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from scipy.io import loadmat
from seed_help_window import UISeedHelpWindow
from jitter_window import UIJitterWindow
from about_window import UIAboutWindow
from element_help_window import UIElementHelpWindow
from equation_window import UIEquationWindow
from equation_comparison_window import UIComparisonEquationWindow
import platform
from itertools import product
from gui.scripts.alerts import showWarning, showCritical
from graphics import PerformanceGraphic, ReconstructionGraphic, TuningGraphic, ComparisonPerformanceGraphic, \
    ComparisonReconstructionGraphic, CustomToolbar


def solve_path(relative_path):
    """Solve path for different OS"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath('.'), relative_path)


class UIMainAWindow(QMainWindow, Ui_mainWindow):
    """
        Class for module A. Trace reconstruction.

        Attributes
        ----------
        global_variables : dict
            Dictionary with global variables.
        directories : dict
            Dictionary with directories.
        tab_widgets : dict
            Dictionary with tab widgets.

    """

    def __init__(self, launcher):
        super(UIMainAWindow, self).__init__()
        self.setupUi(self)
        self.launcher = launcher
        self.sampling = Sampling()
        self.global_variables = None
        # others
        self.init_global_variables()
        self.init_actions()
        self.init_visible_widgets(width=420)


    def element_help_clicked(self):
        '''
        Element help clicked.
        '''
        self.ui_element_help_window = UIElementHelpWindow()
        self.ui_element_help_window.setupUi()
        self.ui_element_help_window.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.ui_element_help_window.show()

    def seed_help_clicked(self):
        '''
        Seed help clicked.
        '''
        self.ui_seed_help_window = UISeedHelpWindow()
        self.ui_seed_help_window.setupUi()
        self.ui_seed_help_window.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.ui_seed_help_window.show()

    def jitter_sampling_clicked(self):
        '''
        Jitter sampling clicked.
        '''
        self.jitter_sampling_window = UIJitterWindow()
        self.jitter_sampling_window.setupUi()
        self.jitter_sampling_window.show()

    def init_visible_widgets(self, width=320):
        '''
        Initialize the visible widgets

        Parameters
        ----------
        width : int, optional
            The width of the widget, by default 320
        '''
        self.inputGroupBox.setMinimumWidth(width)
        self.algorithmGroupBox.setMinimumWidth(width)
        self.inputGroupBox.setMaximumWidth(width)
        self.algorithmGroupBox.setMaximumWidth(width)
        self.samplingGroupBox.setMaximumWidth(width)
        self.comparisonGroupBox.setMaximumWidth(width)
        self.runGroupBox.setMaximumWidth(width)
        self.tuningGroupBox.setMaximumWidth(width)
        self.resultGroupBox.setMaximumWidth(width)

        algorithm = self.algorithmComboBox.currentText().lower()
        tuning_type = self.paramTuningComboBox.currentText().lower()

        self.update_main_visible_algorithms(algorithm)
        self.update_tuning_visible_algorithms(algorithm, tuning_type)

        self.tuningGroupBox.setVisible(False)
        self.tuningTabWidget.setVisible(False)
        self.comparisonsToolBox.setVisible(False)
        self.comparisonGroupBox.setVisible(False)
        self.paramComboBox.view().setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.elementLabel.setVisible(False)
        self.elementLineEdit.setVisible(False)
        self.elementHelpButton.setVisible(False)
        self.gammaLabel.setVisible(False)
        self.gammaSpinBox.setVisible(False)
        self.epsilonLabel.setVisible(False)
        self.epsilonSpinBox.setVisible(False)
        self.jitterPushButton.setVisible(False)

    def init_global_variables(self):
        '''
        Initialize the global variables.

        These variables are used to store the data of the experiments.

        Variables
        ----------
        global_varibles : dict
            The global variables
                # Tab mode ['main', 'tuning', 'comparison']
                # View mode ['normal', 'report']
                # Data mode ['complete', 'incomplete']

        tab_mode : list
            The tab mode according to selected tool
        data_mode : list
            The view mode according to kind of datasets
        directories : dict
            The directories of the project
        state : dict
            The state of the project, considering tab mode, view mode and data mode
        graphics : dict
            The graphics of the project
        threads : list
            The threads of the project for multiple experiments
        workers : list
            The workers of the project for multiple experiments
        all_tabs : dict
            The tabs of the project for generating a normal or report view
        tab_widgets: dict
            The tab widgets of the project
        iters: int
            The number of iterations for experiments
        max_iter: int
            The maximum number of iterations for experiments
        max_iter_progress: int
            The maximum number of iterations for progress bar for multiple experiments
        icons_path : str
            The path of the icons
        param_type : str
            The type of the parameter
        main_params : list
            The main parameters
        tuning_params : list
            The tuning parameters
        comparison_params : list
            The comparison parameters
        '''
        self.global_variables = dict(tab_mode='main', view_mode='normal', data_mode='complete', algorithm_name='')

        tab_mode = ['main', 'tuning', 'comparison']
        data_mode = ['complete', 'incomplete']

        dirs = lambda: dict(uploaded=[], temp_saved='', saved='', report=[])
        self.directories = {t_mode: {d_mode: dirs() for d_mode in data_mode} for t_mode in tab_mode}

        self.state = dict(main=dict(progress=dict(iteration={}, error={}, psnr={}, ssim={}, tv={})),
                          tuning=dict(progress=dict(total_runs={}, fixed_params={}, current_scale={})),
                          comparison=dict(progress=dict(iteration={}, errors={}, psnrs={}, ssims={}, tvs={})))

        graphs = lambda x: {} if x == 'tuning' else dict(performance={}, report={})
        self.graphics = {t_mode: {d_mode: graphs(t_mode) for d_mode in data_mode} for t_mode in tab_mode}

        self.thread_tab = None
        self.worker_tab = None

        self.workers = []
        self.threads = []

        # tabs setup

        self.all_tabs = dict(normal=[], report=[])  # contains all tab references distributed by view mode
        self.tab_widgets = dict(main=[self.performanceTabWidget, self.reportTabWidget], tuning=[self.tuningTabWidget],
                                comparison=[self.comparisonPerformanceTabWidget, self.comparisonReportTabWidget])

        # parameters setup

        self.iters = 0
        self.max_iter = 1
        self.max_iter_progress = 1

        self.icons_path = 'assets/parameters'

        lmb = 'lambda'
        mu = 'mu'
        rho = 'rho'
        alpha = 'alpha'
        beta = 'beta'
        gamma = 'gamma'

        self.param_type = ['init', 'end', 'list']

        self.params = dict(fista=[[lmb, 0.1, 0.5], [mu, 0.3, 0.7]],
                           gap=[[lmb, 1.0, 1.5]],
                           twist=[[lmb, 0.9, 1.5], [alpha, 1.2, 1.7], [beta, 1.998, 2.3]],
                           admm=[[rho, 0.5, 1.5], [gamma, 1.0, 1.7], [lmb, 0.0078, 0.009]])

        self.main_params = [[self.param1Label, self.param1LineEdit],
                            [self.param2Label, self.param2LineEdit],
                            [self.param3Label, self.param3LineEdit]]

        self.tuning_params = [[self.param1InitLabel, self.param1InitLineEdit,
                               self.param1EndLabel, self.param1EndLineEdit],
                              [self.param2InitLabel, self.param2InitLineEdit,
                               self.param2EndLabel, self.param2EndLineEdit],
                              [self.param3InitLabel, self.param3InitLineEdit,
                               self.param3EndLabel, self.param3EndLineEdit]]

        self.comparison_params = [[self.compParam1LineEdit1, self.compParam2LineEdit1],
                                  [self.compParam1LineEdit2],
                                  [self.compParam1LineEdit3, self.compParam2LineEdit3, self.compParam3LineEdit3],
                                  [self.compParam1LineEdit4, self.compParam2LineEdit4, self.compParam3LineEdit4]]

    def init_actions(self):
        '''
        Initialize the actions.
        '''
        self.onlydouble = QtGui.QDoubleValidator(decimals=10)
        self.onlyInt = QtGui.QIntValidator()
        self.experimentProgressBar.setValue(0)

        # tab
        self.mainAction.triggered.connect(self.show_main)
        self.tuningAction.triggered.connect(self.show_tuning)
        self.comparisonAction.triggered.connect(self.show_comparison)
        self.aboutOfAction.triggered.connect(self.show_about_of)

        # algorithms

        self.dataComboBox.currentTextChanged.connect(self.update_data)
        self.algorithmComboBox.currentTextChanged.connect(self.algorithm_changed)
        self.algorithmPushButton.clicked.connect(self.algorithm_equation_clicked)
        self.comparisonAlgorithmPushButton.clicked.connect(self.comparison_algorithm_equation_clicked)
        self.elementHelpButton.clicked.connect(self.element_help_clicked)
        self.seedHelpButton.clicked.connect(self.seed_help_clicked)

        # tuning
        self.paramTuningComboBox.currentTextChanged.connect(self.param_tuning_changed)
        self.paramComboBox.currentTextChanged.connect(self.param_changed)
        self.jitterPushButton.clicked.connect(self.jitter_sampling_clicked)

        # sampling
        self.samplingTypeComboBox.currentTextChanged.connect(self.on_sampling_changed)

        # buttons

        self.loadPushButton.clicked.connect(self.load_files)
        self.clearDataPushButton.clicked.connect(self.clear_data)
        self.saveAsPushButton.clicked.connect(self.save_files)
        self.saveAsLineEdit.editingFinished.connect(self.save_as_text_changed)
        self.startPushButton.clicked.connect(self.start_experiment)
        self.resultPushButton.clicked.connect(self.show_results)

        self.seedCheckBox.stateChanged.connect(self.activate_seed)

    def closeEvent(self, event):
        '''
        Close event.

        Parameters
        ----------
        event : QEvent
            Close event.
        '''
        message_box = QMessageBox(self)
        message_box.pos()
        message_box.setIcon(QMessageBox.Question)
        message_box.setWindowTitle('Cerrar aplicación')
        message_box.setText('¿Estás segur@ que quieres cerrar este módulo?')
        message_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        yesButton = message_box.button(QMessageBox.Yes)
        yesButton.setText('Si')
        buttonN = message_box.button(QMessageBox.No)
        buttonN.setText('No')
        message_box.exec_()

        if message_box.clickedButton() == yesButton:
            event.accept()
            print('Window closed')
            self.launcher.setVisible(True)
            # sys.exit(0)
        else:
            event.ignore()
    def add_tab(self, data_name, tab_widget, graphic):
        '''
        Add a tab to the tab widget.
        '''
        tab = QtWidgets.QWidget()
        tab.setObjectName(data_name)
        tabHLayout = QtWidgets.QHBoxLayout(tab)
        tabHLayout.setObjectName(f"{data_name}TabHLayout")
        graphicWidget = QtWidgets.QWidget(tab)
        graphicWidget.setObjectName(f"{data_name}graphicWidget")
        graphicWidgetVLayout = QtWidgets.QVBoxLayout(graphicWidget)
        graphicWidgetVLayout.setContentsMargins(0, 0, 0, 0)
        graphicWidgetVLayout.setSpacing(0)
        graphicWidgetVLayout.setObjectName(f"{data_name}graphicWidgetVLayout")
        graphicVLayout = QtWidgets.QVBoxLayout()
        graphicVLayout.setSpacing(0)
        graphicVLayout.setObjectName(f"{data_name}graphicVLayout")
        graphicWidgetVLayout.addLayout(graphicVLayout)
        tabHLayout.addWidget(graphicWidget)
        tabHLayout.setStretch(0, 9)
        tab_widget.addTab(tab, "")

        _translate = QtCore.QCoreApplication.translate
        tab_widget.setTabText(tab_widget.indexOf(tab), _translate("mainWindow", data_name))

        # graphics

        graphicVLayout.addWidget(NavigationToolbar(graphic, self))
        graphicVLayout.addWidget(graphic)

        return [tab, graphic]

    def set_visible_normal_tabs(self, directories, data_mode, set_visible):
        '''
        Set visible the normal tabs.
        '''
        tab_mode = self.global_variables['tab_mode']
        for uploaded_directory in directories[data_mode]['uploaded']:
            tab_name = uploaded_directory.split('/')[-1].split('.')[0]

            for (tab_widget,) in zip(self.tab_widgets[tab_mode]):
                all_tabs = tab_widget.findChildren(QtWidgets.QWidget)
                index = [tab_widget.indexOf(x) for x in all_tabs if
                         tab_widget.indexOf(x) != -1 and tab_name in x.objectName()]
                if len(index) > 0:
                    tab_widget.setTabVisible(index[0], set_visible)

                    if set_visible:
                        tab_widget.setCurrentIndex(index[0])

    def load_report_tabs(self):
        '''
        Load the report tabs.
        '''
        tab_mode = self.global_variables['tab_mode']
        data_mode = self.global_variables['data_mode']
        is_complete = True if data_mode == 'complete' else False

        for uploaded_directory in self.directories[tab_mode][data_mode]['report']:
            if tab_mode == 'main':
                try:
                    data = np.load(uploaded_directory, allow_pickle=True)
                    performance_data = {item[0]: item[1] for item in data['performance_data']}

                    data_name = uploaded_directory.split('/')[-1].split('.')[0]

                    expPerformanceTab, performanceGraphic = self.add_tab(data_name, self.performanceTabWidget,
                                                                         PerformanceGraphic(is_complete=is_complete))
                    self.graphics['main'][data_mode]['performance'][data_name] = performanceGraphic
                    performanceGraphic.update_values(**performance_data)
                    performanceGraphic.update_figure()

                    expReportTab, reconstructionGraphic = self.add_tab(data_name, self.reportTabWidget,
                                                                       ReconstructionGraphic(is_complete=is_complete))
                    self.graphics['main'][data_mode]['report'][data_name] = reconstructionGraphic
                    reconstructionGraphic.update_report(data)
                    reconstructionGraphic.update_figure()

                    self.all_tabs['report'].append([expPerformanceTab, performanceGraphic])
                    self.all_tabs['report'].append([expReportTab, reconstructionGraphic])

                except BaseException as err:
                    msg = f"Unexpected {err=}, {type(err)=}"
                    showCritical(
                        "Se intentó cargar un resultados que no corresponden a la herramienta actual."
                        "Por favor, solo cargue resultados obtenidos en el menú principal", details=msg)
                    return

            elif tab_mode == 'tuning':
                try:
                    data = np.load(uploaded_directory, allow_pickle=True)
                    algorithm_name = str(data['algorithm']).lower()
                    tuning_data = pd.DataFrame({item[0]: item[1] for item in data['tuning_data']})
                    fixed_params = {item[0]: item[1] for item in data['fixed_params']}
                    current_scale = str(data['scale']).lower()

                    data_name = uploaded_directory.split('/')[-1].split('.')[0]

                    expTuningTab, tuningGraphic = self.add_tab(data_name, self.tuningTabWidget,
                                                               TuningGraphic(is_complete=is_complete))
                    self.graphics['tuning'][data_mode][data_name] = tuningGraphic
                    tuningGraphic.update_tuning(algorithm_name, tuning_data, fixed_params, current_scale)
                    tuningGraphic.update_figure()

                    self.all_tabs['report'].append([expTuningTab, tuningGraphic])

                except BaseException as err:
                    msg = f"Unexpected {err=}, {type(err)=}"
                    showCritical(
                        "Se intentó cargar un resultados que no corresponden a la herramienta actual."
                        "Por favor, solo cargue resultados obtenidos en el menú de ajuste de parámetros", details=msg)
                    return

            else:
                try:
                    data = np.load(uploaded_directory, allow_pickle=True)
                    comparison_data = {item[0]: item[1] for item in data['comparison_data']}

                    data_name = uploaded_directory.split('/')[-1].split('.')[0]

                    expCompPerformanceTab, compPerformanceGraphic = self.add_tab(data_name,
                                                                                 self.comparisonPerformanceTabWidget,
                                                                                 ComparisonPerformanceGraphic(
                                                                                     is_complete=is_complete))
                    self.graphics['comparison'][data_mode]['performance'][data_name] = compPerformanceGraphic
                    compPerformanceGraphic.update_values(**comparison_data)
                    compPerformanceGraphic.update_figure()

                    expCompReportTab, compReconstructionGraphic = self.add_tab(data_name,
                                                                               self.comparisonReportTabWidget,
                                                                               ComparisonReconstructionGraphic(
                                                                                   is_complete=is_complete))
                    self.graphics['comparison'][data_mode]['report'][data_name] = compReconstructionGraphic
                    compReconstructionGraphic.update_report(data)
                    compReconstructionGraphic.update_figure()

                    self.all_tabs['report'].append([expCompPerformanceTab, compPerformanceGraphic])
                    self.all_tabs['report'].append([expCompReportTab, compReconstructionGraphic])

                except BaseException as err:
                    msg = f"Unexpected {err=}, {type(err)=}"
                    showCritical(
                        "Se intentó cargar un resultados que no corresponden a la herramienta actual."
                        "Por favor, solo cargue resultados obtenidos en el menú de comparaciones", details=msg)
                    return

    def remove_report_tabs(self):
        '''
        Remove the report tabs.
        '''
        for page, graph in self.all_tabs['report']:
            for tab_widget in self.tab_widgets[self.global_variables['tab_mode']]:
                index = tab_widget.indexOf(page)
                tab_widget.removeTab(index)

        plt.close('all')

    def update_tabs(self):
        '''
        Update the tabs.
        '''
        self.remove_report_tabs()

        directories = self.directories[self.global_variables['tab_mode']]
        data_mode = self.global_variables['data_mode']
        if self.global_variables['view_mode'] == 'normal':  # visible
            self.set_visible_normal_tabs(directories, data_mode, True)

        else:  # view_mode == 'report'
            self.set_visible_normal_tabs(directories, data_mode, False)
            self.load_report_tabs()

        if data_mode == 'complete':
            self.set_visible_normal_tabs(directories, 'incomplete', False)
        else:
            self.set_visible_normal_tabs(directories, 'complete', False)

        # self.set_current_index_tab_last()

    def set_current_index_tab_last(self):
        '''
        Set the current index of the tab to the last one.
        '''
        self.performanceTabWidget.setCurrentIndex(self.performanceTabWidget.count() - 1)
        self.reportTabWidget.setCurrentIndex(self.reportTabWidget.count() - 1)
        self.tuningTabWidget.setCurrentIndex(self.tuningTabWidget.count() - 1)
        self.comparisonPerformanceTabWidget.setCurrentIndex(self.comparisonPerformanceTabWidget.count() - 1)
        self.comparisonReportTabWidget.setCurrentIndex(self.comparisonReportTabWidget.count() - 1)

    def update_tab_thread(self):
        '''
        Update the tabs in a thread.
        '''
        self.thread_tab = QtCore.QThread()
        self.worker_tab = TabWorker()

        self.worker_tab.moveToThread(self.thread_tab)
        self.thread_tab.started.connect(self.worker_tab.run)
        self.worker_tab.finished.connect(self.thread_tab.quit)
        self.worker_tab.finished.connect(self.worker_tab.deleteLater)
        self.thread_tab.finished.connect(self.thread_tab.deleteLater)
        self.worker_tab.progress.connect(self.update_tabs)
        # Start the thread_tab
        self.thread_tab.start()

        # Final resets
        self.mainAction.setEnabled(False)
        self.tuningAction.setEnabled(False)
        self.comparisonAction.setEnabled(False)
        self.aboutOfAction.setEnabled(False)

        self.loadPushButton.setEnabled(False)
        self.loadPushButton.setText('Cargando...')
        self.clearDataPushButton.setEnabled(False)

        self.resultPushButton.setEnabled(False)
        self.thread_tab.finished.connect(self.update_tab_finished)

    def update_tab_finished(self):
        '''
        Update the tabs finished.
        '''
        self.mainAction.setEnabled(True)
        self.tuningAction.setEnabled(True)
        self.comparisonAction.setEnabled(True)
        self.aboutOfAction.setEnabled(True)

        self.loadPushButton.setEnabled(True)
        self.loadPushButton.setText('Cargar')
        self.clearDataPushButton.setEnabled(True)

        self.resultPushButton.setEnabled(True)

    def update_main_visible_algorithms(self, algorithm):
        '''
        Update the main visible algorithms.
        '''
        for i in range(3):
            label = self.main_params[i][0]
            line_edit = self.main_params[i][1]

            comparison = i < len(self.params[algorithm])
            if comparison:
                param_names = self.params[algorithm][i][0]
                icon_path = f'{self.icons_path}/{param_names}.png'

                value = str(self.params[algorithm][i][1])

                label.setPixmap(QtGui.QPixmap(solve_path(icon_path)))
                line_edit.setText(value)

            comparison1 = True if comparison else False

            label.setVisible(comparison1)
            line_edit.setVisible(comparison1)

            line_edit.setValidator(self.onlydouble)

    def update_tuning_visible_param(self, param):
        '''
        Update the tuning visible parameters.
        '''
        algorithm = self.algorithmComboBox.currentText().lower()
        tuning_type = self.paramTuningComboBox.currentText().lower()
        self.update_tuning_visible_algorithms(algorithm, tuning_type)

    def update_tuning_visible_algorithms(self, algorithm, tuning_type):
        '''
        Update the tuning visible algorithms.
        '''
        for i in range(3):
            label_init = self.tuning_params[i][0]
            line_edit_init = self.tuning_params[i][1]
            label_end = self.tuning_params[i][2]
            line_edit_end = self.tuning_params[i][3]

            comparison1 = i < len(self.params[algorithm])
            if comparison1:
                param_names = self.params[algorithm][i][0]
                icon_path = f'{self.icons_path}/{param_names}'

                icon_path_init = f'{icon_path}_{self.param_type[0]}.png'
                icon_path_end = f'{icon_path}_{self.param_type[1]}.png'
                icon_path_list = f'{icon_path}_{self.param_type[2]}.png'

                value_init = str(self.params[algorithm][i][1])
                value_end = str(self.params[algorithm][i][2])

                label_init.setPixmap(
                    QtGui.QPixmap(solve_path(icon_path_init if tuning_type == 'intervalo' else icon_path_list)))
                label_end.setPixmap(QtGui.QPixmap(solve_path(icon_path_end)))

                line_edit_init.setText(value_init)
                line_edit_end.setText(value_end)

            comparison2 = True if tuning_type == 'intervalo' else False

            comparison3 = True if comparison1 else False
            comparison4 = True if comparison1 and comparison2 else False

            label_init.setVisible(comparison3)
            line_edit_init.setVisible(comparison3)

            label_end.setVisible(comparison4)
            line_edit_end.setVisible(comparison4)

            line_edit_init.setValidator(self.onlydouble if tuning_type == 'intervalo' else None)
            line_edit_end.setValidator(self.onlydouble if tuning_type == 'intervalo' else None)

            if i != self.paramComboBox.currentIndex():
                label_init.setPixmap(QtGui.QPixmap(solve_path(icon_path)))
                label_end.setVisible(False)
                line_edit_end.setVisible(False)

    def set_visible_algorithm(self, algorithm):
        '''
        Set the visible algorithm.
        '''
        algorithm = algorithm.lower()

        if self.global_variables['tab_mode'] == 'main':
            self.update_main_visible_algorithms(algorithm)

        else:
            count = 0
            for i in range(3):
                if i < len(self.params[algorithm]):
                    param_names = self.params[algorithm][i][0]
                    icon_path = f'{self.icons_path}/{param_names}.png'

                    self.paramComboBox.setItemIcon(i, QtGui.QIcon(icon_path))
                    count += 1

            self.paramComboBox.setCurrentIndex(0)
            self.paramComboBox.setMaxVisibleItems(count)

            tuning_param = self.paramTuningComboBox.currentText().lower()
            self.update_tuning_visible_algorithms(algorithm, tuning_param)

    def clear_data(self):
        '''
        Clear the data in the software.
        '''
        if self.dataTreeWidget.findItems('', Qt.MatchContains | Qt.MatchRecursive):
            message_box = QtWidgets.QMessageBox(self)
            message_box.pos()
            message_box.setIcon(QtWidgets.QMessageBox.Question)
            message_box.setWindowTitle('Limpiar area de trabajo')
            message_box.setText('¿Estás segur@ que quiere remover todos los datos sísmicos del area de trabajo?'
                                'Tendrá que volverlos a cargar si desea usarlos nuevamente.')
            message_box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            yesButton = message_box.button(QtWidgets.QMessageBox.Yes)
            yesButton.setText('Si')
            buttonN = message_box.button(QtWidgets.QMessageBox.No)
            buttonN.setText('No')
            message_box.exec_()

            if message_box.clickedButton() == yesButton:
                self.dataTreeWidget.clear()

                tab_mode = self.global_variables['tab_mode']
                view_mode = self.global_variables['view_mode']
                data_mode = self.global_variables['data_mode']

                self.directories[tab_mode][self.global_variables['data_mode']][
                    'uploaded' if view_mode == 'normal' else 'report'] = []
                self.update_tab_thread()

                # clear tabs
                if tab_mode in ['main', 'comparison']:
                    self.graphics[tab_mode][data_mode]['performance'] = {}
                    self.graphics[tab_mode][data_mode]['report'] = {}
                else:
                    self.graphics[tab_mode][data_mode] = {}

                for tab_widget in self.tab_widgets[tab_mode]:
                    for page, graph in self.all_tabs[view_mode if view_mode == 'report' else 'normal']:
                        index = tab_widget.indexOf(page)
                        tab_widget.removeTab(index)

                    plt.close('all')

    def update_directories(self, file_type, filenames):
        '''
        Update the directories.
        '''
        tab_mode = self.global_variables['tab_mode']
        data_mode = self.global_variables['data_mode']

        # validate dir with current tool

        if file_type == 'report':
            valid_dict = dict(main=['performance_data', 'principal'], tuning=['tuning_data', 'de ajuste'],
                              comparison=['comparison_data', 'de comparación'])
            idx_list = []
            for i, filepath in enumerate(filenames):
                filename = filepath.split('/')[-1]
                data = np.load(filepath, allow_pickle=True)

                if not valid_dict[tab_mode][0] in list(data.keys()):
                    idx_list.append(i)

                    if 'performance_data' in list(data.keys()):
                        tool = valid_dict['main'][1]
                    elif 'tuning_data' in list(data.keys()):
                        tool = valid_dict['tuning'][1]
                    else:
                        tool = valid_dict['comparison'][1]

                    showWarning(f"Se intentó cargar {filename} obtenido de la herramienta {tool}. "
                                f"Por favor, solo cargue resultados obtenidos con la heramienta "
                                f"{valid_dict[tab_mode][1]}.")

            filenames = [filename for i, filename in enumerate(filenames) if i not in idx_list]

        new_filenames = []
        for filepath in filenames:
            filename = filepath.split('/')[-1]
            existing_filenames = [fnames.split('/')[-1] for fnames in
                                  self.directories[tab_mode][data_mode][file_type]]
            if filename in existing_filenames:
                showWarning(f"El dato con el nombre {filename} ya está cargado. Se descartará esta nueva carga.")
            else:
                self.directories[tab_mode][data_mode][file_type].append(filepath)
                new_filenames.append(filename)

    def load_files(self):
        '''
        Load files in the software as filenames.
        '''
        kwargs = {}
        if 'SNAP' in os.environ:
            kwargs['options'] = QtWidgets.QFileDialog.DontUseNativeDialog

        tab_mode = self.global_variables['tab_mode']
        view_mode = self.global_variables['view_mode']
        data_mode = self.global_variables['data_mode']

        uploaded_directory = self.directories[tab_mode][data_mode][view_mode if view_mode == 'report' else 'uploaded']

        if not uploaded_directory:
            uploaded_directory = ['']

        if view_mode == 'normal':
            message = 'Abrir dato sísmico'
            file_type = 'npy'
        else:  # 'report'
            message = 'Abrir datos sísmicos reconstruidos'
            file_type = 'npz'

        self.data_fname = QtWidgets.QFileDialog.getOpenFileNames(self, message, uploaded_directory[-1],
                                                                 filter=f'todos los archivos (*.mat *.npy *segy *sgy);;numpy file (*.{file_type});;matlab file (*.mat);;segy file (*.sgy *.segy)',
                                                                 **kwargs)

        if self.data_fname[0] in ['', []]:
            return
        self.data_fname = list(self.data_fname)

        # verify if data is complete or incomplete (optional)
        self.verify_type_data(self.data_fname[0])

        if view_mode == 'normal':
            self.update_directories('uploaded', self.data_fname[0])
            self.update_data_tree(self.directories[tab_mode][data_mode]['uploaded'])

        else:  # view_mode == 'report'
            self.update_directories('report', self.data_fname[0])
            self.update_data_tree(self.directories[tab_mode][data_mode]['report'])
            self.update_tab_thread()

    def verify_type_data(self, directories):
        '''
        Verify if data is complete or incomplete.
        '''
        view_mode = self.global_variables['view_mode']
        data_type = self.dataComboBox.currentText().lower()

        indices = []
        for i, directory in enumerate(directories):
            filename = directory.split('/')
            child_name = filename[-1]

            data = self.load_seismic_data(directory)[1:-1, 1:-1]
            data = np.nan_to_num(data, nan=0)

            # check if some row or column of data contains only zeros
            if np.all(data == 0, axis=1).any():
                if data_type == 'datos completos':
                    showWarning(f"El dato cargado {child_name} tiene algunas filas en zeros.")
            else:
                if data_type == 'datos incompletos':
                    showWarning(f"El dato cargado {child_name} no es incompleto. Se ignorará el dato.")
                    continue

            indices.append(i)

        self.data_fname[0] = [self.data_fname[0][i] for i in indices]

    def save_files(self):
        '''
        Save files in the software with filenames.
        '''
        kwargs = {}
        if 'SNAP' in os.environ:
            kwargs['options'] = QFileDialog.DontUseNativeDialog

        tab_mode = self.global_variables['tab_mode']
        data_mode = self.global_variables['data_mode']

        directories = self.directories[tab_mode][data_mode]
        temp_saved_directory = directories['temp_saved']
        if temp_saved_directory == '':
            temp_saved_directory = directories['uploaded']

        if not temp_saved_directory:
            temp_saved_directory = ['']

        save_name = QFileDialog.getSaveFileName(self, 'Guardar reconstrucciones', temp_saved_directory[-1],
                                                # filter=f'numpy file (*.npz)',
                                                **kwargs)
        if save_name[0] == '':
            return

        self.saveAsLineEdit.setText(save_name[0])
        self.directories[tab_mode][data_mode]['temp_saved'] = save_name[0]

    def update_data_tree(self, directories):
        '''
        Update data tree with the new data.
        '''
        self.dataTreeWidget.clear()

        if directories in ['', []]:
            return

        for directory in directories:
            filename = directory.split('/')
            parent_name = filename[-2]
            child_name = filename[-1]

            # parent = self.dataTreeWidget.findItems('', Qt.MatchContains | Qt.MatchRecursive)
            parent = self.dataTreeWidget.findItems(parent_name, Qt.MatchContains)

            if parent:
                child = self.dataTreeWidget.findItems(child_name, Qt.MatchContains | Qt.MatchRecursive)

                if not child:
                    new_child = QtWidgets.QTreeWidgetItem([child_name])
                    parent[0].addChild(new_child)

            else:
                parent = QtWidgets.QTreeWidgetItem(self.dataTreeWidget)
                parent.setText(0, parent_name)
                child = QtWidgets.QTreeWidgetItem(parent)
                child.setText(0, filename[-1])

                parent.setExpanded(True)

    def save_as_text_changed(self):
        '''
        Update the save name when the text is changed.
        '''
        save_name = self.saveAsLineEdit.text()

        op_sys = platform.system().lower()
        if op_sys == 'windows':
            if len(save_name.split('\\')) == 1:
                save_name = f'C:{os.environ["HOMEPATH"]}\\{save_name}'
        elif op_sys == 'linux' or op_sys == 'darwin':
            if len(save_name.split('/')) == 1:
                save_name = f'{os.environ["HOME"]}/{save_name}'
        else:
            showWarning('No se pudo determinar el sistema operativo. Por favor, especifique la ruta completa.')
            return

        self.saveAsLineEdit.setText(save_name)
        self.directories[self.global_variables['tab_mode']][self.global_variables['data_mode']][
            'temp_saved'] = save_name

    def show_about_of(self):
        '''
        Show about of the software.
        '''
        self.about_window = QtWidgets.QWidget()
        self.ui_about_window = UIAboutWindow()
        self.ui_about_window.setupUi(self.about_window)
        self.about_window.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.about_window.show()

    def show_main(self):
        '''
        Show main window mode.
        '''
        self.global_variables['tab_mode'] = 'main'
        view_mode = self.global_variables['view_mode']
        comparison = True if view_mode == 'normal' else False

        self.algorithmGroupBox.setVisible(comparison)
        self.tuningGroupBox.setVisible(False)
        self.comparisonGroupBox.setVisible(False)
        self.resultsToolBox.setVisible(True)
        self.tuningTabWidget.setVisible(False)
        self.comparisonsToolBox.setVisible(False)

        self.set_visible_algorithm(self.algorithmComboBox.currentText().lower())
        self.set_result_view()

    def show_tuning(self):
        '''
        Show tuning window mode.
        '''
        self.global_variables['tab_mode'] = 'tuning'
        view_mode = self.global_variables['view_mode']
        comparison = True if view_mode == 'normal' else False

        self.algorithmGroupBox.setVisible(comparison)
        self.tuningGroupBox.setVisible(comparison)
        self.comparisonGroupBox.setVisible(False)
        self.resultsToolBox.setVisible(False)
        self.tuningTabWidget.setVisible(True)
        self.comparisonsToolBox.setVisible(False)

        self.param1Label.setVisible(False)
        self.param1LineEdit.setVisible(False)
        self.param2Label.setVisible(False)
        self.param2LineEdit.setVisible(False)
        self.param3Label.setVisible(False)
        self.param3LineEdit.setVisible(False)

        self.set_visible_algorithm(self.algorithmComboBox.currentText().lower())
        self.set_result_view()

    def show_comparison(self):
        '''
        Show comparison window mode.
        '''
        self.global_variables['tab_mode'] = 'comparison'
        view_mode = self.global_variables['view_mode']
        comparison = True if view_mode == 'normal' else False

        self.algorithmGroupBox.setVisible(False)
        self.tuningGroupBox.setVisible(False)
        self.comparisonGroupBox.setVisible(comparison)
        self.resultsToolBox.setVisible(False)
        self.tuningTabWidget.setVisible(False)
        self.comparisonsToolBox.setVisible(True)

        self.set_visible_algorithm(self.algorithmComboBox.currentText().lower())
        self.set_result_view()

    def set_result_view(self):
        '''
        Set the result view.
        '''
        self.update_tab_thread()
        tab_mode = self.global_variables['tab_mode']
        data_mode = self.global_variables['data_mode']

        if self.global_variables['view_mode'] == 'normal':
            self.saveAsLineEdit.setText(self.directories[tab_mode][data_mode]['temp_saved'])
            self.update_data_tree(self.directories[tab_mode][data_mode]['uploaded'])

        else:
            self.update_data_tree(self.directories[tab_mode][data_mode]['report'])

    def show_results(self):
        '''
        Show results window mode.
        '''
        self.dataTreeWidget.clear()

        icon = QtGui.QIcon()
        if self.global_variables['view_mode'] == 'normal':
            self.global_variables['view_mode'] = 'report'
            self.set_report_view()
            icon.addPixmap(QtGui.QPixmap(solve_path("assets/icons/seismic.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.resultLabel.setText('Realizar experimentos')

        else:
            self.global_variables['view_mode'] = 'normal'
            self.set_main_view()
            icon.addPixmap(QtGui.QPixmap(solve_path("assets/icons/report.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.resultLabel.setText('Ver resultados')

        self.resultPushButton.setIcon(icon)

    def set_main_view(self):
        '''
        Set the main view.
        '''
        self.set_visible_algorithm(self.algorithmComboBox.currentText().lower())

        tab_mode = self.global_variables['tab_mode']
        self.algorithmGroupBox.setVisible(True if tab_mode == 'main' else False)
        self.tuningGroupBox.setVisible(True if tab_mode == 'tuning' else False)
        self.samplingGroupBox.setVisible(True)
        self.runGroupBox.setVisible(True)
        self.comparisonGroupBox.setVisible(True if tab_mode == 'comparison' else False)

        _translate = QtCore.QCoreApplication.translate
        self.inputGroupBox.setTitle(_translate("mainWindow", "Datos sísmicos"))

        self.set_result_view()

    def set_report_view(self):
        '''
        Set the report view.
        '''
        self.algorithmGroupBox.setVisible(False)
        self.tuningGroupBox.setVisible(False)
        self.samplingGroupBox.setVisible(False)
        self.runGroupBox.setVisible(False)
        self.comparisonGroupBox.setVisible(False)

        _translate = QtCore.QCoreApplication.translate
        self.inputGroupBox.setTitle(_translate("mainWindow", "Datos sísmicos reconstruidos"))

        self.set_result_view()

    def update_data(self, value):
        '''
        Update data.
        '''
        if value.lower() == 'datos completos':
            self.samplingGroupBox.setVisible(True if self.global_variables['view_mode'] == 'normal' else False)
            self.global_variables['data_mode'] = 'complete'
        else:
            self.samplingGroupBox.setVisible(False)
            self.global_variables['data_mode'] = 'incomplete'

        self.set_visible_algorithm(self.algorithmComboBox.currentText().lower())
        self.set_result_view()

    def algorithm_changed(self, value):
        '''
        Algorithm changed.
        '''
        self.set_visible_algorithm(value.lower())

    def algorithm_equation_clicked(self):
        '''
        Algorithm equation clicked.
        '''
        self.ui_equation_window = UIEquationWindow()
        self.ui_equation_window.setupUi(self.algorithmComboBox.currentText())
        self.ui_equation_window.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.ui_equation_window.show()

    def comparison_algorithm_equation_clicked(self):
        '''
        Comparison algorithm equation clicked.
        '''
        self.ui_comparison_equation_window = UIComparisonEquationWindow()
        self.ui_comparison_equation_window.setupUi()
        self.ui_comparison_equation_window.show()


    def param_tuning_changed(self, value):
        '''
        Param tuning changed.
        '''
        self.paramValuesLabel.setVisible(True if value.lower() == 'intervalo' else False)
        self.paramValuesSpinBox.setVisible(True if value.lower() == 'intervalo' else False)

        algorithm = self.algorithmComboBox.currentText().lower()
        self.update_tuning_visible_algorithms(algorithm, value.lower())

    def param_changed(self, value):
        '''
        Param changed.
        '''
        self.update_tuning_visible_param(value.lower())

    def activate_seed(self, activate):
        '''
        Activate seed.
        '''
        self.seedSpinBox.setEnabled(activate)

    def on_sampling_changed(self, value):
        '''
        On sampling changed.
        '''
        sampling = value.lower()

        self.spacerItem4.changeSize(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.spacerItem5.changeSize(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.samplingHLine.setVisible(False if sampling in 'uniforme' else True)

        visible = True if sampling not in ['jitter', 'lista'] else False
        self.compressLabel.setVisible(visible)
        self.compressSpinBox.setVisible(visible)
        self.compressSpinBox.setMaximum(99 if sampling in ['aleatorio', 'jitter'] else 50)

        visible = True if sampling in ['aleatorio', 'jitter'] else False
        self.seedCheckBox.setVisible(visible)
        self.seedLabel.setVisible(visible)
        self.seedSpinBox.setVisible(visible)
        self.seedHelpButton.setVisible(visible)

        visible = True if sampling in 'jitter' else False
        self.gammaLabel.setVisible(visible)
        self.gammaSpinBox.setVisible(visible)
        self.epsilonLabel.setVisible(visible)
        self.epsilonSpinBox.setVisible(visible)
        self.jitterPushButton.setVisible(visible)

        visible = True if sampling in 'lista' else False
        self.elementLabel.setVisible(visible)
        self.elementLineEdit.setVisible(visible)
        self.elementHelpButton.setVisible(visible)

        if sampling in ['aleatorio']:
            self.spacerItem4.changeSize(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
            self.spacerItem5.changeSize(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

    def verify_parameters(self, uploaded_directories):
        '''
        Verify parameters for experiments.
        '''
        if not uploaded_directories:
            return False

        for uploaded_directory in uploaded_directories:
            if uploaded_directory == '':
                showWarning("Para iniciar, debe cargar el dato sísmico dando click al boton 'Cargar'")
                return False

            if self.directories[self.global_variables['tab_mode']][self.global_variables['data_mode']][
                'temp_saved'] == '':
                showWarning("Por favor seleccione un nombre de archivo para guardar los resultados del algoritmo.")
                return False

            if self.global_variables['tab_mode'] == 'tuning':
                algorithm = self.algorithmComboBox.currentText().lower()
                tuning_type = self.paramTuningComboBox.currentText().lower()
                fixed_param = self.paramComboBox.currentIndex()

                validate_interval = True
                validate_list = True

                for i in range(len(self.params[algorithm])):
                    number_init = self.tuning_params[i][1].text()
                    number_end = self.tuning_params[i][3].text()

                    i_comparison = i == fixed_param
                    if tuning_type == 'intervalo' and i_comparison:
                        validate_interval = float(number_init) < float(number_end)

                    if tuning_type == 'lista' and i_comparison:
                        try:
                            lista = [float(number) for number in number_init.replace(' ', '').split(',')]
                            lista.sort()

                            if any(number < 0 for number in lista):
                                validate_list = False

                        except:

                            validate_list = False

                if not validate_interval:
                    showWarning("Los parámetros iniciales deben ser menores que los parámetros finales.")
                    return False

                if not validate_list:
                    showWarning("La sintaxis de la lista no es correcta, verifiquela e ingresela nuevamente.")
                    return False

        return True

    def update_variables(self, data_name):
        '''
        Update variables for experiments.
        '''
        self.experimentProgressBar.setValue(0)

        tab_mode = self.global_variables['tab_mode']
        if tab_mode in ['main', 'comparison']:
            suffix = '' if tab_mode == 'main' else 's'
            self.state[tab_mode]['progress']['iteration'][data_name] = []
            self.state[tab_mode]['progress'][f'error{suffix}'][data_name] = []
            self.state[tab_mode]['progress'][f'psnr{suffix}'][data_name] = []
            self.state[tab_mode]['progress'][f'ssim{suffix}'][data_name] = []
            self.state[tab_mode]['progress'][f'tv{suffix}'][data_name] = []

        else:
            self.state[tab_mode]['progress']['total_runs'][data_name] = 0
            self.state[tab_mode]['progress']['fixed_params'][data_name] = {}

    def load_seismic_data(self, uploaded_directory):
        '''
        Load seismic data for experiments.
        '''
        if Path(uploaded_directory).suffix == '.npy':
            data = np.load(uploaded_directory)
        elif Path(uploaded_directory).suffix == '.mat':
            data = loadmat(uploaded_directory)
            keys = list(data.keys())
            keys.remove('__header__')
            keys.remove('__version__')
            keys.remove('__globals__')
            data = data[keys[0]]
        elif Path(uploaded_directory).suffix.lower() == '.sgy' or Path(uploaded_directory).suffix.lower() == '.segy':
            data = None
            with segyio.open(uploaded_directory) as f:
                data = np.zeros((len(f.samples), len(f.xlines)))
                pos = 0
                for trace in f.trace:
                    data[:, pos] = trace
                    pos = pos + 1

        if data.ndim > 2:
            data = data[..., int(data.shape[-1] / 2)]
        if self.TransposeCheckBox.isChecked():
            data = data.T
        data = np.nan_to_num(data, nan=0)
        data = data / np.max(np.abs(data))

        # data direction
        data = np.nan_to_num(data, nan=0)

        return data

    def load_parameters(self, data):
        '''
        Load parameters for experiments.
        '''
        data_mode = self.global_variables['data_mode']
        seed = None
        if self.seedCheckBox.checkState():
            seed = int(self.seedSpinBox.text())

        compression_ratio = float(self.compressSpinBox.text().split('%')[0]) / 100

        mode = self.samplingTypeComboBox.currentText().lower()
        jitter_params = dict(gamma=int(self.gammaSpinBox.text()), epsilon=int(self.epsilonSpinBox.text()))
        lista = self.elementLineEdit

        try:
            sampling_dict, H = self.sampling.apply_sampling(data, mode, jitter_params, lista, seed,
                                                            compression_ratio)

            if data_mode == 'incomplete':
                sampling_dict = {key: value for key, value in sampling_dict if key == 'x_ori'}

        except:
            return

        return sampling_dict, H if data_mode == 'complete' else None

    def load_algorithm(self, data_name, seismic_data, H, sampling_dict):
        '''
        Load algorithm for experiments.
        '''
        self.algorithm_name = self.algorithmComboBox.currentText().lower()
        tuning_type = self.paramTuningComboBox.currentText().lower()
        fixed_param = self.paramComboBox.currentIndex()
        self.current_scale = self.scaleComboBox.currentText().lower()
        data_mode = self.global_variables['data_mode']
        is_complete = True if data_mode == 'complete' else False

        if self.global_variables['tab_mode'] == 'main':
            params = dict(param1=self.param1LineEdit.text(),
                          param2=self.param2LineEdit.text(),
                          param3=self.param3LineEdit.text())

            Alg = Algorithms(seismic_data, H, 'DCT2D', 'IDCT2D')  # Assuming using DCT2D ad IDCT2D for all algorithms
            algorithm, parameters = Alg.get_algorithm(self.algorithm_name, self.max_iter, **params)

            if data_name in self.graphics['main'][data_mode]['performance'].keys():
                performance_graphic = self.graphics['main'][data_mode]['performance'][data_name]
            else:
                performance_tab, performance_graphic = self.add_tab(data_name, self.performanceTabWidget,
                                                                    PerformanceGraphic(is_complete=is_complete))
                self.graphics['main'][data_mode]['performance'][data_name] = performance_graphic
                self.all_tabs['normal'].append([performance_tab, performance_graphic])

            if data_name in self.graphics['main'][data_mode]['report'].keys():
                report_graphic = self.graphics['main'][data_mode]['report'][data_name]
            else:
                report_tab, report_graphic = self.add_tab(data_name, self.reportTabWidget,
                                                          ReconstructionGraphic(is_complete=is_complete))
                self.graphics['main'][data_mode]['report'][data_name] = report_graphic
                self.all_tabs['normal'].append([report_tab, report_graphic])

            # update worker behaviour

            if not is_complete:
                sampling_dict['H'] = Alg.H_raw
                sampling_dict = np.array(list(sampling_dict.items()), dtype=object)
            return Worker(data_name, algorithm, parameters, self.max_iter, sampling_dict,
                          performance_graphic, report_graphic)

        elif self.global_variables['tab_mode'] == 'tuning':
            param_list = []
            parameters = []

            if data_name in self.graphics['tuning'][data_mode].keys():
                tuning_graphic = self.graphics['tuning'][data_mode][data_name]
            else:
                tuning_tab, tuning_graphic = self.add_tab(data_name, self.tuningTabWidget,
                                                          TuningGraphic(is_complete=is_complete))
                self.graphics['tuning'][data_mode][data_name] = tuning_graphic
                self.all_tabs['normal'].append([tuning_tab, tuning_graphic])

            num_params = len(self.params[self.algorithm_name])
            for i in range(num_params):

                number_init = self.tuning_params[i][1].text()
                number_end = self.tuning_params[i][3].text()

                i_comparison = i == fixed_param
                if tuning_type == 'intervalo':
                    if i_comparison:
                        scale = np.linspace
                        if self.current_scale != 'lineal':
                            scale = np.logspace
                            number_init, number_end = np.log10(float(number_init)), np.log10(float(number_end))

                        param_list.append(list(scale(float(number_init), float(number_end),
                                                     int(self.paramValuesSpinBox.text()))))

                    else:
                        num_init = float(number_init)
                        aux_fixed_param = {self.params[self.algorithm_name][i][0]: num_init}
                        self.state[self.global_variables['tab_mode']]['progress']['fixed_params'][data_name].update(
                            aux_fixed_param)
                        param_list.append([num_init])

                if tuning_type == 'lista':
                    if i_comparison:
                        lista = [float(number) for number in number_init.replace(' ', '').split(',')]
                        lista.sort()

                        param_list.append(lista)

                    else:
                        num_init = float(number_init)
                        aux_fixed_param = {self.params[self.algorithm_name][i][0]: num_init}
                        self.state[self.global_variables['tab_mode']]['progress']['fixed_params'][data_name].update(
                            aux_fixed_param)
                        param_list.append([num_init])

            func = None
            param_arg_names = ['param1', 'param2', 'param3']
            for ps in product(*param_list):
                aux_params = {param_arg_names[i]: ps[i] for i in range(num_params)}

                Alg = Algorithms(seismic_data, H, 'DCT2D', 'IDCT2D')
                func, params = Alg.get_algorithm(self.algorithm_name, self.max_iter, **aux_params)

                parameters.append(params)

            self.total_num_run = len(parameters)

            # update worker behaviour
            return TuningWorker(data_name, func, parameters, self.max_iter, tuning_graphic)

        else:
            funcs = []
            param_list = []

            if data_name in self.graphics['comparison'][data_mode]['performance'].keys():
                comp_performance_graphic = self.graphics['comparison'][data_mode]['performance'][data_name]
            else:
                comp_performance_tab, comp_performance_graphic = self.add_tab(data_name,
                                                                              self.comparisonPerformanceTabWidget,
                                                                              ComparisonPerformanceGraphic(
                                                                                  is_complete=is_complete))
                self.graphics['comparison'][data_mode]['performance'][data_name] = comp_performance_graphic
                self.all_tabs['normal'].append([comp_performance_tab, comp_performance_graphic])

            if data_name in self.graphics['comparison'][data_mode]['report'].keys():
                comp_report_graphic = self.graphics['comparison'][data_mode]['report'][data_name]
            else:
                comp_report_tab, comp_report_graphic = self.add_tab(data_name, self.comparisonReportTabWidget,
                                                                    ComparisonReconstructionGraphic(
                                                                        is_complete=is_complete))
                self.graphics['comparison'][data_mode]['report'][data_name] = comp_report_graphic
                self.all_tabs['normal'].append([comp_report_tab, comp_report_graphic])

            algorithm_names = ['fista', 'gap', 'twist', 'admm','deep-red']
            param_arg_names = ['param1', 'param2', 'param3']
            for alg_name, params in zip(algorithm_names, self.comparison_params):
                aux_params = {param_arg_names[i]: param.text() for i, param in enumerate(params)}

                Alg = Algorithms(seismic_data, H, 'DCT2D', 'IDCT2D')
                func, params = Alg.get_algorithm(alg_name, self.max_iter, **aux_params)

                funcs.append(func)
                param_list.append(params)

            # # update worker behaviour

            if not is_complete:
                sampling_dict['H'] = Alg.H_raw
                sampling_dict = np.array(list(sampling_dict.items()), dtype=object)
            return ComparisonWorker(data_name, funcs, param_list, self.max_iter, sampling_dict,
                                    comp_performance_graphic, comp_report_graphic)

    def start_experiment(self):
        '''
        Start the experiment
        '''
        uploaded_directories = self.directories[self.global_variables['tab_mode']][self.global_variables['data_mode']][
            'uploaded']
        validate = self.verify_parameters(uploaded_directories)

        if not validate:
            showWarning("Los directorios de archivos no son validos. Por favor cargarlos nuevamente.")
            return

        try:
            self.iters = 0
            self.max_iter = int(self.maxiterSpinBox.text())

            for uploaded_directory in uploaded_directories:
                self.current_file = uploaded_directory
                data_name = uploaded_directory.split('/')[-1].split('.')[0]

                self.update_variables(data_name)
                seismic_data = self.load_seismic_data(uploaded_directory)
                sampling_dict, H = self.load_parameters(seismic_data)
                worker = self.load_algorithm(data_name, seismic_data, H, sampling_dict)

                # run experiment in a thread

                self.workers.append(worker)
                self.threads.append(QtCore.QThread())
                self.workers[-1].moveToThread(self.threads[-1])

                self.threads[-1].started.connect(self.workers[-1].run)
                self.workers[-1].finished.connect(self.threads[-1].quit)
                self.workers[-1].finished.connect(self.workers[-1].deleteLater)
                self.threads[-1].finished.connect(self.threads[-1].deleteLater)

                tab_mode = self.global_variables['tab_mode']
                if tab_mode == 'main':
                    report_progress = self.report_main_progress
                    save_experiment = self.save_main_experiment
                    self.max_iter_progress = len(uploaded_directories) * self.max_iter
                elif tab_mode == 'tuning':
                    report_progress = self.report_tuning_progress
                    save_experiment = self.save_tuning_experiment
                    self.max_iter_progress = len(uploaded_directories) * self.total_num_run
                else:
                    report_progress = self.report_comparison_progress
                    save_experiment = self.save_comparison_experiment
                    self.max_iter_progress = len(uploaded_directories) * self.max_iter

                self.workers[-1].progress.connect(report_progress)
                self.threads[-1].start()
                self.workers[-1].finished.connect(save_experiment)  # save results

                # Final resets
                self.startPushButton.setEnabled(False)
                self.threads[-1].finished.connect(self.reset_values)

            self.set_current_index_tab_last()
            print(f'max progress: {self.max_iter_progress}')

        except BaseException as err:
            msg = f"Unexpected {err=}, {type(err)=}"
            showCritical("Ocurrió un error inesperado al procesar el dato sísmico. Por favor, intente nuevamente o "
                         "utilice un dato diferente.", details=msg)
            self.experimentProgressBar.setValue(0)
            return

    def report_main_progress(self, data_name, iter, res_dict, sampling_dict, graphics):
        '''
        Report progress of the main experiment
        '''
        self.iters += 1
        self.experimentProgressBar.setValue(int((self.iters / self.max_iter_progress) * 100))

        # update figure
        err = res_dict['hist'][iter, 0]
        psnr = np.round(res_dict['hist'][iter, 1], 3)
        ssim = np.round(res_dict['hist'][iter, 2], 3)
        tv = np.round(res_dict['hist'][iter, 3], 3)

        iteration_list = self.state[self.global_variables['tab_mode']]['progress']['iteration'][data_name]
        error_list = self.state[self.global_variables['tab_mode']]['progress']['error'][data_name]
        psnr_list = self.state[self.global_variables['tab_mode']]['progress']['psnr'][data_name]
        ssim_list = self.state[self.global_variables['tab_mode']]['progress']['ssim'][data_name]
        tv_list = self.state[self.global_variables['tab_mode']]['progress']['tv'][data_name]

        iteration_list.append(iter)
        error_list.append(err)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        tv_list.append(tv)

        if iter % (self.max_iter // 10) == 0 or iter == self.max_iter:
            graphics['performance'].update_values(iteration_list, error_list, psnr_list, ssim_list, tv_list)
            graphics['performance'].update_figure()

            graphics['report'].update_report(
                dict(x_result=res_dict['result'], hist=res_dict['hist'], sampling=sampling_dict,
                     algorithm_name=self.algorithm_name))
            graphics['report'].update_figure()

    def save_main_experiment(self, data_name, res_dict, graphics):
        '''
        Save the main experiment
        '''
        performance_data = np.array(list(graphics['performance'].performance_data.items()), dtype=object)

        tab_mode = self.global_variables['tab_mode']
        data_mode = self.global_variables['data_mode']

        temp_saved = self.directories[tab_mode][data_mode]['temp_saved']

        os.makedirs(temp_saved, exist_ok=True)
        save_path = str(Path(temp_saved) / f'exp_{tab_mode}_{data_mode}_{data_name}.npz')

        self.directories[tab_mode][data_mode]['saved'] = save_path
        np.savez(save_path, x_result=res_dict['result'], hist=res_dict['hist'], sampling=res_dict['sampling_dict'],
                 algorithm_name=self.algorithm_name, performance_data=performance_data,
                 is_complete=True if data_mode == 'complete' else False)
        #aca
        if Path(self.current_file).suffix.lower() == '.sgy' or Path(self.current_file).suffix.lower() == '.segy':
            output_file = str(Path(temp_saved) / f'reconstructed_{data_name}.sgy')
            copyfile(self.current_file, output_file)
            with segyio.open(output_file, "r+") as src:
                for i in range(src.trace.length):
                    src.trace[i] = res_dict['result'][:, i]

        print("Results saved [Ok]")

    def report_tuning_progress(self, data_name, num_run, res_dict, params, graphics):
        '''
        Report progress of the tuning experiment
        '''
        self.iters += 1
        self.experimentProgressBar.setValue(int((self.iters / self.max_iter_progress) * 100))

        # update figure
        data = {key: [float(value)] for key, value in params.items()}
        data['error'] = [res_dict['hist'][-1, 0]]
        data['psnr'] = [np.round(res_dict['hist'][-1, 1], 3)]
        data['ssim'] = [np.round(res_dict['hist'][-1, 2], 3)]
        data['tv'] = [np.round(res_dict['hist'][-1, 3], 3)]

        if num_run == 1:
            self.tuning_data = pd.DataFrame(data)
        else:
            self.tuning_data = pd.concat([self.tuning_data, pd.DataFrame(data)])

        fixed_params = self.state[self.global_variables['tab_mode']]['progress']['fixed_params'][data_name]
        graphics['tuning'].update_tuning(self.algorithm_name.lower(), self.tuning_data, fixed_params,
                                         self.current_scale.lower())
        graphics['tuning'].update_figure()

    def save_tuning_experiment(self, data_name, graphics):
        '''
        Save the tuning experiment
        '''
        fixed_params = self.state[self.global_variables['tab_mode']]['progress']['fixed_params'][data_name]
        fixed_params = np.array(list(fixed_params.items()), dtype=object)
        tuning_data = np.array(list(graphics['tuning'].tuning_data.items()), dtype=object)

        tab_mode = self.global_variables['tab_mode']
        data_mode = self.global_variables['data_mode']
        temp_saved = self.directories[tab_mode][data_mode]['temp_saved']

        os.makedirs(temp_saved, exist_ok=True)
        save_path = str(Path(temp_saved) / f'exp_{tab_mode}_{data_mode}_{data_name}.npz')

        self.directories[tab_mode][data_mode]['saved'] = save_path
        np.savez(save_path,
                 algorithm=self.algorithm_name, tuning_data=tuning_data, fixed_params=fixed_params,
                 scale=self.current_scale, is_complete=True if data_mode == 'complete' else False)
        print("Results saved [Ok]")

    def report_comparison_progress(self, data_name, iter, outputs, sampling_dict, graphics):
        '''
        Report progress of the comparison experiment
        '''
        self.iters += 1
        self.experimentProgressBar.setValue(int((self.iters / self.max_iter_progress) * 100))

        # update figure
        errs, psnrs, ssims, tvs = [], [], [], []
        for output in outputs:
            errs.append(output['hist'][iter, 0])
            psnrs.append(np.round(output['hist'][iter, 1], 3))
            ssims.append(np.round(output['hist'][iter, 2], 3))
            tvs.append(np.round(output['hist'][iter, 3], 3))

        iteration_list = self.state[self.global_variables['tab_mode']]['progress']['iteration'][data_name]
        error_list = self.state[self.global_variables['tab_mode']]['progress']['errors'][data_name]
        psnr_list = self.state[self.global_variables['tab_mode']]['progress']['psnrs'][data_name]
        ssim_list = self.state[self.global_variables['tab_mode']]['progress']['ssims'][data_name]
        tv_list = self.state[self.global_variables['tab_mode']]['progress']['tvs'][data_name]

        iteration_list.append(iter)
        error_list.append(errs)
        psnr_list.append(psnrs)
        ssim_list.append(ssims)
        tv_list.append(tvs)

        if iter % (self.max_iter // 10) == 0 or iter == self.max_iter:
            graphics['performance'].update_values(iteration_list, error_list, psnr_list, ssim_list, tv_list)
            graphics['performance'].update_figure()

            x_results, hists = [], []
            for output in outputs:
                x_results.append(output['result'])
                hists.append(output['hist'])

            graphics['report'].update_report(
                dict(x_results=x_results, hist=hists, sampling=sampling_dict, algorithm_name=self.algorithm_name))
            graphics['report'].update_figure()

    def save_comparison_experiment(self, data_name, res_dict, graphics):
        '''
        Save the comparison experiment
        '''
        comparison_data = np.array(list(graphics['performance'].comparison_data.items()), dtype=object)

        tab_mode = self.global_variables['tab_mode']
        data_mode = self.global_variables['data_mode']
        temp_saved = self.directories[tab_mode][data_mode]['temp_saved']

        os.makedirs(temp_saved, exist_ok=True)
        save_path = str(Path(temp_saved) / f'exp_{tab_mode}_{data_mode}_{data_name}.npz')

        self.directories[tab_mode][data_mode]['saved'] = save_path
        np.savez(save_path,
                 x_results=res_dict['results'], hists=res_dict['hists'], sampling=res_dict['sampling_dict'],
                 comparison_data=comparison_data, is_complete=True if data_mode == 'complete' else False)
        print("Results saved [Ok]")

    def reset_values(self):
        '''
        Reset values
        '''
        if self.iters / self.max_iter_progress == 1.0:
            self.startPushButton.setEnabled(True)
            self.experimentProgressBar.setValue(0)
            self.workers = []
            self.threads = []

