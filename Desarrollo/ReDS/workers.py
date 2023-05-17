import numpy as np
from PyQt5 import QtCore


class Worker(QtCore.QObject):
    '''
    Worker for run experiment in a thread in main mode

    Parameters
    ----------
    name : str
        Name of the experiment
    function : function
        Function to run
    parameters : dict
        Parameters of the function
    maxiter : int
        Maximum number of iterations
    sampling_dict : dict
        Dictionary with the sampling information
    performance_graphic : pyqtgraph.PlotWidget
        Graphic to show the performance
    report_graphic : pyqtgraph.PlotWidget
        Graphic to show the report
    '''
    finished = QtCore.pyqtSignal(str, dict, dict)
    progress = QtCore.pyqtSignal(str, int, dict, np.ndarray, dict)

    def __init__(self, name, function, parameters, maxiter, sampling_dict, performance_graphic, report_graphic):
        super().__init__()
        self.data_name = name
        self.function = function
        self.parameters = parameters
        self.maxiter = maxiter
        self.sampling_dict = sampling_dict
        self.graphics = dict(performance=performance_graphic, report=report_graphic)

    def run(self):
        generator = self.function(**self.parameters)

        for itr, res_dict in generator:
            self.progress.emit(self.data_name, itr, res_dict, self.sampling_dict, self.graphics)
            if itr == self.maxiter:
                break
        # get last yield
        x_result, hist = next(generator)

        self.finished.emit(self.data_name, {'result': x_result, 'hist': hist, 'sampling_dict': self.sampling_dict},
                           self.graphics)


class TuningWorker(QtCore.QObject):
    '''
    Worker for run experiment in a thread in tuning mode

    Parameters
    ----------
    name : str
        Name of the experiment
    function : function
        Function to run
    parameters : dict
        Parameters of the function
    maxiter : int
        Maximum number of iterations
    tuning_graphic : pyqtgraph.PlotWidget
        Graphic to show the tuning
    '''
    finished = QtCore.pyqtSignal(str, dict)
    progress = QtCore.pyqtSignal(str, int, dict, dict, dict)

    def __init__(self, name, function, parameters, maxiter, tuning_graphic):
        super().__init__()
        self.data_name = name
        self.function = function
        self.parameters = parameters
        self.maxiter = maxiter
        self.graphic = dict(tuning=tuning_graphic)

    def run(self):
        for num_run, params in enumerate(self.parameters):
            generator = self.function(**params)
            for itr, res_dict in generator:

                if itr == self.maxiter:
                    params.pop('max_itr')
                    self.progress.emit(self.data_name, num_run + 1, res_dict, params, self.graphic)
                    break

        self.finished.emit(self.data_name, self.graphic)


class ComparisonWorker(QtCore.QObject):
    '''
    Worker for run experiment in a thread in comparison mode

    Parameters
    ----------
    name : str
        Name of the experiment
    function : function
        Function to run
    parameters : dict
        Parameters of the function
    maxiter : int
        Maximum number of iterations
    sampling_dict : dict
        Dictionary with the sampling information
    comp_performance_graphic : pyqtgraph.PlotWidget
        Graphic to show the performance
    comp_report_graphic : pyqtgraph.PlotWidget
        Graphic to show the report
    '''
    finished = QtCore.pyqtSignal(str, dict, dict)
    progress = QtCore.pyqtSignal(str, int, tuple, np.ndarray, dict)

    def __init__(self, name, functions, param_list, maxiter, sampling_dict, comp_performance_graphic,
                 comp_report_graphic):
        super().__init__()
        self.data_name = name
        self.functions = functions
        self.param_list = param_list
        self.maxiter = maxiter
        self.sampling_dict = sampling_dict
        self.graphics = dict(performance=comp_performance_graphic, report=comp_report_graphic)

    def run(self):
        generators = []
        for function, parameters in zip(self.functions, self.param_list):
            generators.append(function(**parameters))

        for (itr, res_dict_fista), (_, res_dict_gap), (_, res_dict_twist), (_, res_dict_admm) in zip(*generators):
            self.progress.emit(self.data_name, itr, (res_dict_fista, res_dict_gap, res_dict_twist, res_dict_admm),
                               self.sampling_dict, self.graphics)

            if itr == self.maxiter:
                break

        # get last yield
        x_results, hists = [], []
        for generator in generators:
            x_result, hist = next(generator)
            x_results.append(x_result)
            hists.append(hist)

        self.finished.emit(self.data_name, {'results': x_results, 'hists': hists, 'sampling_dict': self.sampling_dict},
                           self.graphics)


class TabWorker(QtCore.QObject):
    '''
    Worker for run tabs in a thread

    Parameters
    ----------

    '''
    finished = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal()

    def run(self):
        self.progress.emit()
        self.finished.emit()


# ------------------------------------------------- Shots -------------------------------------------------

class ShotWorker(QtCore.QObject):
    '''
    Worker for run experiment in a thread for shot reconstruction

    Parameters
    ----------
    name : str
        Name of the experiment
    function : function
        Function to run
    maxiter : int
        Maximum number of iterations
    sampling_dict : dict
        Dictionary with the sampling information
    performance_graphic : pyqtgraph.PlotWidget
        Graphic to show the performance
    report_graphic : pyqtgraph.PlotWidget
        Graphic to show the report
    '''
    finished = QtCore.pyqtSignal(str, dict, dict)
    progress = QtCore.pyqtSignal(str, int, dict, np.ndarray, dict)

    def __init__(self, name, function, maxiter, sampling_dict, performance_graphic, report_graphic):
        super().__init__()
        self.data_name = name
        self.function = function
        self.maxiter = maxiter
        self.sampling_dict = sampling_dict
        self.graphics = dict(performance=performance_graphic, report=report_graphic)

    def run(self):
        generator = self.function()
        for itr, res_dict in generator:
            self.progress.emit(self.data_name, itr, res_dict, self.sampling_dict, self.graphics)

            if itr == self.maxiter:
                x_result, hist = res_dict['result'], res_dict['hist']
                break

        # # get last yield
        # x_result, hist = next(generator)

        self.finished.emit(self.data_name, {'result': x_result, 'hist': hist, 'sampling_dict': self.sampling_dict},
                           self.graphics)
