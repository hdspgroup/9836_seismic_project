import numpy as np
from PyQt5 import QtCore


class Worker(QtCore.QObject):
    finished = QtCore.pyqtSignal(str, dict, dict)
    progress = QtCore.pyqtSignal(str, int, dict, np.ndarray, dict)

    def __init__(self, name, function, parameters, maxiter, sampling_dict, performance_graphic, report_graphic):
        super().__init__()
        self.name = name
        self.function = function
        self.parameters = parameters
        self.maxiter = maxiter
        self.sampling_dict = sampling_dict
        self.graphics = dict(performance=performance_graphic, report=report_graphic)

    def run(self):
        generator = self.function(**self.parameters)
        for itr, res_dict in generator:
            self.progress.emit(self.name, itr, res_dict, self.sampling_dict, self.graphics)

            if itr == self.maxiter:
                break

        # get last yield
        x_result, hist = next(generator)

        self.finished.emit(self.name, {'result': x_result, 'hist': hist, 'sampling_dict': self.sampling_dict},
                           self.graphics)


class TuningWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(str, dict)
    progress = QtCore.pyqtSignal(str, int, dict, dict, dict)

    def __init__(self, name, function, parameters, maxiter, tuning_graphic):
        super().__init__()
        self.name = name
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
                    self.progress.emit(self.name, num_run + 1, res_dict, params, self.graphic)
                    break

        self.finished.emit(self.name, self.graphic)


class ComparisonWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(dict)
    progress = QtCore.pyqtSignal(int, tuple, np.ndarray)

    def __init__(self, functions, param_list, maxiter, sampling_dict):
        super().__init__()
        self.functions = functions
        self.param_list = param_list
        self.maxiter = maxiter
        self.sampling_dict = sampling_dict

    def run(self):
        generators = []
        for function, parameters in zip(self.functions, self.param_list):
            generators.append(function(**parameters))

        for (itr, res_dict_fista), (_, res_dict_gap), (_, res_dict_twist), (_, res_dict_admm) in zip(*generators):
            self.progress.emit(itr, (res_dict_fista, res_dict_gap, res_dict_twist, res_dict_admm), self.sampling_dict)

            if itr == self.maxiter:
                break

        # get last yield
        x_results, hists = [], []
        for generator in generators:
            x_result, hist = next(generator)
            x_results.append(x_result)
            hists.append(hist)

        self.finished.emit({'results': x_results, 'hists': hists, 'sampling_dict': self.sampling_dict})
