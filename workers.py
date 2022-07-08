import numpy as np
from PyQt5 import QtCore


class Worker(QtCore.QObject):
    finished = QtCore.pyqtSignal(dict)
    progress = QtCore.pyqtSignal(int, dict)

    def __init__(self, function, parameters, maxiter):
        super().__init__()
        self.function = function
        self.parameters = parameters
        self.maxiter = maxiter

    def run(self):
        generator = self.function(**self.parameters)
        for itr, res_dict in generator:
            self.progress.emit(itr, res_dict)

            if itr == self.maxiter:
                break

        # get last yield
        x_result, hist = next(generator)

        self.finished.emit({'result': x_result, 'hist': hist})


class TuningWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(int, dict, dict)

    def __init__(self, function, parameters, maxiter):
        super().__init__()
        self.function = function
        self.parameters = parameters
        self.maxiter = maxiter

    def run(self):
        for num_run, params in enumerate(self.parameters):
            generator = self.function(**params)
            for itr, res_dict in generator:

                if itr == self.maxiter:
                    params.pop('max_itr')
                    self.progress.emit(num_run + 1, res_dict, params)
                    break

        self.finished.emit()


class ComparisonWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(dict)
    progress = QtCore.pyqtSignal(int, tuple)

    def __init__(self, functions, param_list, maxiter):
        super().__init__()
        self.functions = functions
        self.param_list = param_list
        self.maxiter = maxiter

    def run(self):
        generators = []
        for function, parameters in zip(self.functions, self.param_list):
            generators.append(function(**parameters))

        for (itr, res_dict_fista), (_, res_dict_gap), (_, res_dict_twist), (_, res_dict_admm) in zip(*generators):
            self.progress.emit(itr, (res_dict_fista, res_dict_gap, res_dict_twist, res_dict_admm))

            if itr == self.maxiter:
                break

        # get last yield
        x_results, hists = [], []
        for generator in generators:
            x_result, hist = next(generator)
            x_results.append(x_result)
            hists.append(hist)

        self.finished.emit({'results': x_results, 'hists': hists})


# class ComparisonWorker(QtCore.QObject):
#     finished = QtCore.pyqtSignal(dict)
#     progress = QtCore.pyqtSignal(str, int, dict)
#
#     def __init__(self, algorithm, function, parameters, maxiter):
#         super().__init__()
#         self.algorithm = algorithm
#         self.function = function
#         self.parameters = parameters
#         self.maxiter = maxiter
#
#     def run(self):
#         generator = self.function(**self.parameters)
#         for itr, res_dict in generator:
#             self.progress.emit(self.algorithm, itr, res_dict)
#
#             if itr == self.maxiter:
#                 break
#
#         # get last yield
#         x_result, hist = next(generator)
#
#         self.finished.emit({'algorithm': self.algorithm, 'result': x_result, 'hist': hist})


class ComparisonWorkers:
    def __init__(self, algorithms, param_list, max_iter):
        self.fista_worker = ComparisonWorker('fista', algorithms[0], param_list[0], max_iter)
        self.gap_worker = ComparisonWorker('gap', algorithms[1], param_list[1], max_iter)
        self.twist_worker = ComparisonWorker('twist', algorithms[2], param_list[2], max_iter)
        self.admm_worker = ComparisonWorker('admm', algorithms[3], param_list[3], max_iter)

        self.workers = [self.fista_worker, self.gap_worker, self.twist_worker, self.admm_worker]
