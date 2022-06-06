from PyQt5 import QtCore, QtGui, QtWidgets, QtSvg


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
