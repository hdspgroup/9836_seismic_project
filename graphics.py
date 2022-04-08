from itertools import product

import numpy as np
from skimage.metrics import structural_similarity as ssim

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.ticker import MaxNLocator

from Algorithms.Function import PSNR
from gui.alerts import showCritical

class PerformanceGraphic(FigureCanvasQTAgg):
    def __init__(self):
        self.performance_data = dict(iteracion=[], error=[], psnr=[])
        self.figure = plt.figure()
        plt.subplots_adjust(left=0.07, right=0.93, bottom=0.08, top=0.92)
        super(PerformanceGraphic, self).__init__(self.figure)

    def update_values(self, iteracion, error, psnr):
        self.performance_data['iteracion'] = iteracion
        self.performance_data['error'] = error
        self.performance_data['psnr'] = psnr

    def update_figure(self):
        try:
            iteracion = self.performance_data['iteracion']
            error = self.performance_data['error']
            psnr = self.performance_data['psnr']

            self.figure.clear()
            self.figure.suptitle(f'Resultados del experimento')

            axes_1 = self.figure.add_subplot(111)
            axes_2 = axes_1.twinx()

            color = 'tab:red'
            axes_1.set_xlabel('iteraciones')
            axes_1.set_ylabel('error', color=color)
            axes_1.plot(iteracion, error, color=color)
            axes_1.tick_params(axis='y', labelcolor=color, length=5)
            axes_1.yaxis.set_major_locator(MaxNLocator(8))

            color = 'tab:blue'
            axes_2.set_ylabel('psnr', color=color)
            axes_2.plot(iteracion, psnr, color=color)
            axes_2.tick_params(axis='y', labelcolor=color, length=5)
            axes_2.yaxis.set_major_locator(MaxNLocator(8))

            self.draw()

        except BaseException as err:
            msg = f"Unexpected {err=}, {type(err)=}"
            showCritical("Ocurrió un error inesperado al procesar el dato sísmico. Por favor, intente nuevamente o "
                         "utilice un dato diferente.", details=msg)
            return


class ReportGraphic(FigureCanvasQTAgg):
    def __init__(self):
        self.report_data = None
        self.figure = plt.figure()
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90)
        super(ReportGraphic, self).__init__(self.figure)

    def update_report(self, report_data):
        self.report_data = report_data

    def update_figure(self):
        try:
            self.figure.clear()

            x_result = self.report_data['x_result']
            sampling = {item[0]: item[1] for item in self.report_data['sampling']}

            x = sampling['x_ori']
            y_rand = sampling['y_rand']
            pattern_rand = sampling['pattern_rand']

            temp = np.asarray(range(0, pattern_rand.shape[0]))
            pattern_rand_b2 = np.asarray(pattern_rand, dtype=bool) == 0
            H_elim = temp[pattern_rand_b2]

            case = str(self.report_data['algorithm_name'])
            self.figure.suptitle(f'Resultos del algoritmo {case}')
            axs = self.figure.subplots(2, 2)

            axs[0, 0].imshow(x, cmap='seismic', aspect='auto')
            axs[0, 0].set_title('Referencia')

            ytemp = y_rand.copy()
            ytemp[:, H_elim] = 0
            axs[1, 0].imshow(ytemp, cmap='seismic', aspect='auto')
            axs[1, 0].set_title('Medidas')

            # axs[1, 0].sharex(axs[0, 0])
            # metric = PSNR(x[:, H_elim], x_result[:, H_elim])
            metric = PSNR(x, x_result)
            metric_ssim = ssim(x[:, H_elim], x_result[:, H_elim], win_size=3)
            axs[0, 1].imshow(x_result, cmap='seismic', aspect='auto')
            axs[0, 1].set_title(f'Reconstruido - PSNR: {metric:0.2f} dB, SSIM:{metric_ssim:0.2f}')

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

class TuningGraphic(FigureCanvasQTAgg):
    def __init__(self):
        self.algorithm = None
        self.tuning_data = None
        self.fixed_params = None
        self.figure = plt.figure()
        # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90)
        super(TuningGraphic, self).__init__(self.figure)

    def update_tuning(self, algorithm, tuning_data, fixed_params):
        self.algorithm = algorithm
        self.tuning_data = tuning_data
        self.fixed_params = fixed_params

    def update_figure(self):
        try:
            self.figure.clear()

            params = list(self.tuning_data.keys())
            params.remove('error')
            params.remove('psnr')

            for key in self.fixed_params.keys():
                params.remove(key)

            if self.algorithm == 'gap':
                xlabel = f'{params[0]}'
            else:
                xlabel = f'{params[0]} | Valores fijos: '
                for key, value in self.fixed_params.items():
                    xlabel += f'{key}={np.round(value, 4)} '

            self.figure.suptitle(f'Algoritmo {self.algorithm}. Ajuste de parámetros.')
            axes_1 = self.figure.add_subplot(111)
            axes_2 = axes_1.twinx()

            color = 'tab:red'
            axes_1.set_xlabel(xlabel)
            axes_1.set_ylabel('error', color=color)
            graphic = axes_1.plot
            if len(self.tuning_data['error']) == 1:
                graphic = axes_1.scatter
            graphic(self.tuning_data[params[0]], self.tuning_data['error'], color=color)
            axes_1.tick_params(axis='y', labelcolor=color, length=5)
            axes_1.yaxis.set_major_locator(MaxNLocator(8))

            color = 'tab:blue'
            axes_2.set_ylabel('psnr', color=color)
            graphic = axes_2.plot
            if len(self.tuning_data['psnr']) == 1:
                graphic = axes_2.scatter
            graphic(self.tuning_data[params[0]], self.tuning_data['psnr'], color=color)
            axes_2.tick_params(axis='y', labelcolor=color, length=5)
            axes_2.yaxis.set_major_locator(MaxNLocator(8))

            self.draw()

        except BaseException as err:
            msg = f"Unexpected {err=}, {type(err)=}"
            showCritical("Ocurrió un error inesperado al procesar el dato sísmico. Por favor, intente nuevamente o "
                         "utilice un dato diferente.", details=msg)
            return
