from itertools import product

import numpy as np
from skimage.metrics import structural_similarity as ssim

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.ticker import MaxNLocator

from Algorithms.Function import PSNR
from gui.scripts.alerts import showCritical


class PerformanceGraphic(FigureCanvasQTAgg):
    def __init__(self):
        self.performance_data = dict(iteracion=[], error=[], psnr=[], ssim=[])
        self.figure = plt.figure()
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.08, top=0.92)
        super(PerformanceGraphic, self).__init__(self.figure)

    def update_values(self, iteracion, error, psnr, ssim):
        self.performance_data['iteracion'] = iteracion
        self.performance_data['error'] = error
        self.performance_data['psnr'] = psnr
        self.performance_data['ssim'] = ssim

    def update_figure(self):
        try:
            iteracion = self.performance_data['iteracion']
            error = self.performance_data['error']
            psnr = self.performance_data['psnr']
            ssim = self.performance_data['ssim']

            self.figure.clear()
            self.figure.suptitle(f'Resultados del experimento')

            axes_1 = self.figure.add_subplot(111)
            axes_2 = axes_1.twinx()

            color = 'tab:red'
            axes_1.set_xlabel('iteraciones')
            axes_1.set_ylabel('ssim', color=color)
            axes_1.plot(iteracion, ssim, color=color)
            axes_1.tick_params(axis='y', labelcolor=color, length=5)
            # axes_1.yaxis.set_major_locator(MaxNLocator(8))
            axes_1.grid(axis='both', linestyle='--')

            axes_1.set_yticks(np.linspace(axes_1.get_ybound()[0], axes_1.get_ybound()[1], 8))

            color = 'tab:blue'
            axes_2.set_ylabel('psnr', color=color)
            axes_2.plot(iteracion, psnr, color=color)
            axes_2.tick_params(axis='y', labelcolor=color, length=5)
            # axes_2.yaxis.set_major_locator(MaxNLocator(8))
            axes_2.grid(axis='both', linestyle='--')

            axes_2.set_yticks(np.linspace(axes_2.get_ybound()[0], axes_2.get_ybound()[1], 8))

            self.draw()

        except BaseException as err:
            msg = f"Unexpected {err=}, {type(err)=}"
            showCritical("Ocurrió un error inesperado al procesar el dato sísmico. Por favor, intente nuevamente o "
                         "utilice un dato diferente.", details=msg)
            return


class ReconstructionGraphic(FigureCanvasQTAgg):
    def __init__(self):
        self.report_data = None
        self.figure = plt.figure()
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90)
        super(ReconstructionGraphic, self).__init__(self.figure)

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
            self.figure.suptitle(f'Resultados del algoritmo {case}')
            axs = self.figure.subplots(2, 2)

            axs[0, 0].imshow(x, cmap='gray', aspect='auto')
            axs[0, 0].set_title('Referencia')

            ytemp = y_rand.copy()
            condition = H_elim.size > 0
            if condition:
                ytemp[:, H_elim] = None
            axs[1, 0].imshow(ytemp, cmap='gray', aspect='auto')
            axs[1, 0].set_title('Medidas')

            # axs[1, 0].sharex(axs[0, 0])
            # metric = PSNR(x[:, H_elim], x_result[:, H_elim])
            aux_x = x[:, H_elim] if condition else x
            aux_x_result = x_result[:, H_elim] if condition else x_result
            metric = PSNR(aux_x, aux_x_result)
            metric_ssim = ssim(aux_x, aux_x_result)
            axs[0, 1].imshow(x_result, cmap='gray', aspect='auto')
            axs[0, 1].set_title(f'Reconstruido - PSNR: {metric:0.2f} dB, SSIM:{metric_ssim:0.2f}')

            index = 5
            aux_H_elim = index if condition else H_elim[index]
            axs[1, 1].plot(x[:, aux_H_elim], 'r', label='Referencia')
            axs[1, 1].plot(x_result[:, aux_H_elim], 'b', label='Recuperado')
            axs[1, 1].legend(loc='best')
            axs[1, 1].set_title('Traza ' + str("{:.0f}".format(aux_H_elim)))
            axs[1, 1].grid(axis='both', linestyle='--')

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
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.90)
        super(TuningGraphic, self).__init__(self.figure)

    def update_tuning(self, algorithm, tuning_data, fixed_params, current_scale):
        self.algorithm = algorithm
        self.tuning_data = tuning_data
        self.fixed_params = fixed_params
        self.current_scale = current_scale

    def update_figure(self):
        try:
            self.figure.clear()

            params = list(self.tuning_data.keys())
            params.remove('error')
            params.remove('psnr')
            params.remove('ssim')

            params = [param.replace('lmb', 'lambda') for param in params]  # replace lmb by lambda if it exists

            if self.algorithm == 'gap':  # build x label for the graphics
                xlabel = f'$\\{params[0]}$'
            else:
                for key in self.fixed_params.keys():
                    params.remove(key)

                hspace = r'\,\,\,'
                xlabel = f'$\\{params[0]}$ | Valores fijos: $'
                for key, value in self.fixed_params.items():
                    xlabel += f'\\{key}={np.round(value, 4)} {hspace}'
                xlabel += '$'

            self.figure.suptitle(f'Algoritmo {self.algorithm} - Ajuste de parámetros - Escala {self.current_scale}')
            axes_1 = self.figure.add_subplot(111)
            axes_2 = axes_1.twinx()

            color = 'tab:red'
            axes_1.set_xlabel(xlabel)
            axes_1.set_ylabel('ssim', color=color)
            graphic = axes_1.plot
            if len(self.tuning_data['ssim']) == 1:
                graphic = axes_1.scatter
            graphic(self.tuning_data['lmb' if params[0] == 'lambda' else params[0]], self.tuning_data['ssim'],
                    '--o' if len(self.tuning_data) > 1 else None, color=color)
            axes_1.tick_params(axis='y', labelcolor=color, length=5)
            axes_1.yaxis.set_major_locator(MaxNLocator(8))
            axes_1.set_xscale('linear' if self.current_scale == 'lineal' else 'log')
            axes_1.grid(axis='both', which="both", linestyle='--')

            axes_1.set_yticks(np.linspace(axes_1.get_ybound()[0], axes_1.get_ybound()[1], 8))

            color = 'tab:blue'
            axes_2.set_ylabel('psnr', color=color)
            graphic = axes_2.plot
            if len(self.tuning_data['psnr']) == 1:
                graphic = axes_2.scatter
            graphic(self.tuning_data['lmb' if params[0] == 'lambda' else params[0]], self.tuning_data['psnr'],
                    '--o' if len(self.tuning_data) > 1 else None, color=color)
            axes_2.tick_params(axis='y', labelcolor=color, length=5)
            axes_2.yaxis.set_major_locator(MaxNLocator(8))
            axes_2.set_xscale('linear' if self.current_scale == 'lineal' else 'log')
            axes_2.grid(axis='both', which="both", linestyle='--')

            axes_2.set_yticks(np.linspace(axes_2.get_ybound()[0], axes_2.get_ybound()[1], 8))

            self.draw()

        except BaseException as err:
            msg = f"Unexpected {err=}, {type(err)=}"
            showCritical("Ocurrió un error inesperado al procesar el dato sísmico. Por favor, intente nuevamente o "
                         "utilice un dato diferente.", details=msg)
            return


class ComparisonPerformanceGraphic(FigureCanvasQTAgg):
    def __init__(self):
        self.algorithm_names = ['FISTA', 'GAP', 'TwIST', 'ADMM']
        self.performance_data = dict(iteracion=[], errors=[], psnrs=[], ssims=[])
        self.figure = plt.figure()
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.08, top=0.92)
        super(ComparisonPerformanceGraphic, self).__init__(self.figure)

    def update_values(self, iteracion, errors, psnrs, ssims):
        self.performance_data['iteracion'] = iteracion
        self.performance_data['errors'] = errors
        self.performance_data['psnrs'] = psnrs
        self.performance_data['ssims'] = ssims

    def update_figure(self):
        try:
            self.figure.clear()

            iteracion = np.array(self.performance_data['iteracion'])
            errors = np.array(self.performance_data['errors'])
            psnrs = np.array(self.performance_data['psnrs'])
            ssims = np.array(self.performance_data['ssims'])

            self.figure.suptitle(f'Resultados del experimento')
            axs = self.figure.subplots(2, 2)

            for idx, (i, j) in enumerate(product([0, 1], [0, 1])):
                axes_1 = axs[i, j]
                axes_2 = axes_1.twinx()

                color = 'tab:red'
                axes_1.set_xlabel('iteraciones')
                axes_1.set_ylabel('ssim', color=color)
                axes_1.plot(iteracion, ssims[:, idx], color=color)
                axes_1.set_title(self.algorithm_names[idx])
                axes_1.tick_params(axis='y', labelcolor=color, length=5)
                axes_1.grid(axis='both', linestyle='--')

                axes_1.set_yticks(np.linspace(axes_1.get_ybound()[0], axes_1.get_ybound()[1], 8))

                color = 'tab:blue'
                axes_2.set_ylabel('psnr', color=color)
                axes_2.plot(iteracion, psnrs[:, idx], color=color)
                axes_2.tick_params(axis='y', labelcolor=color, length=5)
                axes_2.grid(axis='both', linestyle='--')

                axes_2.set_yticks(np.linspace(axes_2.get_ybound()[0], axes_2.get_ybound()[1], 8))

            self.draw()

        except BaseException as err:
            msg = f"Unexpected {err=}, {type(err)=}"
            showCritical("Ocurrió un error inesperado al procesar el dato sísmico. Por favor, intente nuevamente o "
                         "utilice un dato diferente.", details=msg)
            return


class ComparisonReconstructionGraphic(FigureCanvasQTAgg):
    def __init__(self):
        self.algorithm_names = ['FISTA', 'GAP', 'TwIST', 'ADMM']
        self.comparison_data = None
        self.figure = plt.figure()
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90)
        super(ComparisonReconstructionGraphic, self).__init__(self.figure)

    def update_report(self, comparison_data):
        self.comparison_data = comparison_data

    def update_figure(self):
        try:
            self.figure.clear()

            sampling = {item[0]: item[1] for item in self.comparison_data['sampling']}
            x = sampling['x_ori']
            y_rand = sampling['y_rand']
            pattern_rand = sampling['pattern_rand']

            temp = np.asarray(range(0, pattern_rand.shape[0]))
            pattern_rand_b2 = np.asarray(pattern_rand, dtype=bool) == 0
            H_elim = temp[pattern_rand_b2]

            # =#=#=#=#=#=#=#

            self.figure.suptitle(f'Comparaciones de algoritmos')
            axs = self.figure.subplots(2, 3)

            axs[0, 0].imshow(x, cmap='gray', aspect='auto')
            axs[0, 0].set_title('Referencia')

            ytemp = y_rand.copy()
            condition = H_elim.size > 0
            if condition:
                ytemp[:, H_elim] = None
            axs[1, 0].imshow(ytemp, cmap='gray', aspect='auto')
            axs[1, 0].set_title('Medidas')

            indices = [(i, j) for i, j in product([0, 1], [1, 2])]
            for idxs, algorithm_name, x_result in zip(indices, self.algorithm_names, self.comparison_data['x_results']):
                i, j = idxs
                aux_x = x[:, H_elim] if condition else x
                aux_x_result = x_result[:, H_elim] if condition else x_result
                metric = PSNR(aux_x, aux_x_result)
                metric_ssim = ssim(aux_x, aux_x_result)
                axs[i, j].imshow(x_result, cmap='gray', aspect='auto')
                axs[i, j].set_title(f'{algorithm_name} - PSNR: {metric:0.2f} dB, SSIM:{metric_ssim:0.2f}')

            self.draw()

        except BaseException as err:
            msg = f"Unexpected {err=}, {type(err)=}"
            showCritical(
                "Ocurrió un error inesperado al procesar el dato sísmico. Por favor, intente nuevamente o "
                "utilice un dato diferente.", details=msg)
            return
