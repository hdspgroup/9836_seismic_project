from itertools import product

import numpy as np
from skimage.metrics import structural_similarity as ssim

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.ticker import MaxNLocator

from Algorithms.Function import PSNR
from Algorithms.fk_domain import fk
from Algorithms.tv_norm import tv_norm
from gui.scripts.alerts import showCritical


# custom toolbar with lorem ipsum text
class CustomToolbar(NavigationToolbar2QT):
    def __init__(self, canvas_, parent_):
        self.toolitems = (
            ('Home', 'Volver a la vista original', 'home', 'home'),
            ('Back', 'Volver a la vista previa', 'back', 'back'),
            ('Forward', 'Volver a la siguiente vista', 'forward', 'forward'),
            (None, None, None, None),
            ('Pan', 'El botón izquierdo panea, el botón derecho hace zoom\n'
                    'x/y fija el eje, CTRL fija el aspecto', 'move', 'pan'),
            ('Zoom', 'Zoom a rectángulo\nx/y fija el eje', 'zoom_to_rect', 'zoom'),
            ('Subplots', 'Configurar subgráficas', 'subplots', 'configure_subplots'),
            ("Customize", "Editar ejes, curvas y parámetros de la gráfica", "qt4_editor_options", "edit_parameters"),
            (None, None, None, None),
            ('Save', 'Guardar la figura', 'filesave', 'save_figure'),
        )
        NavigationToolbar2QT.__init__(self, canvas_, parent_)


class PerformanceGraphic(FigureCanvasQTAgg):
    def __init__(self, is_complete=True):
        self.is_complete = is_complete
        self.performance_data = dict(iteracion=[], error=[], psnr=[], ssim=[], tv=[])
        self.figure = plt.figure()
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.08, top=0.92)
        super(PerformanceGraphic, self).__init__(self.figure)

    def update_values(self, iteracion, error, psnr, ssim, tv):
        self.performance_data['iteracion'] = iteracion
        self.performance_data['error'] = error
        self.performance_data['psnr'] = psnr
        self.performance_data['ssim'] = ssim
        self.performance_data['tv'] = tv

    def update_figure(self):
        try:
            iteracion = self.performance_data['iteracion']
            error = self.performance_data['error']
            psnr = self.performance_data['psnr']
            ssim = self.performance_data['ssim']
            tv = self.performance_data['tv']

            self.figure.clear()
            self.figure.suptitle(f'Resultados del experimento')

            axes_1 = self.figure.add_subplot(111)
            axes_2 = axes_1.twinx()

            color = 'tab:red'
            axes_1.set_xlabel('iteraciones')
            axes_1.plot(iteracion, ssim if self.is_complete else error, color=color,
                        label='SSIM' if self.is_complete else 'Error residual')
            axes_1.tick_params(axis='y', labelcolor=color, length=5)
            axes_1.grid(axis='both', linestyle='--')

            axes_1.set_yticks(np.linspace(axes_1.get_ybound()[0], axes_1.get_ybound()[1], 8))

            color = 'tab:blue'
            axes_2.plot(iteracion, psnr if self.is_complete else tv, '--', color=color)
            axes_1.plot(np.nan, '--', color=color, label='PSNR' if self.is_complete else 'Norma TV')
            axes_2.tick_params(axis='y', labelcolor=color, length=5)
            axes_2.grid(axis='both', linestyle='--')

            axes_2.set_yticks(np.linspace(axes_2.get_ybound()[0], axes_2.get_ybound()[1], 8))

            if self.is_complete:
                if np.abs(psnr[-1]) == np.inf:
                    # print a text in the bottom part of axes_2 that indicates that the psnr is infinite
                    text_kwargs = dict(ha='center', va='center', fontsize=16, color=color)
                    axes_2.text(0.5, 0.1, f'PSNR is {psnr[-1]}, it will be not plotted', horizontalalignment='center',
                                verticalalignment='center', transform=axes_2.transAxes, **text_kwargs)

            axes_1.legend(loc='best')
            self.draw()

        except BaseException as err:
            msg = f"Unexpected {err=}, {type(err)=}"
            showCritical("Ocurrió un error inesperado al procesar el dato sísmico. Por favor, intente nuevamente o "
                         "utilice un dato diferente.", details=msg)
            return


class ReconstructionGraphic(FigureCanvasQTAgg):
    def __init__(self, is_complete=True):
        self.is_complete = is_complete
        self.report_data = None
        self.figure = plt.figure()
        left, right, bottom, top = [0.05, 0.95, 0.05, 0.90] if is_complete else [0.05, 0.98, 0.05, 0.85]
        plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
        super(ReconstructionGraphic, self).__init__(self.figure)

    def update_report(self, report_data):
        self.report_data = report_data

    def update_figure(self):
        try:
            self.figure.clear()

            x_result = self.report_data['x_result']
            sampling = {item[0]: item[1] for item in self.report_data['sampling']}

            if self.is_complete:
                x = sampling['x_ori']
                y_rand = sampling['y_rand']
                pattern_rand = sampling['pattern_rand']
            else:
                x = sampling['x_ori']
                y_rand = x
                pattern_rand = np.double(sampling['H'])

            temp = np.asarray(range(0, pattern_rand.shape[0]))
            pattern_rand_b2 = np.asarray(pattern_rand, dtype=bool) == 0
            H_elim = temp[pattern_rand_b2]

            case = str(self.report_data['algorithm_name'])
            self.figure.suptitle(f'Resultados del algoritmo {case}')

            if self.is_complete:
                axs = self.figure.subplots(2, 2)

                axs[0, 0].imshow(x, cmap='gray', aspect='auto')
                axs[0, 0].set_title('Referencia')

                ytemp = y_rand.copy()
                condition = H_elim.size > 0
                if condition:
                    ytemp[:, H_elim] = 1
                axs[1, 0].imshow(ytemp, cmap='gray', aspect='auto')
                axs[1, 0].set_title('Medidas')

                aux_x = x[:, H_elim] if condition else x
                aux_x_result = x_result[:, H_elim] if condition else x_result
                metric = PSNR(aux_x, aux_x_result)
                metric_ssim = ssim(aux_x, aux_x_result)
                axs[0, 1].imshow(x_result, cmap='gray', aspect='auto')
                axs[0, 1].set_title(f'Reconstruido - PSNR: {metric:0.2f} dB, SSIM:{metric_ssim:0.2f}')

                index = 5
                # aux_H_elim = index if condition else H_elim[index]
                aux_H_elim = H_elim[index]
                axs[1, 1].plot(x[:, aux_H_elim], 'r', label='Referencia')
                axs[1, 1].plot(x_result[:, aux_H_elim], 'b', label='Recuperado')
                axs[1, 1].legend(loc='best')
                axs[1, 1].set_title('Traza ' + str("{:.0f}".format(aux_H_elim)))
                axs[1, 1].grid(axis='both', linestyle='--')

            else:
                axs = self.figure.subplots(2, 3)

                metric = tv_norm(x)
                axs[0, 0].imshow(x, cmap='gray', aspect='auto')
                axs[0, 0].set_title(f'Reference \n TV-norm: {metric:0.2f}')

                metric = tv_norm(x_result)
                axs[0, 1].imshow(x_result, cmap='gray', aspect='auto')
                axs[0, 1].set_title(f'Reconstructed \n TV norm: {metric:0.2f}')

                index = 5
                axs[0, 2].plot(x_result[:, H_elim[index]], 'b', label='Recovered')
                axs[0, 2].legend(loc='best')
                axs[0, 2].set_title('Trace ' + str("{:.0f}".format(H_elim[index])))
                axs[0, 2].grid(axis='both', linestyle='--')

                # fk domain

                # calculate FK domain
                FK, f, kx = fk(x, dt=0.568, dx=5)
                axs[1, 0].imshow(FK[:200], aspect='auto', cmap='jet', extent=[kx.min(), kx.max(), 65, 0])
                axs[1, 0].set_title('FK reference')

                FK, f, kx = fk(x_result, dt=0.568, dx=5)
                axs[1, 1].imshow(FK[:200], aspect='auto', cmap='jet', extent=[kx.min(), kx.max(), 65, 0])
                axs[1, 1].set_title('FK reconstruction')

                index = -1
                axs[1, 2].plot(x_result[:, H_elim[index]], 'b', label='Recovered')
                axs[1, 2].legend(loc='best')
                axs[1, 2].set_title('Trace ' + str("{:.0f}".format(H_elim[index])))
                axs[1, 2].grid(axis='both', linestyle='--')

            self.draw()

        except BaseException as err:
            msg = f"Unexpected {err=}, {type(err)=}"
            showCritical("Ocurrió un error inesperado al procesar el dato sísmico. Por favor, intente nuevamente o "
                         "utilice un dato diferente.", details=msg)
            return


class TuningGraphic(FigureCanvasQTAgg):
    def __init__(self, is_complete):
        self.is_complete = is_complete
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
            params.remove('tv')

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
            graphic = axes_1.plot
            if len(self.tuning_data['ssim']) == 1:
                graphic = axes_1.scatter
            graphic(self.tuning_data['lmb' if params[0] == 'lambda' else params[0]],
                    self.tuning_data['ssim'] if self.is_complete else self.tuning_data['error'],
                    '-o' if len(self.tuning_data) > 1 else None, color=color,
                    label='SSIM' if self.is_complete else 'Error residual')
            axes_1.tick_params(axis='y', labelcolor=color, length=5)
            axes_1.yaxis.set_major_locator(MaxNLocator(8))
            axes_1.set_xscale('linear' if self.current_scale == 'lineal' else 'log')
            axes_1.grid(axis='both', which="both", linestyle='--')

            axes_1.set_yticks(np.linspace(axes_1.get_ybound()[0], axes_1.get_ybound()[1], 8))

            color = 'tab:blue'
            graphic = axes_2.plot
            if len(self.tuning_data['psnr']) == 1:
                graphic = axes_2.scatter
            graphic(self.tuning_data['lmb' if params[0] == 'lambda' else params[0]],
                    self.tuning_data['psnr'] if self.is_complete else self.tuning_data['tv'],
                    '--o' if len(self.tuning_data) > 1 else None, color=color)
            axes_1.plot(np.nan, '--o', color=color, label='PSNR' if self.is_complete else 'Norma TV')
            axes_2.tick_params(axis='y', labelcolor=color, length=5)
            axes_2.yaxis.set_major_locator(MaxNLocator(8))
            axes_2.set_xscale('linear' if self.current_scale == 'lineal' else 'log')
            axes_2.grid(axis='both', which="both", linestyle='--')

            axes_2.set_yticks(np.linspace(axes_2.get_ybound()[0], axes_2.get_ybound()[1], 8))

            axes_1.legend(loc='best')
            self.draw()

        except BaseException as err:
            msg = f"Unexpected {err=}, {type(err)=}"
            showCritical("Ocurrió un error inesperado al procesar el dato sísmico. Por favor, intente nuevamente o "
                         "utilice un dato diferente.", details=msg)
            return


class ComparisonPerformanceGraphic(FigureCanvasQTAgg):
    def __init__(self, is_complete):
        self.is_complete = is_complete
        self.algorithm_names = ['FISTA', 'GAP', 'TwIST', 'ADMM']
        self.comparison_data = dict(iteracion=[], errors=[], psnrs=[], ssims=[])
        self.figure = plt.figure()
        plt.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.92, wspace=0.5, hspace=0.3)
        super(ComparisonPerformanceGraphic, self).__init__(self.figure)

    def update_values(self, iteracion, errors, psnrs, ssims, tvs):
        self.comparison_data['iteracion'] = iteracion
        self.comparison_data['errors'] = errors
        self.comparison_data['psnrs'] = psnrs
        self.comparison_data['ssims'] = ssims
        self.comparison_data['tvs'] = tvs

    def update_figure(self):
        try:
            self.figure.clear()

            iteracion = np.array(self.comparison_data['iteracion'])
            errors = np.array(self.comparison_data['errors'])
            psnrs = np.array(self.comparison_data['psnrs'])
            ssims = np.array(self.comparison_data['ssims'])
            tvs = np.array(self.comparison_data['tvs'])

            self.figure.suptitle(f'Resultados del experimento')
            axs = self.figure.subplots(2, 2)

            for idx, (i, j) in enumerate(product([0, 1], [0, 1])):
                axes_1 = axs[i, j]
                axes_2 = axes_1.twinx()

                color = 'tab:red'
                axes_1.set_xlabel('iteraciones')
                axes_1.plot(iteracion, ssims[:, idx] if self.is_complete else errors[:, idx], color=color,
                            label='SSIM' if self.is_complete else 'Error residual')
                axes_1.set_title(self.algorithm_names[idx])
                axes_1.tick_params(axis='y', labelcolor=color, length=5)
                axes_1.grid(axis='both', linestyle='--')

                axes_1.set_yticks(np.linspace(axes_1.get_ybound()[0], axes_1.get_ybound()[1], 8))

                color = 'tab:blue'
                axes_2.plot(iteracion, psnrs[:, idx] if self.is_complete else tvs[:, idx], color=color)
                axes_1.legend(loc='best')
                axes_1.plot(np.nan, color=color, label='PSNR' if self.is_complete else 'Norma TV')
                axes_2.tick_params(axis='y', labelcolor=color, length=5)
                axes_2.grid(axis='both', linestyle='--')

                axes_2.set_yticks(np.linspace(axes_2.get_ybound()[0], axes_2.get_ybound()[1], 8))

                axes_1.legend(loc='best')

            self.draw()

        except BaseException as err:
            msg = f"Unexpected {err=}, {type(err)=}"
            showCritical("Ocurrió un error inesperado al procesar el dato sísmico. Por favor, intente nuevamente o "
                         "utilice un dato diferente.", details=msg)
            return


class ComparisonReconstructionGraphic(FigureCanvasQTAgg):
    def __init__(self, is_complete):
        self.is_complete = is_complete
        self.algorithm_names = ['FISTA', 'GAP', 'TwIST', 'ADMM']
        self.comparison_data = None
        self.figure = plt.figure()
        left, right, bottom, top = [0.07, 0.93, 0.05, 0.90] if is_complete else [0.05, 0.98, 0.05, 0.85]
        plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=0.3, hspace=0.3)
        super(ComparisonReconstructionGraphic, self).__init__(self.figure)

    def update_report(self, comparison_data):
        self.comparison_data = comparison_data

    def update_figure(self):
        try:
            self.figure.clear()
            sampling = {item[0]: item[1] for item in self.comparison_data['sampling']}

            if self.is_complete:
                x = sampling['x_ori']
                y_rand = sampling['y_rand']
                pattern_rand = sampling['pattern_rand']
            else:
                x = sampling['x_ori']
                y_rand = x
                pattern_rand = np.double(sampling['H'])

            temp = np.asarray(range(0, pattern_rand.shape[0]))
            pattern_rand_b2 = np.asarray(pattern_rand, dtype=bool) == 0
            H_elim = temp[pattern_rand_b2]

            # =#=#=#=#=#=#=#

            self.figure.suptitle(f'Comparaciones de algoritmos')

            if self.is_complete:
                axs = self.figure.subplots(2, 3)

                axs[0, 0].imshow(x, cmap='gray', aspect='auto')
                axs[0, 0].set_title('Referencia')

                ytemp = y_rand.copy()
                condition = H_elim.size > 0
                if condition:
                    ytemp[:, H_elim] = 1
                axs[0, 1].imshow(ytemp, cmap='gray', aspect='auto')
                axs[0, 1].set_title('Medidas')

                indices = [(i, j) for i, j in product([0, 1], [1, 2])]
                for idxs, algorithm_name, x_result in zip(indices, self.algorithm_names,
                                                          self.comparison_data['x_results']):
                    i, j = idxs
                    aux_x = x[:, H_elim] if condition else x
                    aux_x_result = x_result[:, H_elim] if condition else x_result
                    metric = PSNR(aux_x, aux_x_result)
                    metric_ssim = ssim(aux_x, aux_x_result)
                    axs[i, j].imshow(x_result, cmap='gray', aspect='auto')
                    axs[i, j].set_title(f'{algorithm_name} - PSNR: {metric:0.2f} dB, SSIM:{metric_ssim:0.2f}')

            else:
                axs = self.figure.subplots(3, 4)

                metric = tv_norm(x)
                axs[0, 0].imshow(x, cmap='gray', aspect='auto')
                axs[0, 0].set_title(f'Reference - TV-norm: {metric:0.2f}')

                x_result_fista = self.comparison_data['x_results'][0]
                metric = tv_norm(x_result_fista)
                axs[1, 0].imshow(x_result_fista, cmap='gray', aspect='auto')
                axs[1, 0].set_title(f'FISTA - TV norm: {metric:0.2f}')

                x_result_gap = self.comparison_data['x_results'][1]
                metric = tv_norm(x_result_gap)
                axs[2, 0].imshow(x_result_gap, cmap='gray', aspect='auto')
                axs[2, 0].set_title(f'GAP - TV norm: {metric:0.2f}')

                # calculate FK domain

                dt = 0.568
                dx = 5

                FK, f, kx = fk(x, dt, dx)
                axs[0, 1].imshow(FK[:200], aspect='auto', cmap='jet', extent=[kx.min(), kx.max(), 65, 0])
                axs[0, 1].set_title('FK reference')

                FK, f, kx = fk(x_result_fista, dt, dx)
                axs[1, 1].imshow(FK[:200], aspect='auto', cmap='jet', extent=[kx.min(), kx.max(), 65, 0])
                axs[1, 1].set_title('FK FISTA')

                FK, f, kx = fk(x_result_gap, dt, dx)
                axs[2, 1].imshow(FK[:200], aspect='auto', cmap='jet', extent=[kx.min(), kx.max(), 65, 0])
                axs[2, 1].set_title('FK GAP')

                # ----------------- --------------------

                x_result_twist = self.comparison_data['x_results'][2]
                x_result_admm = self.comparison_data['x_results'][3]
                index = 5
                axs[0, 2].plot(x_result_fista[:, H_elim[index]], 'b', label='Fista')
                axs[0, 2].plot(x_result_gap[:, H_elim[index]], 'r', label='Gap')
                axs[0, 2].plot(x_result_twist[:, H_elim[index]], 'g', label='Twist')
                axs[0, 2].plot(x_result_admm[:, H_elim[index]], 'm', label='Admm')
                axs[0, 2].legend(loc='best')
                axs[0, 2].set_title('Trace ' + str("{:.0f}".format(H_elim[index])))
                axs[0, 2].grid(axis='both', linestyle='--')

                metric = tv_norm(x_result_twist)
                axs[1, 2].imshow(x_result_twist, cmap='gray', aspect='auto')
                axs[1, 2].set_title(f'TwIST - TV-norm: {metric:0.2f}')

                metric = tv_norm(x_result_admm)
                axs[2, 2].imshow(x_result_admm, cmap='gray', aspect='auto')
                axs[2, 2].set_title(f'ADMM - TV norm: {metric:0.2f}')

                # calculate FK domain

                dt = 0.568
                dx = 5

                index = -5
                axs[0, 3].plot(x_result_fista[:, H_elim[index]], 'b', label='Fista')
                axs[0, 3].plot(x_result_gap[:, H_elim[index]], 'r', label='Gap')
                axs[0, 3].plot(x_result_twist[:, H_elim[index]], 'g', label='Twist')
                axs[0, 3].plot(x_result_admm[:, H_elim[index]], 'm', label='Admm')
                axs[0, 3].legend(loc='best')
                axs[0, 3].set_title('Trace ' + str("{:.0f}".format(H_elim[index])))
                axs[0, 3].grid(axis='both', linestyle='--')

                FK, f, kx = fk(x_result_twist, dt, dx)
                axs[1, 3].imshow(FK[:200], aspect='auto', cmap='jet', extent=[kx.min(), kx.max(), 65, 0])
                axs[1, 3].set_title('FK TwIST')

                FK, f, kx = fk(x_result_admm, dt, dx)
                axs[2, 3].imshow(FK[:200], aspect='auto', cmap='jet', extent=[kx.min(), kx.max(), 65, 0])
                axs[2, 3].set_title('FK ADMM')

            self.draw()

        except BaseException as err:
            msg = f"Unexpected {err=}, {type(err)=}"
            showCritical(
                "Ocurrió un error inesperado al procesar el dato sísmico. Por favor, intente nuevamente o "
                "utilice un dato diferente.", details=msg)
            return


# ---------------------------------------------- shots ----------------------------------------------

class ShotPerformanceGraphic(FigureCanvasQTAgg):
    def __init__(self, is_complete=True):
        self.is_complete = is_complete
        self.performance_data = dict(iteracion=[], psnr=[], ssim=[], tv=[])
        self.figure = plt.figure()
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.08, top=0.92)
        super(ShotPerformanceGraphic, self).__init__(self.figure)

    def update_values(self, iteracion, psnr, ssim, tv):
        self.performance_data['iteracion'] = iteracion
        self.performance_data['psnr'] = psnr
        self.performance_data['ssim'] = ssim
        self.performance_data['tv'] = tv

    def update_figure(self):
        try:
            iteracion = self.performance_data['iteracion']
            psnr = self.performance_data['psnr']
            ssim = self.performance_data['ssim']
            tv = self.performance_data['tv']

            self.figure.clear()
            self.figure.suptitle(f'Resultados del experimento')

            if self.is_complete:
                axes_1 = self.figure.add_subplot(111)
                axes_2 = axes_1.twinx()

                color = 'tab:red'
                axes_1.set_xlabel('iteraciones')
                axes_1.plot(iteracion, ssim if self.is_complete else ssim, color=color,
                            label='SSIM' if self.is_complete else 'SSIM')
                axes_1.tick_params(axis='y', labelcolor=color, length=5)
                axes_1.grid(axis='both', linestyle='--')

                axes_1.set_yticks(np.linspace(axes_1.get_ybound()[0], axes_1.get_ybound()[1], 8))

                color = 'tab:blue'
                axes_2.plot(iteracion, psnr if self.is_complete else tv, '--', color=color)
                axes_1.plot(np.nan, '--', color=color, label='PSNR' if self.is_complete else 'Norma TV/L1')
                axes_2.tick_params(axis='y', labelcolor=color, length=5)
                axes_2.grid(axis='both', linestyle='--')

                axes_2.set_yticks(np.linspace(axes_2.get_ybound()[0], axes_2.get_ybound()[1], 8))

                if self.is_complete:
                    if np.abs(psnr[-1]) == np.inf:
                        # print a text in the bottom part of axes_2 that indicates that the psnr is infinite
                        text_kwargs = dict(ha='center', va='center', fontsize=16, color=color)
                        axes_2.text(0.5, 0.1, f'PSNR is {psnr[-1]}, it will be not plotted', horizontalalignment='center',
                                    verticalalignment='center', transform=axes_2.transAxes, **text_kwargs)

                axes_1.legend(loc='best')

            else:
                axes = self.figure.add_subplot(111)

                color = 'tab:blue'
                axes.set_xlabel('iteraciones')
                axes.plot(iteracion, tv, color=color, label='Norma TV/L1')
                axes.tick_params(axis='y', labelcolor=color, length=5)
                axes.grid(axis='both', linestyle='--')

                axes.set_yticks(np.linspace(axes.get_ybound()[0], axes.get_ybound()[1], 8))
                axes.legend(loc='best')

            self.draw()

        except BaseException as err:
            msg = f"Unexpected {err=}, {type(err)=}"
            showCritical("Ocurrió un error inesperado al procesar el dato sísmico. Por favor, intente nuevamente o "
                         "utilice un dato diferente.", details=msg)
            return


class ShotReconstructionGraphic(FigureCanvasQTAgg):
    def __init__(self, is_complete=True):
        self.is_complete = is_complete
        self.report_data = None
        self.figure = plt.figure()
        left, right, bottom, top = [0.05, 0.95, 0.05, 0.85] if is_complete else [0.05, 0.98, 0.05, 0.85]
        plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=0.3, hspace=0.5)
        super(ShotReconstructionGraphic, self).__init__(self.figure)

    def update_report(self, report_data):
        self.report_data = report_data

    def update_figure(self):
        try:
            self.figure.clear()

            x_result = self.report_data['x_result']
            sampling = {item[0]: item[1] for item in self.report_data['sampling']}

            if self.is_complete:
                x = sampling['x_ori']
                y_rand = sampling['y_rand']
                pattern_rand = sampling['pattern_rand']
            else:
                x = sampling['x_ori']
                y_rand = x[:, int(x.shape[1] / 2)].copy()
                pattern_rand = np.double(sampling['H'])

            temp = np.asarray(range(0, pattern_rand.shape[0]))
            pattern_rand_b2 = np.asarray(pattern_rand, dtype=bool) == 0
            H_elim = temp[pattern_rand_b2]

            case = str(self.report_data['algorithm_name'])
            self.figure.suptitle(f'Resultados del algoritmo {str.replace(case, "_", " ").capitalize()}')

            rem_shots = np.arange(len(pattern_rand))
            rem_shots = rem_shots[pattern_rand == 0]

            if self.is_complete:
                axs = self.figure.subplots(2, 4)

                # psnr_vec = []
                # for s in rem_shots:
                #     psnr_vec.append(PSNR(x[..., s], x_result[..., s]))
                # idxs = (-np.array(psnr_vec)).argsort()
                # rem_shots = rem_shots[idxs]

                axs[0, 0].imshow(x[:, int(x.shape[1] / 2)].copy(), cmap='gray', aspect='auto')
                axs[0, 0].set_title("Reference")
                axs[0, 0].set_xlabel("Shots")
                axs[0, 0].set_ylabel("Time")

                axs[1, 0].imshow(x[..., rem_shots[1]], cmap='gray', aspect='auto')
                axs[1, 0].set_title(f'Reference, shot {rem_shots[1]}')

                axs[0, 1].imshow(y_rand, cmap='gray', aspect='auto')
                axs[0, 1].set_title("Removed shots")
                axs[0, 1].set_xlabel("Shots")
                axs[0, 1].set_ylabel("Time")

                metric = PSNR(x[..., rem_shots[1]], x_result[..., rem_shots[1]])
                metric_ssim = ssim(x[..., rem_shots[1]], x_result[..., rem_shots[1]])
                axs[1, 1].imshow(x_result[..., rem_shots[1]], cmap='gray', aspect='auto')
                axs[1, 1].set_title(
                    f'Reconstructed shot {rem_shots[1]}, \n PSNR: {metric:0.2f} dB, \n SSIM:{metric_ssim:0.2f}')

                # ====
                axs[0, 2].imshow(x[..., rem_shots[2]], cmap='gray', aspect='auto')
                axs[0, 2].set_title(f'Reference, shot {rem_shots[2]}')

                axs[1, 2].imshow(x[..., rem_shots[3]], cmap='gray', aspect='auto')
                axs[1, 2].set_title(f'Reference, shot {rem_shots[3]}')

                metric = PSNR(x[..., rem_shots[2]], x_result[..., rem_shots[2]])
                metric_ssim = ssim(x[..., rem_shots[2]], x_result[..., rem_shots[2]])
                axs[0, 3].imshow(x_result[..., rem_shots[2]], cmap='gray', aspect='auto')
                axs[0, 3].set_title(
                    f'Reconstructed shot {rem_shots[2]}, \n PSNR: {metric:0.2f} dB, \n SSIM:{metric_ssim:0.2f}')

                metric = PSNR(x[..., rem_shots[3]], x_result[..., rem_shots[3]])
                metric_ssim = ssim(x[..., rem_shots[3]], x_result[..., rem_shots[3]])
                axs[1, 3].imshow(x_result[..., rem_shots[3]], cmap='gray', aspect='auto')
                axs[1, 3].set_title(
                    f'Reconstructed shot {rem_shots[3]}, \n PSNR: {metric:0.2f} dB, \n SSIM:{metric_ssim:0.2f}')

            else:
                axs = self.figure.subplots(2, 4)
                dt = 0.568
                dx = 5

                psnr_vec = []
                for s in rem_shots:
                    psnr_vec.append(PSNR(x[..., s], x_result[..., s]))
                idxs = (-np.array(psnr_vec)).argsort()
                rem_shots = rem_shots[idxs]

                axs[0, 0].imshow(x[:, int(x.shape[1] / 2)].copy(), cmap='gray', aspect='auto')
                axs[0, 0].set_title("Reference")
                axs[0, 0].set_xlabel("Shots")
                axs[0, 0].set_ylabel("Time")

                FK, f, kx = fk(x_result[..., rem_shots[1]], dt, dx)
                axs[1, 0].imshow(FK, aspect='auto', cmap='jet', extent=[kx.min(), kx.max(), 65, 0])
                axs[1, 0].set_title(f'FK reference, shot {rem_shots[1]}')

                axs[0, 1].imshow(x_result[:, int(x.shape[1] / 2)].copy(), cmap='gray', aspect='auto')
                axs[0, 1].set_title("Reconstructed shots")
                axs[0, 1].set_xlabel("Shots")
                axs[0, 1].set_ylabel("Time")

                metric = tv_norm(x_result[..., rem_shots[1]])
                axs[1, 1].imshow(x_result[..., rem_shots[1]], cmap='gray', aspect='auto')
                axs[1, 1].set_title(f'Reconstructed shot {rem_shots[1]}, \n Norma TV/L1: {metric:0.4f}')

                # ====
                FK, f, kx = fk(x_result[..., rem_shots[2]], dt, dx)
                axs[0, 2].imshow(FK, aspect='auto', cmap='jet', extent=[kx.min(), kx.max(), 65, 0])
                axs[0, 2].set_title(f'FK Reference, shot {rem_shots[2]}')

                FK, f, kx = fk(x_result[..., rem_shots[3]], dt, dx)
                axs[1, 2].imshow(FK, aspect='auto', cmap='jet', extent=[kx.min(), kx.max(), 65, 0])
                axs[1, 2].set_title(f'FK Reference, shot {rem_shots[3]}')

                metric = tv_norm(x_result[..., rem_shots[2]])
                axs[0, 3].imshow(x_result[..., rem_shots[2]], cmap='gray', aspect='auto')
                axs[0, 3].set_title(f'Reconstructed shot {rem_shots[2]}, \n Norma TV/L1: {metric:0.4f}')

                metric = tv_norm(x_result[..., rem_shots[3]])
                axs[1, 3].imshow(x_result[..., rem_shots[3]], cmap='gray', aspect='auto')
                axs[1, 3].set_title(f'Reconstructed shot {rem_shots[3]}, \n Norma TV/L1: {metric:0.4f}')

            self.draw()

        except BaseException as err:
            msg = f"Unexpected {err=}, {type(err)=}"
            showCritical("Ocurrió un error inesperado al procesar el dato sísmico. Por favor, intente nuevamente o "
                         "utilice un dato diferente.", details=msg)
            return
