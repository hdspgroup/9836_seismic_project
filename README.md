# COmpressive Seismic Acquisition Design (COSAD) project

---

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]
![fund](https://img.shields.io/badge/Fundby-Minciencias--ANH-red)
![coverage](https://img.shields.io/badge/status-90%25-yellowgreen)

Esta herramienta software hace parte del proyecto 9836 - "Nuevas tecnologías computacionales para el diseño de sistemas de adquisición sísmica 3D terrestre con muestreo compresivo para la reducción de costos económicos e impactos ambientales en la exploración de hidrocarburos en cuencas terrestres colombianas" adscrito a la Convocatoria para la financiación de proyectos de investigación en geociencias para el sector de hidrocarburos, desarrollado por la alianza, Universidad Industrial de Santander (UIS), ECOPETROL y la Asociación Colombiana de Geólogos y Geofísicos del Petróleo (ACGGP). Este proyecto es financiado por MINCIENCIAS y la Agencia Nacional de Hidrocarburos (ANH). Los derechos sobre este software están reservados a las entidades aportantes.

# **Inicio rápido**

Esta es una guía para instalar las dependencias necesarias para poder ejecutar la aplicación de forma correcta.

### **Prerequisitos**

1. Instalar anaconda.
2. Crear entorno virtual en anaconda. 
3. Instalar Pycharm.
4. Instalar dependencias en el entorno virtual creado.
5. Correr `launch_window.py` con el entorno virtual creado.

De acuerdo al sistema operativo las anteriores instrucciones se deben cumplir de distintas formas, para obtener indicaciones más detalladas ingrese al siguiente enlace: [Instalación de dependencias](https://github.com/carlosh93/9836_seismic_project/wiki/I.-Manual-de-Instalación-Aplicación-ReDs,-Modo-Desarrolador).

# **Datos sísmicos**

---

There are four available datasets:

* `cube4.npy`
* `data.npy`
* `spii15s.npy`
* `syn3D_cross-spread2.npy`

# **`cube4.npy`**
Datos reales del Stratton 3D Survey, un pequeño conjunto de datos terrestres en 3D del sur de Texas ubicado en Stratton Field, un gas depositado fluvialmente. La geometría completa de fuente/receptor 3D consta de líneas de receptores este-oeste separadas por 402 m (12 arreglos en total) y líneas de fuente norte-sur separadas por 268 m. Para descargar el conjunto de datos completo, consulte https://wiki.seg.org/wiki/Stratton_3D_survey.

**Parámetros de adquisición sísmica:**

* Muestras de tiempo (`nt`) = `1001`
* Número de trazas (`nx`) = `80`
* Número de shots o disparos (`ns`) = `18`
* Intervalo de tiempo (`dt`) = `0.003` ms
* Intervalo de traza (`dx`) = `25` m

# **`data.npy`**
Conjunto de datos sintético compuesto por 40 disparos con 970 ms de profundidad y 3,15 km de longitud horizontal. Para la reconstrucción de las huellas sísmicas, se seleccionó el tiro #20 y se recortó a 800 y 100 muestras en tiempo y huellas, respectivamente. Para obtener más información, consulte https://github.com/PyLops/curvelops/blob/main/examples/Demo_Seismic_Regularization.ipynb.

**Parámetros de adquisición sísmica:**

* Muestras de tiempo (`nt`) = `800`
* Número de trazas (`nx`) = `100`
* Número de shots o disparos (`ns`) = `1`
* Intervalo de tiempo (`dt`) = `0.568` ms
* Intervalo de traza (`dx`) = `5` m

# **`spii15s.npy`**
Estos datos fueron construidos por el Programa de Modelado Avanzado de SEG (SEAM) durante su segundo proyecto, llamado "SEAM Phase II–Land Seismic Challenges". Los modelos de Foothills se centran en regiones montañosas con topografía pronunciada en la superficie y tectónica compresiva de pliegues y empujes en profundidad. Para obtener más información, consulte https://drive.google.com/file/d/12274Q1JupEP5g7jdEb_m_KQCgMunPuNA/view.

**Parámetros de adquisición sísmica:**

* Muestras de tiempo (`nt`) = `1034`
* Número de trazas (`nx`) = `100`
* Número de shots o disparos (`ns`) = `15`
* Intervalo de tiempo (`dt`) = `0.004` ms
* Intervalo de traza (`dx`) = `12.5` m

# **`syn3D_cross-spread2.npy`**
Datos sísmicos sintéticos de dispersión cruzada modelados utilizando diferencias finitas con el paquete `devito` (para obtener más información, consulte https://github.com/devitocodes/devito). Las condiciones geológicas simuladas fueron capas continuas y paralelas con velocidad creciente en profundidad. La principal estructura geológica es un anticlinal con acumulación de hidrocarburos en su núcleo, lo que provoca anomalías de velocidad. La longitud total del diseño sísmico es de 1010 m en horizontal y 1000 m en profundidad.

**Parámetros de adquisición sísmica:**

* Muestras de tiempo (`nt`) = `1106`
* Número de trazas (`nx`) = `101`
* Número de shots o disparos (`ns`) = `15`
* Intervalo de tiempo (`dt`) = `0.000905` ms
* Intervalo de traza (`dx`) = `10` m

[Más información sobre conjuntos de datos está disponible aquí.](https://github.com/carlosh93/9836_seismic_project/blob/652f805a3acf3176a32dbd4966bedbb70ef9545a/data/README.md)

<!--# Survey Binning acquisition

---

Summary scripts fold calculation, offset diagrams and other acquisition parameter needed in survey layout. @Claudia and @Paul

# Reconstruction algorithms

---

Summary - algorithms. All needed documentation, including references and so on. @Bacca,@Karen, @Kareth


Summary about the GUI, screenshots, and some breif description @Hinojosa-->

# Licencia

---

Este trabajo está licenciado bajo
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
