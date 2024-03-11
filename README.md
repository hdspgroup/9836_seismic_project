# COmpressive Seismic Acquisition Design (COSAD) project

---

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]
![fund](https://img.shields.io/badge/Fundby-Minciencias--ANH-red)
![coverage](https://img.shields.io/badge/status-90%25-yellowgreen)

Esta herramienta software hace parte del proyecto 9836 - "Nuevas tecnologías computacionales para el diseño de sistemas de adquisición sísmica 3D terrestre con muestreo compresivo para la reducción de costos económicos e impactos ambientales en la exploración de hidrocarburos en cuencas terrestres colombianas" adscrito a la Convocatoria para la financiación de proyectos de investigación en geociencias para el sector de hidrocarburos, desarrollado por la alianza, Universidad Industrial de Santander (UIS), ECOPETROL y la Asociación Colombiana de Geólogos y Geofísicos del Petróleo (ACGGP). Este proyecto es financiado por MINCIENCIAS y la Agencia Nacional de Hidrocarburos (ANH). Los derechos sobre este software están reservados a las entidades aportantes.

# **SOFTWARE ReDS - Reconstrucción de Trazas y disparos sísmicos**

Esta es una guía rápida para la ejecución y uso del software en 3 modos:

### **Modo desarrollador**

1. Instalar anaconda.
2. Instalar el ambiente virtual yml  `conda env create -f enviro_reds.yml` 
3. Ejecutar `python launch_window.py` con el entorno virtual creado.

De acuerdo al sistema operativo las anteriores instrucciones se deben cumplir de distintas formas, para obtener indicaciones más detalladas ingrese al siguiente enlace: [Instalación de dependencias](https://github.com/carlosh93/9836_seismic_project/wiki/I.-Manual-de-Instalación-Aplicación-ReDs,-Modo-Desarrolador).

### **Modo compilador**

Escribir aqui el modo compilador

### ** Notebooks para ejecutar algoritmos de reconstrucción con deep learning**
En el siguiente [Link](https://github.com/hdspgroup/9836_seismic_project/tree/main/Algorithms/ReconDeepLearning) se presenta un ejemplo con Tensorflow para ejercicios de reconstrucción de datos sísmicos. Lo anterior, permite usar los requerimientos computacionales para el uso de datos masivos en computación de alto rendimiento, computación en paralelo y unidades de procesamiento gráfica. 

# **Datos sísmicos**

---

There are four available datasets:

* `cube4.npy`
* `data.npy`
* `spii15s.npy`
* `syn3D_cross-spread2.npy`

[Más información sobre conjuntos de datos está disponible aquí.](https://github.com/carlosh93/9836_seismic_project/blob/652f805a3acf3176a32dbd4966bedbb70ef9545a/data/README.md)

# **Análisis de geometrías**

Describri aqui lo de análisis de geometrías

# Licencia

---

Este trabajo está licenciado bajo
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
