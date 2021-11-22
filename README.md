[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]
![fund](https://img.shields.io/badge/Fundby-Minciencias--ANH-red)
![coverage](https://img.shields.io/badge/status-10%25-yellowgreen)

# COmpressive Seismic Acquisition Design (COSAD) project

All needed dependencies, if it is possible, some conda environment. @all

This is the repository for the Seismic project No 9836 funded by MINCIENCIAS-ANH.

First install some dependencies:

```
python -m pip install numpy
python -m pip install matplotlib
python -m pip install scikit-image
python -m pip install scipy
```

# **Seismic Data**

There are four available datasets:

* `cube4.npy`
* `data.npy`
* `spii15s.npy`
* `syn3D_cross-spread2.npy`

# **`cube4.npy`**
Real data from the Stratton 3D Survey, a small land 3D data set from South Texas located in Stratton Field, a fluvially deposited gas. The complete 3D source/receiver geometry consists of east-west receivers lines spaced 402 m apart (12 arrays in total) and north-south source lines spaced 268 m apart. To download the complete dataset, refers to https://wiki.seg.org/wiki/Stratton_3D_survey.

**Seismic adquisition parameters:**

* Time samples (`nt`) = `1001`
* Number of traces (`nx`) = `80`
* Number of shots (`ns`) = `18`
* Time interval (`dt`) = `0.003` ms
* Trace interval (`dx`) = `25` m

# **`data.npy`**
Synthetic dataset composed of 40 shots with 970 ms in-depth and 3.15 km of horizontal length. For seismic traces reconstruction, it was selected shot #20 and cropped to 800 and 100 samples in time and traces, respectively. For further information refers to https://github.com/PyLops/curvelops/blob/main/examples/Demo_Seismic_Regularization.ipynb.

**Seismic adquisition parameters:**

* Time samples (`nt`) = `800`
* Number of traces (`nx`) = `100`
* Number of shots (`ns`) = `1`
* Time interval (`dt`) = `0.568` ms
* Trace interval (`dx`) = `5` m

# **`spii15s.npy`**
This data was built by the SEG Advanced Modeling Program (SEAM) during its second project, called "SEAM Phase II–Land Seismic Challenges". The Foothills models are focused on mountainous regions with sharp topography at the surface and compressive fold and thrust tectonics at depth. For further information refers to https://drive.google.com/file/d/12274Q1JupEP5g7jdEb_m_KQCgMunPuNA/view.

**Seismic adquisition parameters:**

* Time samples (`nt`) = `1034`
* Number of traces (`nx`) = `100`
* Number of shots (`ns`) = `15`
* Time interval (`dt`) = `0.004` ms
* Trace interval (`dx`) = `12.5` m

# **`syn3D_cross-spread2.npy`**
Synthetic cross-spread seismic data modeled using finite differences with `devito` package (for further information refers to https://github.com/devitocodes/devito). The simulated geological conditions were continuous and parallel layers with increasing velocity in depth. The main geological structure is an anticline with hydrocarbon accumulation on its core, which causes velocity anomalies. The total length of the seismic design is 1010 m horizontally and 1000 ms in depth.

**Seismic adquisition parameters:**

* Time samples (`nt`) = `1106`
* Number of traces (`nx`) = `101`
* Number of shots (`ns`) = `15`
* Time interval (`dt`) = `0.000905` ms
* Trace interval (`dx`) = `10` m

[More information about datasets is available here.](https://github.com/carlosh93/9836_seismic_project/blob/652f805a3acf3176a32dbd4966bedbb70ef9545a/data/README.md)

# Survey Binning acquisition

Summary scripts fold calculation, offset diagrams and other acquisition parameter needed in survey layout. @Claudia and @Paul

# Reconstruction algorithms

Summary - algorithms. All needed documentation, including references and so on. @Bacca,@Karen, @Kareth

# Graphic User Interface

The main resources for constructing the GUI are placed in the `gui` folder. To run the GUI, first make sure you have installed the
required packages with:

`pip install -r requirements.txt`

Then, run the `main_window.py` script as:

`pip install -r requirements.txt`

<!--Summary about the GUI, screenshots, and some breif description @Hinojosa-->

# License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
