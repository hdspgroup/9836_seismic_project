# **Datasets**

Documentation for the datasets used in **COmpressive Seismic Acquisition Design (COSAD)** project:

* `cube4.npy`
* `data.npy`
* `spii15s.npy`
* `syn3D_cross-spread2.npy`

# **`cube4.npy`**
Data from the Stratton 3D Survey, a small land 3D data set from South Texas located in Stratton Field, a fluvially deposited gas. The complete 3D source/receiver geometry consists of east-west receivers lines spaced 402 m apart (12 arrays in total) and north-south source lines spaced 268 m apart. For further information refer to https://wiki.seg.org/wiki/Stratton_3D_survey.

**Seismic adquisition parameters:**

* Time samples (`nt`) = `1001`
* Number of traces (`nx`) = `80`
* Number of shots (`ns`) = `18`
* Time interval (`dt`) = `0.003` ms
* Trace interval (`dx`) = `25` m

# **`data.npy`**
Synthetic dataset composed of 40 shots with 970 ms in-depth and 3.15 km of horizontal length. For seismic traces reconstruction, it was selected shot #20 and cropped to 800 and 100 samples in time and traces, respectively. For further information refer to https://github.com/PyLops/curvelops/blob/main/examples/Demo_Seismic_Regularization.ipynb.

**Seismic adquisition parameters:**

* Time samples (`nt`) = `800`
* Number of traces (`nx`) = `100`
* Number of shots (`ns`) = `1`
* Time interval (`dt`) = `0.568` ms
* Trace interval (`dx`) = `5` m

# **`spii15s.npy`**
This data was built by the SEG Advanced Modeling Program (SEAM) during its second project, called "SEAM Phase II–Land Seismic Challenges". The Foothills models are focused on mountainous regions with sharp topography at the surface and compressive fold and thrust tectonics at depth. For further information refer to https://drive.google.com/file/d/12274Q1JupEP5g7jdEb_m_KQCgMunPuNA/view.

**Seismic adquisition parameters:**

* Time samples (`nt`) = `1034`
* Number of traces (`nx`) = `100`
* Number of shots (`ns`) = `15`
* Time interval (`dt`) = `0.004` ms
* Trace interval (`dx`) = `12.5` m

# **`syn3D_cross-spread2.npy`**
Synthetic cross-spread seismic data modeled using finite differences with `devito`package (for further information refer to https://github.com/devitocodes/devito). The simulated geological conditions were semi-horizontal layers with minor folds and velocity anomalies. The total length of the seismic design is 1010 m horizontally and 1000 ms in depth.

**Seismic adquisition parameters:**

* Time samples (`nt`) = `1106`
* Number of traces (`nx`) = `101`
* Number of shots (`ns`) = `15`
* Time interval (`dt`) = `0.000905` ms
* Trace interval (`dx`) = `10` m
