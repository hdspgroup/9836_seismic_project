# Divergence-Based Regularization for End-to-End Sensing Matrix Optimization in Compressive Sampling Systems

## Abstract

Sensing Matrix Optimization (SMO) in Compressed Sensing (CS) systems allows improved performance in the underlying signal decoding. Data-driven methods based on deep learning algorithms have opened a new horizon for SMO. The matrix is designed jointly with a decoder network that performs compressed learning tasks. This design paradigm, named End-to-End (E2E) optimization, comprises two parts: the sensing layer that models the acquisition system and the computational decoder. However, SMO in the E2E network has two main issues: i) it suffers from the vanishing of the gradient since the sensing matrix is the first layer of the network, and ii) there is no interpretability in the SMO, resulting in poorly compressed acquisition. To address these issues, we proposed a regularization function that gives some interpretability to the designed matrix and adds an inductive bias in the SMO. The regularization function is based on the Kullback-Leiber Divergence (KLD), which aims to approximate the distribution of the compressed measurements to a prior distribution. Thus, the sensing matrix can concentrate or spread the distribution of the compressed measurements according to the chosen prior distribution. We obtained optimal performance by concentrating the distribution in the recovery task, while in the classification task, the improvement was obtained by increasing the variance of the distribution. We validate the proposed regularized E2E method in general CS scenarios, such as in the Coded Aperture (CA) design for the Single-Pixel Camera (SPC) and Compressive Seismic Acquisition (CSA) geometry design.

## Dataset 

The employed dataset in this work can be downloaded in:

https://drive.google.com/file/d/1mN_hi7KzSjnjOPTyrfyIGTdK6bwMUDhE/view?usp=share_link

# Demo

Run a demo of the End-to-End optimization of the acquistion geometry and the reconstruction network with the proposed the Divergence-based regularization in `Main.ipynb` 

# Sensing models

The differentiable modeling o f the compressive seismic acquistion is in ´SensingModel2D.py´, `sampling_schemes.py` and `models2D.py`

# Recovery models

Different recovery networks are implemented in `Une2D.py`, `Resnet.py and  `Conv.py`

