# Rotation Period measurements for Kepler, K2, TESS, and Gaia stars using Deep Learning

By May Flesher & Rowan Combs

Based on the research from
Hagai Perets & Ilay Kamai


unofficial implementation of
> "Accurate and Robust Stellar Rotation Periods catalog for 82771 Kepler stars using deep learning"
> 
> [![arXiv](https://img.shields.io/badge/arXiv-2407.06858-b31b1b.svg)](https://arxiv.org/abs/2407.06858) (accepted to AJ)

LightPred is a deep learning model to learn stellar period and inclination
using self supervised and simulation based learning. 

![alt text](https://github.com/ilayMalinyak/lightPred/blob/master/images/lightPred.drawio.png?raw=true)

*high level architecture.*
> 
![alt text](https://github.com/ilayMalinyak/lightPred/blob/master/images/period_exp47_scatter.png?raw=true)
*period results on simulations.*

## Setup Environment

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/IlayMalinyak/lightPred.git
    cd lightPred
    pip install -r requirements.txt

## Creating Simulated Samples
   to create simualated lightcurves we used **[butterpy](https://github.com/zclaytor/butterpy)** package.
~~note that we are currently using the deprecated version. this is why we use **butterpy_local** folder.~~
updating this to the most recent version of butter.py / might use lightkurve

the script to generate lightcurves is in [dataset\butter.py](https://github.com/IlayMalinyak/lightPred/blob/master/dataset/butter.py)

## Run Experiments

experiments can be found in [experiments](https://github.com/IlayMalinyak/lightPred/tree/master/experiments)
folder.


## Acknowledgements

- the implementation of Astroconf is based on: https://github.com/panjiashu/Astroconformer. we slightly modified the architecture
- we are using implementations from https://github.com/zclaytor/butterpy to simulate lightcurves
- some of the transformations in [transforms](https://github.com/IlayMalinyak/lightPred/tree/master/transforms) are based on https://github.com/mariomorvan/Denoising-Time-Series-Transformer
