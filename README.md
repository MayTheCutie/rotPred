# New Rotation Period measurements for Kepler stars using Deep Learning

by
Hagai Perets,
Ilay Kamai,

> official implementation of
> "New Rotation Period measurements for Kepler stars using Deep Learning"

This paper has been submitted for publication in *Some Journal*.

> LightPred is a deep learning model to learn stellar period and inclination
> using self supervised and simulation based learning. 
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


## Run Experiments

Before running any code you must activate the conda environment:

    source activate ENVIRONMENT_NAME

or, if you're on Windows:

    activate ENVIRONMENT_NAME

This will enable the environment for your current terminal session.
Any subsequent commands will use software that is installed in the environment.

To build and test the software, produce all results and figures, and compile
the manuscript PDF, run this in the top level of the repository:

    make all

If all goes well, the manuscript PDF will be placed in `manuscript/output`.

You can also run individual steps in the process using the `Makefile`s from the
`code` and `manuscript` folders. See the respective `README.md` files for
instructions.

Another way of exploring the code results is to execute the Jupyter notebooks
individually.
To do this, you must first start the notebook server by going into the
repository top level and running:

    jupyter notebook

This will start the server and open your default web browser to the Jupyter
interface. In the page, go into the `code/notebooks` folder and select the
notebook that you wish to view/run.

The notebook is divided into cells (some have text while other have code).
Each cell can be executed using `Shift + Enter`.
Executing text cells does nothing and executing code cells runs the code
and produces it's output.
To execute the whole notebook, run all cells in order.


## License

All source code is made available under a BSD 3-clause license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.

The manuscript text is not open source. The authors reserve the rights to the
article content, which is currently submitted for publication in the
JOURNAL NAME.