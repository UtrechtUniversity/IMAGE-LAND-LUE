# IMAGE-LAND-LUE
Land cover model from the IMAGE 3.1 Integrated Assessment Model, implemented with the LUE framework.

## Running the model
To run the model, first the repository should be [cloned](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository), then the below steps followed:
1. Install [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) if it is not already on your system. 
2. Create a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) from environment.yml. This will install the packages required to run the model to a conda environment named `image`. Do this by running the following command line prompt in the repository's top directory:
    ```conda env create -f environment.yml```.
3. Activate `image` conda environment by running:
    ```conda activate image```.
4. Run `landalloc.py`. The number of timesteps and number of years each timestep is equivalent to can be adjusted by changing the `N_STEPS` and `INTERVAL` values in `parameters.py`. For more information, see the documentation.

## Docs and contributing to IMAGE-LAND-LUE
The documentation for IMAGE-LAND-LUE is not yet online, however all the source files to build the docs using [Sphinx](https://www.sphinx-doc.org/en/master/) are present in the repository. If the image conda environment is already installed (see steps above), the steps to build and consult the documentation are as follows:
1. Navigate to the docs folder in the terminal.
2. Run the prompt:
    ```sphinx-build -M html source build```.
    Note that `source` and `build` are the directories containing the source material to build the docs and the target directory for the build files, respectively. If the names of these folders are changed, `source` and `build` i nthe above command must be replaced with the new names .
3. The docs can be consulted by navigating to `docs/build/html` and running index.html. The documentation landing page will open in your default web browser.

It is recommended that the docs be consulted before trying to alter the model, as they contain general information about IMAGE-LAND and the allocation process, in addition to the information contained in the file and function docstrings.

## Outputs

The outputs of IMAGE-LAND-LUE depend on the options selected in `parameters.py`. In general, outputs are saved in two formats: <a href='https://numpy.org/devdocs/reference/generated/numpy.lib.format.html'>npy</a> and tiff formats. All outputs are stored in a directory named outputs, in imagelcm, the directory in which the model is stored.