# Setup

## Versions
* Python: 3.12.2

## Operating Systems
Windows, Mac OS, and Linux should all work well.

There are issues with rendering gym's environments with Windows Subsystem for Linux (an OpenGL issue). So, it is not recommended to use WSL (unless you want to debug that issue yourself).

## Editor
It's recommended to use Visual Studio Code, but you may use any IDE you would like. There's nothing specific in this repo to any IDE.

## Anaconda (Optional)
It is recommended, but not required to use an Anaconda. By default, `pip` installs everything in a global context.
This means you can have lots of difficulties when switching between different projects that use, e.g.
different versions of `python` or other packages (e.g. `numpy`).

Anaconda lets you define and switch between different "environments". These environments can all have different versions of python and other `pip` packages installed.

### Windows
For Windows users we recommend installing Miniconda. Install instructions can be found here:  https://docs.anaconda.com/miniconda/.

To open miniconda shell, find it in your start menu by searching for "Anaconda Prompt (miniconda3)".

### Linux & Mac OS

### Setting up the CS394R environment
Open the Anaconda Prompt shell. Run:
* `conda create --name cs394r python=3.12.2`
* `conda activate cs394r`
* Navigate to the the root of this repo.
* `pip install -r requirements.txt`
* Open your code editor, e.g. `code .`

You'll need to run the activate command each time you open the Anaconda Prompt.

## Installing the dependencies
You can install all the dependencies of this project using `pip install -r requirements.txt`.

Ensure this runs without any errors. If you do get errors, please make a public discussion board post.

## SWIG
SWIG is required for running Gym's Box-2D environments.

#### Windows
Follow the instructions [here](https://open-box.readthedocs.io/en/latest/installation/install_swig.html).

If you don't know how to add a system variable refer to [this](https://windowsloop.com/how-to-add-to-windows-path/). 

#### Linux
`sudo apt-get install swig build-essential python-dev python3-dev`

#### Mac
`brew install swig`
