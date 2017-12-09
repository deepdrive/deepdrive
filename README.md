# DeepDrive [![Build Status](https://travis-ci.com/crizCraig/deepdrive-agents-beta.svg?token=hcA6yn9X8yYZspyyCMpp&branch=master)](https://travis-ci.com/crizCraig/deepdrive-agents-beta)
The easiest way to experiment with self-driving AI

## Simulator requirements
- Linux or Windows
- Python 3.5+
- 3GB disk space
- 8GB RAM

## Baseline agent requirements
- CUDA capable GPU (tested and developed on 970, 1070, and 1060's)
- Tensorflow 1.1+ [NVIDIA install tips](#nvidia-install-tips)

## Install

```
git clone https://github.com/deepdrive/deepdrive
```

###### Optional - Activate the Python / virtualenv where your Tensorflow is installed

#### Linux

```
python install.py
```

#### Windows

Make sure the Python you want to use is in your **system** path, then

```
python install.py
```

## Usage

Run baseline agent
```
bin/run_baseline_agent.sh
```

Test your own driving ability!
```
bin/drive_manually.sh
```

Record training data
```
bin/record_training_data.sh
```

Train an imitation learning agent on recorded data
```
bin/train.sh
```

Train on the same dataset we used
```
bin/train_baseline.sh
```

[Frame rate issues?](#framerate-issues-on-linux)

#### Frame rate issues on Linux

If you experience low frame rates on Linux, you may need to install NVIDIA’s display drivers including their OpenGL drivers. We recommend installing these with CUDA which bundles the version you will need to run the baseline agent. Also, make sure to [plugin your laptop](https://help.ubuntu.com/community/PowerManagement/ReducedPower). If CUDA is installed, skip to testing [OpenGL](#opengl).

[CUDA install tips](#cuda-and-nvidia-driver-install)
[cuDNN install tips](#cudnn-install-tips)

#### NVIDIA install tips

- Install [CUDA 8](https://developer.nvidia.com/cuda-toolkit-archive) and cuDNN 6 as Tensorflow does not yet support  NVIDIA's [current downloads]( https://github.com/tensorflow/tensorflow/issues/12052) - If you want to use CUDA 9 / cuDNN 7, you will need to install Tensorflow from sources.
- Follow [this guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) carefully
- The packaged install is highly recommended, i.e. deb[local] on Ubuntu
- If you don't and use the runfile method, be sure to follow [NVIDIA’s instructions](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) on how to disable the Nouveau drivers if you're on Ubuntu.


Make sure to follow the 
[CUDA installation instructions](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) 
thoroughly as you could end up with a broken video driver, login issues, and lots of frustration. We recommend the package manager installation method, i.e. deb[local], for the smoothest install of both CUDA and cuDNN. The runfile method can be fraught with pain, but if you really want to use it, make sure to follow NVIDIA’s instructions on how to disable the Nouveau drivers if you're on Ubuntu. You may want to have another computer handy (or use your mobile if you have to) to search for answers while your machine is unusable. Also, get the older CUDA 8 and cuDNN 6 for a standard tensorflow install later on (read here to see if this changes: https://github.com/tensorflow/tensorflow/issues/12052) - If you want to use CUDA 9 / cuDNN 7, you’ll need to install Tensorflow from source.
http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

#### OpenGL

`glxinfo | grep OpenGL` should return something like:
```
OpenGL vendor string: NVIDIA Corporation
OpenGL renderer string: GeForce GTX 980/PCIe/SSE2
OpenGL core profile version string: 4.5.0 NVIDIA 384.90
OpenGL core profile shading language version string: 4.50 NVIDIA
OpenGL core profile context flags: (none)
OpenGL core profile profile mask: core profile
OpenGL core profile extensions:
OpenGL version string: 4.5.0 NVIDIA 384.90
OpenGL shading language version string: 4.50 NVIDIA
OpenGL context flags: (none)
OpenGL profile mask: (none)
OpenGL extensions:
OpenGL ES profile version string: OpenGL ES 3.2 NVIDIA 384.90
OpenGL ES profile shading language version string: OpenGL ES GLSL ES 3.20
OpenGL ES profile extensions:
```
You may need to disable secure boot in your BIOS in order for NVIDIA’s OpenGL and tools like nvidia-smi to work. This is not DeepDrive specific, but rather a general requirement of Ubuntu’s NVIDIA drivers.


Also `sudo shutdown -r now` is your friend


### Development

To run tests in PyCharm, go to File | Settings | Tools | Python Integrated Tools and change the default test runner 
to py.test.


### Random notes (TODO: Cleanup)

Windows
Controls

TODO:
Integrate packaged project into setup.py of deepdrive

TIPS:
If you lose your mouse pointer while in the game, just Alt-Tab!
deepdrive-agents
Linux
`git clone https://github.com/deepdrive/deepdrive-agents`
Install Tensorflow
Tips:
Make sure the CUDA_HOME environment variable is set (we used Cuda 8 for the baseline model), specifically /usr/local/cuda

Also make sure LD_LIBRARY_PATH is set - i.e. /home/<YOU>/cuda/lib64:

PATH includes /usr/local/cuda-YOUR_CUDA_VERSION/bin
Install dependencies with pipenv
Get pipenv if you don’t have it
pip install --user pipenv
pipenv install
deepdrive/Plugins/DeepDrivePlugin/Source/DeepDrivePython$ python setup.py install



To stop the simulation from the Unreal Editor in Linux, hit Esc. 
Windows
Tensorflow install tips

Add /Users/a/Miniconda3/Scripts and /Users/a/Miniconda3 to your user environment variable path

To run Tensorboard, navigate to your C:\Users\<YOU>\Miniconda3\envs\tensorflow\Scripts and run tensorboard --logdir=/tmp/gtanet

Merge your cuDNN bin, lib, and include into your CUDA install bin, lib, and include directories, i.e.
- bin files to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin
- lib files to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64
- headers to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include

Add cupti64_80.dll to user PATH environment variable, i.e.
cupti64_80.dll
in
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\extras\CUPTI\libx64
Restart programs to get new path

Download https://www.dropbox.com/s/x153v1d001fqu91/bvlc_alexnet.ckpt?dl=1


Freeglut viewer (optional)
OpenGL / Freeglut
conda install pyopengl
http://www.transmissionzero.co.uk/software/freeglut-devel/
Make sure to get the architecture (x64) that corresponds with the Python version

Mac
Mac will be unsupported at first due to poor GPU support.

You will need MacOs to download Xcode 8 and build Unreal from sources but we don’t need that for substance.

External GPU’s will allow Macs to run demanding GPU loads, but we haven’t tried this setup yet. An alternative way to run eGPUs on Apple hardware would be to use Bootcamp to run Windows which appears to have the best eGPU support as of mid-2017. 


Running the baseline agent

All

Install OpenCV
conda install -c https://conda.anaconda.org/menpo opencv3
conda install scipy
conda install pillow
After opening
Turn off background nerfing
Edit->Editor Preferences->Miscellaneous and disabling the Use Less CPU When in Background option.
PyCharm Development of Environment
If you open an Unreal project in Pycharm, add Binaries, Build, Content, Intermediate, and Saved to your project’s “Excluded” directories in Project Structure or simply by right clicking and choosing “Mark Directory as” => “Excluded”. Keeping these large binary directories in the project will cause PyCharm to freeze while it tries to index them.
Development in Unreal
Windows is the best environment for this. We are on Unreal 4.14 due to our use of the substance plugin accessed through unreal’s github (https://www.unrealengine.com/en-US/ue4-on-github)

https://github.com/Allegorithmic/UnrealEngine

In Windows, open the project, generate / refresh Visual Studio files,, then close Unreal and run the project through visual studio. This will allow you to debug in visual studio. When you change code though, compile it in Unreal with the “Compile” button (to the left of Play). It will then say “Compiling C++ code” on the bottom right.

If you change the plugin, recompile through Unreal -> Windows -> Developer -> Modules and fine the DeepDrivePlugin


FAQ
Q: Getting an error in built-in function step. 
A: Try rebuilding the python module - 

```
cd DeepDrivePythonpython
setup.py install
```

Key binds (Set these in Unreal->Project Settings->Input->Action Mappings or in blueprints->Find (uncheck Find in Current Blueprint Only)

Escape - Pause
P - Pause (useful in Unreal Editor)
J - Shared mem stats
1 - Chase cam
2 - Orbit (side) cam
3 - Interior cam
WASD or Up, Down, Left Right - steer / throttle
Space - Handbrake
Shift - Nitro
H - Horn
L - Light
R - Reset
E - Gear Up
Q - Gear down
Z, move mouse outside window and click OR shift+F1 in editor mode OR Alt+Tab - shows mouse cursor

