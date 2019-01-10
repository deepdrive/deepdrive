# Deepdrive [![Build Status](https://travis-ci.com/crizCraig/deepdrive-agents-beta.svg?token=hcA6yn9X8yYZspyyCMpp&branch=master)](https://travis-ci.com/crizCraig/deepdrive-agents-beta)

The easiest way to experiment with self-driving AI

## Simulator requirements

- Linux or Windows
- Python 3.5+ (Recommend Miniconda for Windows)
- 3GB disk space
- 8GB RAM

## Optional - baseline agent requirements

- CUDA capable GPU (tested and developed on 970, 1070, and 1060's)
- Tensorflow 1.7+ [See Tensorflow install tips](#tensorflow-install-tips)

## Install

```
git clone https://github.com/deepdrive/deepdrive
cd deepdrive
```

> Optional - Activate the Python / virtualenv where your Tensorflow is installed, then

#### Linux
```
python install.py
```

#### Windows
Make sure the Python you want to use is in your PATH, then

> Tip: We highly recommend using [Conemu](https://conemu.github.io/) for your Windows terminal

```
python install.py
```

#### Cloud

We've tested on Paperspace's ML-in-a-Box Linux public template which already has Tensorflow installed and just requires

```
python install.py
```

If you run into issues, try starting the sim directly as Unreal may need to install some prerequisetes (i.e. DirectX needs to be installed on the Paperspace Parsec Windows box). The default location of the Unreal sim binary is in your user directory under <kbd>Deepdrive/sim</kbd>.

## Usage

#### Running

Run the **baseline** agent
```
python main.py --baseline --experiment my-baseline-test
```

Run in-game path follower
```
python main.py --path-follower --experiment my-path-follower-test
```

**Record** training data for imitation learning / behavioral cloning
```
python main.py --record --jitter-actions --sync
```

Note that we recorded the baseline dataset in sync mode which is much slower than async mode. 
Async mode _probably_ is fine to record in, we just haven't got around to trying it out for v2.1.

Optional: Convert to HDF5 files to tfrecords (for training MNET2)
```
python main.py --hdf5-2-tfrecord
```


#### Training

**Train** on recorded data
```
python main.py --train [--agent dagger|dagger_mobilenet_v2|bootstrapped_ppo2]
```

**Train** on the same dataset we used 

Grab the [dataset](#dataset)
```
python main.py --train --recording-dir <the-directory-with-the-dataset> [--agent dagger|dagger_mobilenet_v2|bootstrapped_ppo2]
```

**Tensorboard**

```
tensorboard --logdir="<your-deepdrive-home>/tensorflow"
```

Where `<your-deepdrive-home>` below is by default in `$HOME/Deepdrive` and can be configured in `$HOME/.deepdrive/deepdrive_dir`

### Key binds 

* <kbd>Esc</kbd> - Pause (Quit in Unreal Editor)
* <kbd>Alt+Tab</kbd> - Control other windows
* <kbd>P</kbd> - Pause in Unreal Editor
* <kbd>;</kbd> - Toggle FPS
* <kbd>1</kbd> - Chase cam
* <kbd>2</kbd> - Orbit (side) cam
* <kbd>3</kbd> - Hood cam
* <kbd>4</kbd> - Free cam (use WASD to fly)
* WASD or Up, Down, Left Right - steer / throttle
* <kbd>Space</kbd> - Handbrake
* <kbd>Shift</kbd> - Nitro
* <kbd>H</kbd> - Horn
* <kbd>L</kbd> - Light
* <kbd>R</kbd> - Reset
* <kbd>E</kbd> - Gear Up
* <kbd>Q</kbd> - Gear down
* <kbd>Z</kbd> - Show mouse
* <kbd>`</kbd><kbd>`</kbd> - Unreal console (first press releases game input capture)


## Benchmark

| Agent  |  10 lap avg score  | Weights |  Deepdrive version |
| :---    | ---:   | :---    |   ---: |
|Baseline agent (trained with imitation learning)|[1691](https://docs.google.com/spreadsheets/d/1ryFaMFJhcTMBuhXZv0eMFHO35NMcXE2_MFLYqeUosfM/edit#gid=0)|[baseline_agent_weights.zip](https://d1y4edi1yk5yok.cloudfront.net/weights/baseline_agent_weights.zip)|2.0|
|Path follower |[*1069](https://docs.google.com/spreadsheets/d/1T5EuEobdVFn5ewdYTO20i9CqcZ-jIEsAihlV5lpvLQQ/edit#gid=0)| N/A (see [3D spline follower](https://github.com/crizCraig/deepdrive-beta/blob/bde6b8c48314c34a96ce0942fc398fae840720ee/Source/DeepDrive/Private/Car.cpp#L409))|2.0|

*The baseline agent currently outperforms the path follower it was trained on, likely due to the slower
speed the at which the baseline agent drives, resulting in lower lane deviation and g-force penalties. 
Interestingly, reducing the path follower speed causes it to crash at points where it otherwise loses traction and drifts, 
so the baseline agent has actually learned a more robust turning function than the original hardcoded path follower
it was trained on.

## Dataset

100GB (8.2 hours of driving) of camera, depth, steering, throttle, and brake of an 'oracle' path following agent. We rotate between three different cameras: normal, wide, and semi-truck - with random camera intrisic/extrinsic perturbations at the beginning of each episode (lap). This boosted performance on the benchmark by 3x. We also use DAgger to collect course correction data as in previous versions of Deepdrive.

1. Get the [AWS CLI](https://github.com/aws/aws-cli)
2. Ensure you have 104GB of free space
3. Download our dataset of mixed Windows (Unreal PIE + Unreal packaged) and Linux + variable camera and corrective action recordings 
(generated with `--record`)
```
cd <the-directory-you-want>
aws s3 sync s3://deepdrive/data/baseline_tfrecords .
```
or for the legacy HDF5 files for training AlexNet
```
aws s3 sync s3://deepdrive/data/baseline .
```

If you'd like to check out our Tensorboard training session, you can download the 1GB
[tfevents files here](https://d1y4edi1yk5yok.cloudfront.net/tensorflow/mnet2_baseline_training_and_eval.zip),
unzip, and run

```
tensorboard --logdir <your-unzipped-dir>
```

and checkout [this view](http://localhost:6006/#scalars&_smoothingWeight=0.935&runSelectionState=eyIyMDE4LTA3LTE5X18wNS01My0yN1BNIjp0cnVlLCIyMDE4LTA3LTE5X18wNS01MC01NFBNIjp0cnVlfQ%3D%3D&_ignoreYOutliers=false&tagFilter=error)
, which graphs wall time.

## Frame rate issues on Linux

If you experience low frame rates on Linux, you may need to install NVIDIA’s display drivers including their OpenGL drivers. We recommend installing these with CUDA which bundles the version you will need to run the baseline agent. Also, make sure to [plugin your laptop](https://help.ubuntu.com/community/PowerManagement/ReducedPower). If CUDA is installed, skip to testing [OpenGL](#opengl).

## Tensorflow install tips

- Make sure to install the CUDA / cuDNN major and minor version the Tensorflow instructions specify.  i.e. CUDA 9.0 / cuDNN 7.3 for Tensorflow 1.12.0. These will likely be older than the latest version NVIDIA offers. You can see all [CUDA  releases here](https://developer.nvidia.com/cuda-toolkit-archive).
- Use the packaged install, i.e. deb[local] on Ubuntu, referred to in [this guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- If you are feeling dangerous and use the runfile method, be sure to follow [NVIDIA’s instructions](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) on how to disable the Nouveau drivers if you're on Ubuntu.
- On Windows, use standard (non-CUDA packaged) display drivers which meet the min required. When installing CUDA, do a custom install and uncheck the display driver install.

## OpenGL

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
You may need to disable secure boot in your BIOS in order for NVIDIA’s OpenGL and tools like nvidia-smi to work. This is not Deepdrive specific, but rather a general requirement of Ubuntu’s NVIDIA drivers.


## Development

To run tests in PyCharm, go to File | Settings | Tools | Python Integrated Tools and change the default test runner 
to py.test.

