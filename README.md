# Deepdrive [![Build Status](https://travis-ci.com/crizCraig/deepdrive-agents-beta.svg?token=hcA6yn9X8yYZspyyCMpp&branch=master)](https://travis-ci.com/crizCraig/deepdrive-agents-beta)

The easiest way to experiment with self-driving AI

## Simulator requirements

- Linux or Windows
- Python 3.5+ (Recommend Miniconda for Windows)
- 3GB disk space
- 8GB RAM

## Optional - baseline agent requirements

- CUDA capable GPU (tested and developed on 970, 1070, and 1060's)
- Tensorflow 1.1+ [See NVIDIA install tips](#nvidia-install-tips)

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

If you run into issues, try starting the sim directly as Unreal may need to install some prerequisetes. The default location is in your user directory under <kbd>Deepdrive/sim</kbd>

## Usage

Run the **baseline** agent
```
python main.py --baseline
```

Run in-game path follower
```
python main.py --let-game-drive
```

**Record** training data for imitation learning / behavioral cloning
```
python main.py --record --record-recovery-from-random-actions
```

**Train** on recorded data
```
python main.py --train
```

**Train** on the same dataset we used 

Grab the [dataset](#dataset)
```
python main.py --train --recording-dir <the-directory-with-the-dataset>
```

### Key binds 

* <kbd>Esc</kbd> - Pause (Quit in Unreal Editor)
* <kbd>Alt+Tab</kbd> - Control other windows
* <kbd>P</kbd> - Pause in Unreal Editor
* <kbd>J</kbd> - Toggle shared mem stats
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


## Benchmark



| 50 lap avg score  | Weights |  Deepdrive version |
| ---:   | :---    |   ---: |
|[3059](https://d1y4edi1yk5yok.cloudfront.net/benchmarks/2018-01-02__09-49-03PM.csv)|[baseline_agent_weights.zip](https://d1y4edi1yk5yok.cloudfront.net/weights/baseline_agent_weights.zip)|2.0.20180101022103|

## Dataset

1. Get the [AWS CLI](https://github.com/aws/aws-cli)
2. Ensure you have 104GB of free space
3. Download our dataset of mixed Windows (Unreal PIE + Unreal packaged) and Linux + variable camera and corrective action recordings 
(generated with `--record`)
```
cd <the-directory-you-want>
aws s3 sync s3://deepdrive/data/baseline .
```

If you'd like to check out our Tensorboard training session, you can download the 13GB
[tfevents files here](https://d1y4edi1yk5yok.cloudfront.net/tensorflow/baseline_tensorflow_train_and_eval.zip),
unzip, and run

```
tensorboard --logdir <your-unzipped-baseline_tensorflow_train_and_eval>
```

## Frame rate issues on Linux

If you experience low frame rates on Linux, you may need to install NVIDIA’s display drivers including their OpenGL drivers. We recommend installing these with CUDA which bundles the version you will need to run the baseline agent. Also, make sure to [plugin your laptop](https://help.ubuntu.com/community/PowerManagement/ReducedPower). If CUDA is installed, skip to testing [OpenGL](#opengl).

[CUDA install tips](#cuda-and-nvidia-driver-install)
[cuDNN install tips](#cudnn-install-tips)

## NVIDIA install tips

- Use the packaged install, i.e. deb[local] on Ubuntu, referred to in [this guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- If you are feeling dangerous and use the runfile method, be sure to follow [NVIDIA’s instructions](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) on how to disable the Nouveau drivers if you're on Ubuntu.

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


## Thanks

Special thanks to [Rafał Józefowicz](https://scholar.google.com/citations?user=C7zfAI4AAAAJ) for contributing the original [training](#tensorflow_agent/train) code used for the baseline agent

