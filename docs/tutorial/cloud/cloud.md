# Cloud setup


We've tested on Paperspace's ML-in-a-Box Linux public template with a P6000 which already has Tensorflow installed and just requires

To set it up choose your region, then go to Public Templates, and choose the ML-in-a-Box template

![Paperspace](https://i.imgur.com/ZyltYsM.png)

Then download the 64-bit Linux installer from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or click [here](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)

Now run the Miniconda installer

```
bash Miniconda3-latest-Linux-x86_64.sh
```

Next, create a conda env for deepdrive

```
bash # to get conda in your shell
conda create -n dd python=3.7
conda activate dd
```

Now clone our repo

```
git clone https://github.com/deepdrive/deepdrive
```

And finish by running the install

```
cd deepdrive
python install.py
```

Validate everything is working by running

```
python example.py
```

If you run into issues, try starting the sim directly as Unreal may need to install some prerequisetes (i.e. DirectX needs to be installed on the Paperspace Parsec Windows box). The default location of the Unreal sim binary is in your user directory under `Deepdrive/deepdrive-sim*/DeepDrive/Binaries/Linux/DeepDrive`, i.e.

```
/home/YOURUSERNAME/Deepdrive/deepdrive-sim-linux-3.1.20191112232556/DeepDrive/Binaries/Linux/DeepDrive
```

You can continue running the rest of our examples [here](https://docs.deepdrive.io/#examples)
