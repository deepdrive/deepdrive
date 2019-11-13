# Cloud setup


We've tested on Paperspace's ML-in-a-Box Linux public template with a P6000 which already has Tensorflow installed and just requires

```
python install.py
```

To set it up choose your region, then go to Public Templates, and choose the ML-in-a-Box template

![Paperspace](https://i.imgur.com/ZyltYsM.png)

Then download the 64-bit Linux installer from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or click [here](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)

Now run the Miniconda installer

```
bash Miniconda3-latest-Linux-x86_64.sh
```

If you run into issues, try starting the sim directly as Unreal may need to install some prerequisetes (i.e. DirectX needs to be installed on the Paperspace Parsec Windows box). The default location of the Unreal sim binary is in your user directory under <kbd>Deepdrive/sim</kbd>.
