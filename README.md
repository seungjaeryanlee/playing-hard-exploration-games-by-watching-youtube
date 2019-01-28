# exploration-with-youtube

[![black Build Status](https://img.shields.io/travis/com/seungjaeryanlee/playing-hard-exploration-games-by-watching-youtube.svg?label=black)](https://travis-ci.com/seungjaeryanlee/playing-hard-exploration-games-by-watching-youtube)
[![flake8 Build Status](https://img.shields.io/travis/com/seungjaeryanlee/playing-hard-exploration-games-by-watching-youtube.svg?label=flake8)](https://travis-ci.com/seungjaeryanlee/playing-hard-exploration-games-by-watching-youtube)
[![isort Build Status](https://img.shields.io/travis/com/seungjaeryanlee/playing-hard-exploration-games-by-watching-youtube.svg?label=isort)](https://travis-ci.com/seungjaeryanlee/playing-hard-exploration-games-by-watching-youtube)

A PyTorch implementation of Playing Hard Exploration Games by Watching YouTube (DeepMind, May 2018)

## Installation

First, install FFMPEG for `skvideo.io`.

```
sudo apt-get install ffmpeg
```

Then, install Python packages using `requirements.txt`.

```
pip install -r requirements.txt
```

If you get error about being unable to uninstall `certifi`, try the command below instead:

```
pip install --ignored-installed -r requirements.txt
```

## Download Video and Audio

To train TDC and CMC, you need videos and audios. Because of their size, they are not included in the repository.

Running `get_av.sh` should automatically download and extract video and audio. However, in case you encounter issues, the description below explains the code inside `get_av.sh`.

### Video

Videos used for training is hosted in [my public Google Drive folder](https://drive.google.com/drive/folders/18EomHIBO9nUbBvllw0uG6SojlR7Ap0hg?usp=sharing). In Ubuntu, you can download videos through a `curl` command:

```
mkdir data/
cd data/
curl -L -o 2AYaxTiWKoY.mp4 "https://drive.google.com/uc?export=download&id=1d96v8R85Hz0_73zvb78aHIK1cntt8W-4"
curl -L -o 6zXXZvVvTFs.mp4 "https://drive.google.com/uc?export=download&id=1M_kIL-Xmdw-xlf_f6HrF94tDdcS3St8J"
curl -L -o pF6xCZA72o0.mp4 "https://drive.google.com/uc?export=download&id=18tmjd0w7_WprPnEhOEWIpgXxqrbey2mb"
curl -L -o SuZVyOlgVek.mp4 "https://drive.google.com/uc?export=download&id=1rK6h9Ya-3mpNoDAGwR2GYfiVavAv7rNw"
curl -L -o sYbBgkP9aMo.mp4 "https://drive.google.com/uc?export=download&id=18Zhkx4opv7lvpoPBm-L95RSkA_JuHoD8"
```


### Audio

WAV audio is extracted from the downloaded MP4 videos.

```
ffmpeg -i videos/2AYaxTiWKoY.mp4 audios/2AYaxTiWKoY.wav
ffmpeg -i videos/6zXXZvVvTFs.mp4 audios/6zXXZvVvTFs.wav
ffmpeg -i videos/pF6xCZA72o0.mp4 audios/pF6xCZA72o0.wav
ffmpeg -i videos/SuZVyOlgVek.mp4 audios/SuZVyOlgVek.wav
ffmpeg -i videos/sYbBgkP9aMo.mp4 audios/sYbBgkP9aMo.wav
```
