# Playing Hard Exploration Games by Watching YouTube

[![black Build Status](https://img.shields.io/travis/com/seungjaeryanlee/playing-hard-exploration-games-by-watching-youtube/master.svg?label=black)](https://travis-ci.com/seungjaeryanlee/playing-hard-exploration-games-by-watching-youtube)
[![flake8 Build Status](https://img.shields.io/travis/com/seungjaeryanlee/playing-hard-exploration-games-by-watching-youtube/master.svg?label=flake8)](https://travis-ci.com/seungjaeryanlee/playing-hard-exploration-games-by-watching-youtube)
[![isort Build Status](https://img.shields.io/travis/com/seungjaeryanlee/playing-hard-exploration-games-by-watching-youtube/master.svg?label=isort)](https://travis-ci.com/seungjaeryanlee/playing-hard-exploration-games-by-watching-youtube)

[![numpydoc Docstring Style](https://img.shields.io/badge/docstring-numpydoc-blue.svg)](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-blue.svg)](.pre-commit-config.yaml)

A PyTorch implementation of Playing Hard Exploration Games by Watching YouTube (Aytar et al., 2018)

## Installation

First, install FFMPEG for `skvideo.io`.

```
sudo apt-get install ffmpeg
```

Then, install Python packages using `requirements.txt`.

```
pip install -r requirements.txt
```

## Download Video and Audio

To train TDC and CMC, you need videos and audios. Because of their size, they are not included in the repository.

Running `get_av.sh` should automatically download and extract video and audio to `data/`. You can also download them from [my public Google Drive folder](https://drive.google.com/drive/folders/18EomHIBO9nUbBvllw0uG6SojlR7Ap0hg?usp=sharing).

## Download Pretrained Embedders

I have saved TDC and CMC networks that have been trained with videos above. Because of their size, they are not included in the repository.

Running `get_pretrained_models.sh` should automatically download saved parameters to `saves/`. You can also download them from [my public Google Drive folder](https://drive.google.com/drive/folders/18EomHIBO9nUbBvllw0uG6SojlR7Ap0hg?usp=sharing).
