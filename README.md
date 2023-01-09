# StarGAN-Voice-Conversion-2
samira soleymani is a masters student from tehran university
40014140111018 student number
digital signal processing crourse
professor DR:mahde Eslami
https://github.com/SamuelBroughton/StarGAN-Voice-Conversion-2


Non-parallel multi-domain voice conversion (VC) is a technique
for learning mappings among multiple domains without
relying on parallel data. This is important but challenging owing
to the requirement of learning multiple mappings and the nonavailability
of explicit supervision. Recently, StarGAN-VC has
garnered attention owing to its ability to solve this problem only
using a single generator. However, there is still a gap between
real and converted speech. To bridge this gap, we rethink conditional
methods of StarGAN-VC, which are key components
for achieving non-parallel multi-domain VC in a single model,
and propose an improved variant called StarGAN-VC2. Particularly,
we rethink conditional methods in two aspects: training
objectives and network architectures. For the former, we propose
a source-and-target conditional adversarial loss that allows
all source domain data to be convertible to the target domain
data. For the latter, we introduce a modulation-based conditional
method that can transform the modulation of the acoustic
feature in a domain-specific manner. We evaluated our methods
on non-parallel multi-speaker VC. An objective evaluation
demonstrates that our proposed methods improve speech quality
in terms of both global and local structure measures. Furthermore,
a subjective evaluation shows that StarGAN-VC2 outperforms
StarGAN-VC in terms of naturalness and speaker similarity.

A pytorch implementation based on: StarGAN-VC2: https://arxiv.org/pdf/1907.12279.pdf.

* Uses source and target domain codes in D but not G as I found better quality output 
* Doesnt make use of PS in G.

# Installation

**Tested on Python version 3.6.2 in a linux VM environment**

Recommended to use a linux environment - not tested for mac or windows OS 

## Python

* Create a new environment using Anaconda
```shell script
conda create -n stargan-vc python=3.6.2
```
* Install conda dependencies
```shell script
conda install pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.1 -c pytorch
conda install pillow=5.4.1
conda install -c conda-forge librosa=0.6.1
conda install -c conda-forge tqdm=4.43.0
```

* Intall dependencies not available through conda using pip
```shell script
pip install pyworld=0.2.8
pip install mcd=0.4
```

**NB:** For mac users who cannot install pyworld see: https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder

## Libraries

* Install binaries
  * SoX: https://sourceforge.net/projects/sox/files/sox/14.4.2/ 
  * libsndfile: http://linuxfromscratch.org/blfs/view/svn/multimedia/libsndfile.html
  * yasm: http://www.linuxfromscratch.org/blfs/view/svn/general/yasm.html
  * ffmpeg: https://ffmpeg.org/download.html
  * libav: https://libav.org/download/

# Usage

## Download Datasets

* [VCTK](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)
* [VCC2018](https://datashare.is.ed.ac.uk/handle/10283/3061?show=full)

Example with VCTK:

```shell script
mkdir ../data/VCTK-Data
wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2651/VCTK-Corpus.zip?sequence=2&isAllowed=y
unzip VCTK-Corpus.zip -d ../data/VCTK-Data
```

If the downloaded VCTK is in tar.gz, run this:

```shell script
tar -xzvf VCTK-Corpus.tar.gz -C ../data/VCTK-Data
```

## Preprocessing data

We will use Mel-Cepstral coefficients(MCEPs) here.

Example script for VCTK data which we can resample to 22.05kHz. The VCTK dataset is not split into train and test wavs, so we perform a data split.

```shell script
# VCTK-Data
python preprocess.py --perform_data_split y \
                     --resample_rate 22050 \
                     --origin_wavpath ../data/VCTK-Data/VCTK-Corpus/wav48 \
                     --target_wavpath ../data/VCTK-Data/VCTK-Corpus/wav22 \
                     --mc_dir_train ../data/VCTK-Data/mc/train \
                     --mc_dir_test ../data/VCTK-Data/mc/test \
                     --speakers p229 p232 p236 p243
```

Example Script for VCC2018 data which is already seperated into train and test wav folders and is already at 22.05kHz.

```shell script
# VCC2018-Data
python preprocess.py --perform_data_splt n \
                     --target_wav_path_train ../data/VCC2018-Data/VCC2018-Corpus/wav22_train \
                     --target_wav_path_eval ../data/VCC2018-Data/VCC2018-Corpus/wav22_eval \
                     --mc_dir_train ../data/VCC2018-Data/mc/train \
                     --mc_dir_test ../data/VCC2018-Data/mc/test \
                     --speakers VCC2SF1 VCC2SF2 VCC2SM1 VCC2SM2
```

# Training

Example script:

```shell script
# example with VCTK
python main.py --train_data_dir ../data/VCTK-Data/mc/train \
               --test_data_dir ../data/VCTK-Data/mc/test \
               --wav_dir ../data/VCTK-Data/VCTK-Corpus/wav22 \
               --model_save_dir ./models/experiment_name \
               --sample_dir ./samples/experiment_name \
               --speakers p229 p232 p236 p243
```

If you encounter an error such as:

```shell script
ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found
```

You may need to export export LD_LIBRARY_PATH: (See [Stack Overflow](https://stackoverflow.com/questions/49875588/importerror-lib64-libstdc-so-6-version-cxxabi-1-3-9-not-found))

```shell script
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/<PATH>/<TO>/<YOUR>/.conda/envs/<ENV>/lib/
```

## Conversion

For example: restore model at step 90000 and specify the speakers

```shell script
# example with VCTK
python convert.py --resume_model 90000 \
                  --speakers p229 p232 p236 p243 \
                  --train_data_dir ../data/VCTK-Data/mc/train/ \
                  --test_data_dir ../data/VCTK-Data/mc/test/ \
                  --wav_dir ../data/VCTK-Data/VCTK-Corpus/wav22 \
                  --model_save_dir ./models/experiment_name \
                  --convert_dir ./converted/experiment_name
```

# TODO:
- [x] Include converted samples
- [x] Include s-t loss like original paper (NB: not exactly the same, see top of this README)
