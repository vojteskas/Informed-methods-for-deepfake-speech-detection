# Informed methods for deepfake speech detection

**Author:** Vojtěch Staněk ([vojteskas](https://github.com/vojteskas)), istanek@fit.vut.cz

**Abstract:** Deepfake speech technology, which can create highly realistic fake audio, poses significant challenges, from enabling multi-million dollar scams to complicating legal evidence's reliability. This work introduces a novel method for detecting such deepfakes by leveraging bonafide speech samples. Unlike previous strategies, the approach uses trusted ground truth speech samples to identify spoofs, providing critical information that common methods lack. By comparing the bonafide samples with potentially manipulated ones, the aim is to effectively and reliably determine the authenticity of the speech.

## Repository structure

```
Repository
├── augmentation        <- contains various data augmentation techniques
├── classifiers         <- contains the classes for models
│   ├── differential        <- pair-input
│   └── single_input        <- single-input
├── datasets            <- contains Dataset classes (ASVspoof (2019, 2021), ASVspoof5, In-the-Wild, Morphing)
├── extractors          <- contains various feature extractors
├── feature_processors  <- contains pooling implementation (avg pool, MHFA, AASIST, SLS)
├── trainers            <- contains classes for training and evaluating models
├ README.md
├ common.py             <- common code, enums, maps, dataloaders
├ config.py             <- hardcoded config, paths, batch size
├ eval.py               <- script for evaluating trained model
├ parse_arguments.py    <- argument parsing script
├ requirements.txt      <- requirements to install in conda environment
├ scores_utils.py       <- functions for score analysis and evaluation
└ train_and_eval.py     <- main script for training and evaluating models
```

## Requirements

**Python 3.10**, possibly works with newer versions\
**PyTorch >2.2.0** including torchvision and torchaudio \
packages in `requirements.txt`

Simply install the required conda environment with:

```
# optional, create and activate conda env
# conda create -n diff_detection python=3.10
# conda activate diff_detection

# install required packages
# ⚠️ !!always refer to pytorch website https://pytorch.org/ for up-to-date command!! ⚠️
# conda install pytorch torchvision torchaudio cpuonly -c pytorch  # For CPU-only install
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia  # For GPU-enabled install

pip install -r requirements.txt
```

## Usage

Based on the use-case, use either `train_and_eval.py` or `eval.py` scripts with the following arguments:

```
usage: 
train_and_eval.py [-h/--help] (--local | --metacentrum | --sge) [--checkpoint CHECKPOINT] -d DATASET -e EXTRACTOR -p PROCESSOR -c CLASSIFIER [-a/--augment] [-ep NUM_EPOCHS]

Main script for training and evaluating the classifiers.

options:
  -h, --help            show this help message and exit

  --local               Flag for running locally.
  --metacentrum         Flag for running on metacentrum.
  --sge                 Flag for running on SGE

  --checkpoint CHECKPOINT
                        Path to a checkpoint to be loaded. If not specified, the model will be trained from scratch.

  -d DATASET, --dataset DATASET
                        Dataset to be used. See common.DATASETS for available datasets.

  -e EXTRACTOR, --extractor EXTRACTOR
                        Extractor to be used. See common.EXTRACTORS for available extractors.

  -p PROCESSOR, --processor PROCESSOR, --pooling PROCESSOR
                        Feature processor to be used. One of: AASIST, MHFA, Mean

  -c CLASSIFIER, --classifier CLASSIFIER
                        Classifier to be used. See common.CLASSIFIERS for available classifiers.

Optional arguments:
  -a, --augment         Flag for using data augmentation defined in augmentation/Augment.py

Classifier-specific arguments:
  -ep NUM_EPOCHS, --num_epochs NUM_EPOCHS
                        Number of epochs to train for.
```

## Publications

Here, the related publications will be listed.

Rohdin, J., Zhang, L., Oldřich, P., Staněk, V., Mihola, D., Peng, J., Stafylakis, T., Beveraki, D., Silnova, A., Brukner, J., Burget, L. (2024) *BUT systems and analyses for the ASVspoof 5 Challenge*. Proc. The Automatic Speaker Verification Spoofing Countermeasures Workshop (ASVspoof 2024), 24-31, DOI: [10.21437/ASVspoof.2024-4](https://www.isca-archive.org/asvspoof_2024/rohdin24_asvspoof.html)

## Contact

For any inquiries, questions or ask for help/explanation, contact me at istanek@fit.vut.cz.
