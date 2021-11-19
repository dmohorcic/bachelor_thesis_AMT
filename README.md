# Bachelor thesis: Automatic music transcription of piano music with convolutional neural networks

## Abstract

In this thesis we explore the problem of automatic music transcription using deep neural networks, more specific convolutional neural networks. Automatic music transcription is a task of writing the sheet music from musical recordings. We analysed previous studies and found that there was a lack of research about the size and the shape of architecture of deep models. We explored the performance of four different architectures of convolutional neural networks on the piano recordings dataset MAPS, which is a common benchmark for learning automatic music transcription. We also compared two different normalization techniques for spectrograms: standardization and the logarithmic compression. We found out that the performance of transcription is highly correlated with the higher number of convolutional layers. Transcription is also 10% more successful with logarithmic compression instead of standardization.

The thesis is in file [Avtomatska transkripcija klavirske glasbe s konvolucijskimi nevronskimi mrežami](Avtomatska transkripcija klavirske glasbe s konvolucijskimi nevronskimi mrežami.pdf). The presentation for the thesis defense is in file [zagovor](zagovor.pptx).

## About this repository

The bachelor thesis was written in LaTeX and is in folder [bachelorsThesis](bachelorsThesis).
The main experiments were done in file [diploma.ipynb](diploma.ipynb) with [graphics.ipynb](graphics.ipynb) and [times.ipynb](times.ipynb) as side files.
The code for running longer experiments and training is in folder [src](src). It can be run with `python main.py --help` for further instructions.
Informational results of training are in folder [results](results). Folder [tt_files](tt_files) includes TrainTest split files.
The trained models are in folder [cnnModels](cnnModels), which include earlier models, [expModels](expModels), which include trained models, and [hteModels](hteModels), which are models trained on transposed music.