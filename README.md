# MarineSet

Research code and data for **MarineSet**, a curated dataset for large-scale machine learning in underwater acoustics.

MarineSet was developed to facilitate the development of general-purpose acoustic representations for **passive acoustic monitoring (PAM)**. The dataset is constructed from large-scale underwater acoustic recordings and uses automated curation based on both vessel activity and acoustic diversity.

This repository contains the code used for the MarineSet curation pipeline, associated metadata and samples, and code for training and evaluating the MarineNet baseline model.

---

## Overview

Passive acoustic monitoring systems generate large volumes of underwater acoustic recordings. These recordings contain information about vessels, marine mammals, and other components of the marine soundscape.

MarineSet addresses this challenge through **automated data curation**. The curation pipeline combines information from vessel tracking data with acoustic representations to select relevant and diverse recordings from large collections of underwater audio.

The repository contains two main curation approaches:

1. **AIS-based curation**, which selects recordings associated with vessel activity.
2. **Cluster-based acoustic curation**, which selects acoustically diverse recordings using learned audio representations and hierarchical clustering.

In addition, the repository contains the code used to train and evaluate **MarineNet**, a self-supervised underwater acoustic representation model.

---

# MarineSet

MarineSet contains underwater acoustic recordings collected from long-term hydrophone deployments.

The dataset is intended primarily for **self-supervised learning and representation learning**, but can also be used for downstream tasks such as:

- Ship type classification
- Ship-radiated noise classification
- Marine mammal call classification
- Passive acoustic monitoring
- Acoustic similarity and clustering
- Transfer learning
- General-purpose underwater acoustic representation learning

The dataset contains associated metadata, including information about recording windows, hydrophones, AIS information, and direction-of-arrival (DOA) information where available.

---

## Dataset Curation

MarineSet uses two complementary curation strategies.

### AIS-Based Curation

The AIS-based curation pipeline uses Automatic Identification System (AIS) information to identify underwater recordings associated with vessel activity.

The pipeline associates acoustic recordings with nearby vessel activity based on the spatial and temporal relationship between hydrophones and AIS positions.

The general workflow is:

```text
Underwater recordings
        |
        v
Hydrophone metadata
        |
        v
AIS vessel positions
        |
        v
Spatial and temporal alignment
        |
        v
Vessel-associated recordings
        |
        v
MarineSet samples
```

The AIS curation code is located in:

```text
curationCode/AISCuration/
```
The directory contains code for:

- AIS data retrieval
- AIS alignment
- Vessel selection
- Recording selection
- Data administration

---

### Acoustic Cluster-Based Curation

The second curation strategy uses the acoustic content of the recordings rather than relying on labels.

Audio recordings are converted into learned acoustic representations and subsequently clustered using hierarchical k-means (HK-means).

The clustering hierarchy is:

```text
6000
  |
  v
400
  |
  v
40
  |
  v
10
```

This hierarchical approach allows recordings to be grouped at multiple levels of acoustic similarity.

The general workflow is:
```text
Underwater recordings
        |
        v
Acoustic representation
        |
        v
Embedding extraction
        |
        v
Hierarchical k-means
        |
        v
Acoustic clusters
        |
        v
Hierarchical sampling
        |
        v
MarineSet samples
```
The cluster-based curation code is located in:
```text
curationCode/ClusterCuration/
```

The directory contains code for:

- AIS data retrieval
- AIS alignment
- Vessel selection
- Recording selection
- Data administration

---

# MarineSet Statistics

The current MarineSet curation resulted in two main collections.

| Curation strategy | Duration | Hydrophones | Additional information |
|---|---:|---:|---|
| AIS-based curation | 968.2 h | 28 | 6,540 unique vessels |
| Cluster-based curation | 2,031.8 h | 43 | Acoustically diverse recordings |

The AIS-based subset contains recordings associated with vessel activity identified using AIS information.

The cluster-based subset was selected based on acoustic diversity and does not require manual acoustic labels.

---

# MarineNet

The repository also contains the code for **MarineNet**, a self-supervised underwater acoustic representation model.

MarineNet uses a Wav2Vec 2.0 architecture initialized from speech-pretrained weights and further trained on underwater acoustic recordings using self-supervised learning.

The MarineNet code is located in:

```text
MarineNet/
```

The main components are:

```text
MarineNet/
|
├── MarineNet.py
|
├── Classification.py
|
├── classification_data/
|
└── support/
```

---

## Self-Supervised Pretraining

The self-supervised training code is located in:

```text
MarineNet/MarineNet.py
```

The model is initialized from:

```text
facebook/wav2vec2-base
```

and further pretrained using underwater acoustic recordings.

The audio is processed at a sampling rate of **16 kHz** and divided into fixed-length windows.

The resulting model can be used as a general-purpose feature extractor for underwater acoustic recordings.

---

## Downstream Evaluation

The downstream evaluation code is located in:

```text
MarineNet/Classification.py
```

The pretrained encoder is used to extract fixed-size acoustic representations.

The encoder is kept frozen while a linear classifier is trained on the extracted representations.

The general workflow is:

```text
Audio
  |
  v
MarineNet encoder
  |
  v
Frame-level representations
  |
  v
Temporal mean pooling
  |
  v
Fixed-size embedding
  |
  v
Linear classifier
  |
  v
Prediction
```

The classification code evaluates the representations using metrics including:

- Accuracy
- Mean Average Precision (mAP)
- ROC-AUC
- Weighted ROC-AUC
- Confusion matrices

---

# Downstream Datasets

MarineNet is evaluated on four downstream datasets:

### Ship Type Classification

- **DeepShip**
- **ShipsEar**

### Marine Mammal Call Classification

- **Watkins**
- **DCLDE 2026**

The split procedures used for these datasets are described below.

---

## DeepShip

DeepShip is used for **ship type classification**.

The dataset contains four ship classes:

```text
Cargo
Passengership
Tanker
Tug
```

The repository includes a script for creating a temporally separated train/test split:

```text
MarineNet/classification_data/ship_data.py
```

The split is based on the date encoded in the filename.

The temporal boundary is:

```text
1 December 2017
```

Recordings before this date are assigned to the **training set**, while recordings on or after this date are assigned to the **test set**.

This creates a temporal rather than random train/test split.

```text
DeepShip recordings
        |
        +--------------------+
        |                    |
        v                    v
Before 2017-12-01       2017-12-01 or later
        |                    |
        v                    v
     Training               Test
```

---

## ShipsEar

ShipsEar is also used for **ship type classification**.

The repository contains the corresponding split metadata:

```text
MarineNet/classification_data/shipsEar_train.csv
MarineNet/classification_data/shipsEar_val.csv
MarineNet/classification_data/shipsEar_test.csv
```

The dataset is divided into:

- **Training**
- **Validation**
- **Test**

The CSV files contain the metadata associated with each split.

The corresponding data preparation code is contained in:

```text
MarineNet/classification_data/ship_data.py
```

---

## Watkins

The Watkins dataset is used for **marine mammal call classification**.

The repository contains a script for organizing the dataset according to the supplied annotation splits:

```text
MarineNet/classification_data/Watkins.py
```

The original Watkins annotation files are used to construct:

```text
Training
Validation
Test
```

The corresponding annotation files are:

```text
annotations.train.csv
annotations.valid.csv
annotations.test.csv
```

The script organizes the audio files into class-specific directories within each split.

```text
Watkins
|
├── train
│   ├── class_1
│   ├── class_2
│   └── ...
|
├── val
│   ├── class_1
│   ├── class_2
│   └── ...
|
└── test
    ├── class_1
    ├── class_2
    └── ...
```

---

## DCLDE 2026

DCLDE 2026 is used for **marine mammal call classification**.

The split is generated using:

```text
MarineNet/classification_data/DCLDE2026_split.py
```

Unlike a random window-level split, the DCLDE 2026 data are divided **chronologically at the recording level**.

Recordings are first ordered chronologically. Approximately **80% of the recordings** are assigned to the training set, with the remaining recordings assigned to the test set.

```text
DCLDE 2026 recordings
        |
        v
Chronological ordering
        |
        v
Approximately 80%
        |
        +----------------------+
        |                      |
        v                      v
     Training                 Test
```

Importantly, complete recordings are kept within a single split. This prevents windows originating from the same recording from appearing in both training and test sets.

The split script also saves:

```text
train.csv
test.csv
```

containing the corresponding metadata.

---

# Dataset Splits Summary

| Dataset | Task | Training | Validation | Test | Split strategy |
|---|---|---|---|---|---|
| DeepShip | Ship type classification | ✓ | — | ✓ | Temporal split at 2017-12-01 |
| ShipsEar | Ship type classification | ✓ | ✓ | ✓ | Predefined train/validation/test split |
| Watkins | Marine mammal call classification | ✓ | ✓ | ✓ | Original dataset annotation splits |
| DCLDE 2026 | Marine mammal call classification | ✓ | — | ✓ | Chronological ~80/20 recording-level split |

---

# Installation

The code is written in Python and uses common scientific machine learning libraries.

The main dependencies include:

```bash
pip install torch
pip install transformers
pip install numpy
pip install pandas
pip install scikit-learn
pip install torchmetrics
pip install librosa
pip install soundfile
pip install tqdm
```

For GPU-based training, install a version of PyTorch compatible with the CUDA version available on your system.

---

# Running the Code

## MarineNet Pretraining

Update the data and output paths in:

```text
MarineNet/MarineNet.py
```

and run:

```bash
cd MarineNet
python MarineNet.py
```

---

## Downstream Classification

Update the dataset paths in:

```text
MarineNet/Classification.py
```

and run:

```bash
python Classification.py
```

The classification code extracts MarineNet representations and trains a linear classifier on the downstream dataset.

---

# Data Availability

The MarineSet dataset is currently being prepared for publication on **Zenodo**.

> **Data release: Work in progress**
>
> The full MarineSet dataset and associated metadata will be made available through Zenodo.
>
> **Zenodo DOI:** *To be added*

The full dataset is large and is therefore not hosted directly in this GitHub repository.

Once the Zenodo record has been published, the download link and DOI will be added here.

---

# Publication

A publication describing MarineSet and the associated data curation methodology is in preparation.

> **Publication:** *To be added*

```bibtex
% Publication citation will be added here.
%
% @article{...,
%   title     = {...},
%   author    = {...},
%   journal   = {...},
%   year      = {...},
%   doi       = {...}
% }
```

---

# Citation

If you use MarineSet, the associated curation code, or MarineNet in your research, please cite the forthcoming MarineSet publication.

The citation will be added to this section once the publication is available.

---

# Acknowledgements

MarineSet builds on data collected by a number of organizations and research initiatives. Please consult the accompanying dataset documentation and publication for detailed information about the original data sources and acknowledgements.

---

# License

Please see the repository license for information about permitted use, modification, and redistribution.

---

# Contact

For questions, issues, or suggestions regarding MarineSet or the accompanying code, please open an issue in this repository.

Repository:

https://github.com/hildeingvildhummel/MarineSet
