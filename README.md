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

## Data Availability

The MarineSet data will be publicly released through Zenodo.

> **Data release: Work in progress**
>
> The MarineSet data release on Zenodo is currently being prepared.
> A Zenodo DOI will be added to this repository once the release is available.
>
> **Zenodo DOI:** *To be added*

The repository nevertheless contains the code required to reproduce the data selection and, where the necessary source data and credentials are available, download the selected audio from the original NOAA data source.

### Important: Do not retrain the HK-means model unless necessary

The hierarchical K-means (HK-means) curation pipeline is computationally expensive and time-consuming. The current implementation downloads large amounts of audio, extracts Conformer embeddings, and subsequently trains a hierarchical K-means model with four levels using:

```text
[6000, 400, 40, 10]
```

clusters. The pipeline processes the data in batches, but retraining the HK-means model still requires substantial computational resources and can take a considerable amount of time.

**Therefore, retraining HK-means is not recommended simply to obtain the released MarineSet samples.**

Once the Zenodo release is available, users should download the already-curated MarineSet data directly from Zenodo rather than rerunning the complete curation pipeline.

The curation code is provided primarily for:

- understanding how MarineSet was constructed;
- reproducing the curation procedure;
- investigating alternative curation strategies;
- extending the dataset with additional source data; and
- reproducing the experiments described in the associated publication.

The HK-means configuration used for the MarineSet curation is stored in:

```text
curationCode/ClusterCuration/config.yaml
```

and contains the hierarchy, sampling parameters, and total number of samples used during curation.

---

## Downloading MarineSet Audio

### Recommended approach: download the Zenodo release

Once the Zenodo publication is available, the recommended procedure is:

1. Download the MarineSet archive from Zenodo.
2. Extract the archive locally.
3. Use the provided metadata files to identify recordings, hydrophones, timestamps, and other associated information.
4. Use the extracted audio directly for downstream experiments.

The Zenodo DOI and exact file structure will be added to this README after the data publication has been finalized.

**Zenodo DOI:** *To be added*

---

### Reproducing the download using the repository code

Before the Zenodo release is available, the repository provides code for downloading the selected source recordings directly from the NOAA Passive Bioacoustic Google Cloud Storage bucket.

The relevant code is located in:

```text
curationCode/
├── AISCuration/
└── ClusterCuration/
```

The two curation approaches use slightly different procedures.

### 1. AIS-based MarineSet selection

The AIS-based curation code is located in:

```text
curationCode/AISCuration/
```

The main entry point is:

```text
curationCode/AISCuration/main.py
```

The pipeline consists of two main stages:

1. AIS-based selection of suitable recording periods.
2. Extraction and download of the corresponding audio.

The configuration is stored in:

```text
curationCode/AISCuration/config.yaml
```

The default configuration specifies the NOAA Google Cloud Storage bucket:

```yaml
cloud:
  bucket_name: noaa-passive-bioacoustic
```

The pipeline also uses:

```text
Administration/AISInformation.xlsx
```

and the AIS alignment/selection information contained in the `AISCuration` directory.

#### Step 1: Clone the repository

```bash
git clone https://github.com/hildeingvildhummel/MarineSet.git
cd MarineSet
```

#### Step 2: Install the required dependencies

The AIS download code uses Google Cloud Storage and therefore requires the Google Cloud Python libraries, in addition to the packages used by the curation code.

At minimum, the Google Cloud Storage functionality requires:

```bash
pip install google-cloud-storage
```

Additional dependencies used by the repository include packages such as:

```bash
pip install numpy pandas pyyaml soundfile
```

Depending on which parts of the pipeline are executed, additional dependencies may be required.

#### Step 3: Configure Google Cloud authentication

The download code uses the Google Cloud Storage Python client:

```python
from google.cloud import storage
```

and accesses the bucket:

```text
noaa-passive-bioacoustic
```

Google Cloud authentication therefore needs to be configured before running the download pipeline.

The repository's `main.py` also points users to the Google Cloud authentication documentation:

```text
https://googleapis.dev/python/google-api-core/latest/auth.html
```

Authentication is required to access the source data. The repository does **not** contain credentials.

#### Step 4: Check the AIS configuration

Open:

```text
curationCode/AISCuration/config.yaml
```

The default configuration contains:

```yaml
curation:
  run: false
  t: 250
  ais_folder: AISalignment/

paths:
  info_excel: Administration/AISInformation.xlsx
  selection_dir: Selection
  temp_dir: temp
  output_dir: AIS

cloud:
  bucket_name: noaa-passive-bioacoustic
```

The `run` parameter determines whether the AIS curation is rerun.

If an existing curated AIS selection is available, the pipeline can reuse it. If AIS curation needs to be performed, it can be forced using:

```bash
python main.py --config config.yaml --run-curation
```

The main script first creates or loads the curated AIS selection and then passes this selection to the audio extraction stage.

#### Step 5: Download the selected audio

The audio extraction is performed by the `AISAudioExtractor` used in:

```text
curationCode/AISCuration/AISSelection.py
```

The extractor uses the curated AIS selection together with the recording information in:

```text
Administration/AISInformation.xlsx
```

to identify the corresponding source recordings.

The selected audio is downloaded from the NOAA Google Cloud Storage bucket into the configured temporary/output directories.

The download therefore does **not** require downloading the entire NOAA archive. The purpose of the selection step is to identify only the recordings corresponding to the curated AIS samples.

---

## Downloading the HK-means-curated MarineSet samples

The cluster-based curation code is located in:

```text
curationCode/ClusterCuration/
```

The directory contains the following main stages:

```text
ClusterCuration/
├── Conformer_Embeddings.py
├── Encoder.py
├── AudioCuration_TrainHKmeans.py
├── HierarchicalKMeans.py
├── AudioCuration_Sampling.py
├── Download_Selection.py
├── GoogleCloudConnection.py
├── hierarchical_sampling.py
├── kmeans.py
└── main.py
```

The complete pipeline implemented in `main.py` consists of:

```text
NOAA audio
    ↓
Download source recordings
    ↓
Conformer embedding extraction
    ↓
Hierarchical K-means training
    ↓
Sampling from HK-means
    ↓
Download selected audio
```

The current configuration uses:

```yaml
audio:
  samplerate: 16000
  sample_size: 10

hkmeans:
  n_clusters: [6000, 400, 40, 10]
  n_levels: 4
  sample_sizes: [2200, 8, 5, 2]
  N_total: 731447
```

### Do not run the complete pipeline just to download MarineSet

Running:

```bash
python main.py --config config.yaml --dest <destination> --save <model_name>
```

runs the complete cluster-based pipeline.

In particular, the script contains a stage that trains HK-means:

```text
TRAIN HIERARCHICAL KMEANS
```

and subsequently performs resampling using the trained model.

This is **not recommended for ordinary MarineSet users**, because it requires downloading source audio, extracting embeddings, and retraining the hierarchical clustering model.

Instead, users should use the precomputed MarineSet release once it is available through Zenodo.

---

## How the repository downloads individual selected recordings

The lower-level download functionality is implemented in:

```text
curationCode/ClusterCuration/GoogleCloudConnection.py
```

This module provides functions for:

- obtaining the available files in a Google Cloud Storage folder;
- downloading an individual file; and
- downloading files from a source folder.

The NOAA bucket used by the code is:

```text
noaa-passive-bioacoustic
```

The `download_file()` function downloads a specific object from the bucket:

```python
download_file(
    "noaa-passive-bioacoustic",
    source_file,
    destination_file_name
)
```

The `download_folder()` function can download files belonging to a particular source folder.

The cluster-based pipeline constructs source folders from the AIS information and then downloads the corresponding audio before processing it.

The `Download_Selection.py` module contains the higher-level logic for converting the selected timestamps into the corresponding NOAA audio files. In particular, `sampled_curation()`:

1. reads the selected timestamps from the curation output;
2. identifies the corresponding hydrophone and source path;
3. queries the NOAA Google Cloud Storage bucket;
4. determines which source audio file contains the selected timestamp;
5. extracts the relevant temporal section; and
6. downloads the corresponding source file.

The code uses 10-second sections when extracting the selected audio:

```python
audio, sr = read_audio_section(
    file,
    start_secs,
    start_secs + 10
)
```

This allows the curated selections to be mapped back to the original NOAA recordings without requiring the complete source archive to be downloaded.

---

## Reproducing the HK-means selection from precomputed models

If the pre-trained HK-means model and the associated intermediate files are available, the final sampling stage can be reproduced without retraining the HK-means model.

The relevant code is:

```text
curationCode/ClusterCuration/AudioCuration_Sampling.py
```

and:

```text
curationCode/ClusterCuration/HierarchicalKMeans.py
```

The intended workflow is:

```text
Precomputed Conformer embeddings
        +
Pretrained HK-means model
        ↓
HK-means sampling
        ↓
MarineSet selection
        ↓
Download selected NOAA recordings
```

This is substantially preferable to retraining the HK-means model.

However, the exact intermediate files required for this procedure are part of the internal curation workflow and will be superseded for normal users by the finalized Zenodo release.

---

## Source Data

The audio used to construct MarineSet originates from the NOAA Passive Bioacoustic Data Collection and is accessed in the repository through the Google Cloud Storage bucket:

```text
noaa-passive-bioacoustic
```

The repository therefore contains code for accessing the source data, but users should distinguish between:

1. **the original NOAA source data**;
2. **the intermediate curation products**, such as AIS selections, embeddings, and HK-means models; and
3. **the final curated MarineSet dataset**.

MarineSet users interested only in using the dataset should download the **final curated dataset** from Zenodo once the release becomes available.

---

## Downstream Dataset Splits

MarineSet is evaluated together with existing labeled datasets for both ship-type classification and marine mammal call classification.

### Ship-type classification

The ship-type datasets are:

- **DeepShip**
- **ShipsEar**

### DeepShip

DeepShip contains four ship categories:

- Cargo
- Passengership
- Tanker
- Tug

The repository creates a temporal train/test split based on the recording date.

The split boundary is:

```text
2017-12-01
```

Recordings before this date are assigned to the training set, while recordings from this date onward are assigned to the test set.

The implementation can be found in:

```text
MarineNet/classification_data/ship_data.py
```

This is therefore a **date-based split**, rather than a random sample-level split.

### ShipsEar

ShipsEar uses explicit train, validation, and test split files provided in the repository:

```text
MarineNet/classification_data/
├── shipsEar_train.csv
├── shipsEar_val.csv
└── shipsEar_test.csv
```

These files define the samples used for training, validation, and testing.

---

## Marine Mammal Call Classification

The marine mammal datasets are:

- **Watkins Marine Mammal Sound Database**
- **DCLDE 2026**

### Watkins Marine Mammal Sound Database

The Watkins split follows the split used by **BEANS (The Benchmark of Animal Sounds)**.

The BEANS paper is:

> Hagiwara, M., Hoffman, B., Liu, J.-Y., Cusimano, M., Effenberger, F., & Zacarian, K. (2022). *BEANS: The Benchmark of Animal Sounds*. arXiv:2210.12300.

The BEANS benchmark uses the Watkins Marine Mammal Sound Database as a classification dataset and creates a **6:2:2 train/validation/test split with stratification**. The paper reports 1,017 training samples, 339 validation samples, and 339 test samples across 31 labels. :contentReference[oaicite:2]{index=2}

The corresponding split is implemented in:

```text
MarineNet/classification_data/Watkins.py
```

The paper is available at:

```text
https://arxiv.org/abs/2210.12300
```

### DCLDE 2026

The DCLDE 2026 dataset is split chronologically at the recording level.

The repository first orders recordings chronologically and then assigns approximately 80% of the available windows to training and the remaining approximately 20% to testing.

Importantly, the split boundary is determined at the **recording level**, meaning that individual recordings are kept entirely within either the training or testing set.

The split can be reproduced using:

```text
MarineNet/classification_data/DCLDE2026_split.py
```

The resulting files are:

```text
train.csv
test.csv
```

and the corresponding windows are organized into:

```text
train/
test/
```

This prevents windows from the same original recording from being distributed across both training and testing sets.

---

## Citation

If you use MarineSet, please cite the associated publication:

```bibtex
@article{MarineSet,
  title   = {To be added},
  author  = {Hummel, Hilde},
  journal = {To be added},
  year    = {2026},
  doi     = {To be added}
}
```

The final citation will be added once the associated publication and Zenodo release are available.

### Watkins / BEANS

If you use the Watkins split, please also cite the BEANS paper:

```bibtex
@article{hagiwara2022beans,
  title         = {BEANS: The Benchmark of Animal Sounds},
  author        = {Hagiwara, Masato and Hoffman, Benjamin and Liu, Jen-Yu and Cusimano, Maddie and Effenberger, Felix and Zacarian, Katie},
  journal       = {arXiv preprint arXiv:2210.12300},
  year          = {2022},
  doi           = {10.48550/arXiv.2210.12300}
}
```
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
