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

The directory contains code for:

- AIS-based data curation and alignment;
- AIS-based audio selection and downloading;
- Conformer embedding extraction;
- hierarchical K-means (HK-means) clustering;
- cluster-based audio sampling;
- downloading selected audio from the NOAA Google Cloud Storage bucket; and
- administrative and data-processing steps used during the construction of MarineSet.

### AIS-based curation

The AIS curation code is located in:

```text
curationCode/AISCuration/
```

The directory contains:

```text
AISCuration/
├── AISalignment/
├── Administration/
├── Selection/
├── AISSelection.py
├── AISdata.py
├── GoogleCloudConnection.py
├── config.yaml
└── main.py
```

The AIS-based pipeline uses AIS information to identify recordings associated with vessel activity and subsequently retrieves the corresponding acoustic recordings.

The main entry point is:

```text
curationCode/AISCuration/main.py
```

The pipeline consists of two main stages:

1. AIS-based curation and selection.
2. Retrieval of the corresponding acoustic recordings from the NOAA Google Cloud Storage bucket.

The `main.py` script loads its settings from:

```text
curationCode/AISCuration/config.yaml
```

The configuration specifies, among other things, the AIS curation parameters, input/output directories, and the NOAA Google Cloud Storage bucket.

Before running the AIS pipeline, Google Cloud authentication needs to be configured. The repository refers to the Google Cloud authentication instructions in the `main.py` script.

The AIS pipeline can be run from the `AISCuration` directory using:

```bash
python main.py
```

Alternatively, a different configuration file can be supplied:

```bash
python main.py --config <path_to_config.yaml>
```

The AIS curation parameter `t` can also be overridden from the command line:

```bash
python main.py --config <path_to_config.yaml> --t <value>
```

To explicitly rerun the AIS curation rather than using an existing curated AIS selection:

```bash
python main.py --config <path_to_config.yaml> --run-curation
```

The pipeline stores the resulting AIS selection as a pickle file and subsequently passes this selection to the audio extraction stage.

### Downloading NOAA audio using the AIS selection

The NOAA audio is accessed through Google Cloud Storage.

The relevant functionality is implemented in:

```text
curationCode/AISCuration/GoogleCloudConnection.py
```

The code uses the NOAA Passive Bioacoustic Google Cloud project and provides functions for:

- listing files in a Google Cloud Storage folder;
- obtaining the names of available `.flac` and `.wav` files;
- downloading an individual file; and
- downloading a complete folder.

The bucket used by the code is:

```text
noaa-passive-bioacoustic
```

The higher-level audio extraction is implemented in:

```text
curationCode/AISCuration/AISSelection.py
```

The `AISAudioExtractor` uses the curated AIS selection together with the recording information stored in the administrative data to identify the corresponding NOAA recordings.

The overall AIS-based workflow is therefore:

```text
AIS data
   │
   ▼
AIS alignment
   │
   ▼
AIS-based selection
   │
   ▼
Selected recording locations/timestamps
   │
   ▼
NOAA Google Cloud Storage
   │
   ▼
Selected acoustic recordings
```

This approach avoids downloading the entire NOAA archive. Instead, the pipeline identifies the recordings required by the AIS-based selection and retrieves those recordings from the NOAA cloud storage.

### Cluster-based curation

The cluster-based curation code is located in:

```text
curationCode/ClusterCuration/
```

The directory contains:

```text
ClusterCuration/
├── Administration/
├── AudioCuration_Sampling.py
├── AudioCuration_TrainHKmeans.py
├── Conformer_Embeddings.py
├── Download_Selection.py
├── Encoder.py
├── GoogleCloudConnection.py
├── HierarchicalKMeans.py
├── clusters.py
├── config.yaml
├── hierarchical_sampling.py
├── kmeans.py
├── main.py
└── support.py
```

The cluster-based pipeline consists of:

```text
NOAA acoustic recordings
        │
        ▼
Conformer embeddings
        │
        ▼
Hierarchical K-means
        │
        ▼
Cluster-based sampling
        │
        ▼
Selected recordings
        │
        ▼
NOAA audio download
```

The main entry point is:

```text
curationCode/ClusterCuration/main.py
```

The configuration is specified in:

```text
curationCode/ClusterCuration/config.yaml
```

The configuration contains the parameters required for the audio processing, embedding extraction, HK-means hierarchy, sampling procedure, and paths to intermediate files.

### Important: HK-means retraining is not recommended

**Retraining the hierarchical K-means model is computationally expensive and time-consuming and is not recommended if the goal is simply to obtain or use MarineSet.**

The cluster-based curation pipeline requires processing a large amount of acoustic data, extracting Conformer embeddings, and training a hierarchical K-means model. The HK-means hierarchy contains multiple clustering levels, making the training substantially more expensive than applying an already-trained model.

The relevant training code is:

```text
curationCode/ClusterCuration/AudioCuration_TrainHKmeans.py
curationCode/ClusterCuration/HierarchicalKMeans.py
```

The sampling stage is implemented separately in:

```text
curationCode/ClusterCuration/AudioCuration_Sampling.py
```

For this reason, users should **not retrain HK-means merely to download or use the MarineSet samples**.

The curation code is provided primarily to:

- document the MarineSet curation procedure;
- reproduce the curation methodology;
- investigate the effect of alternative sampling strategies;
- extend the curation procedure to additional data; and
- support methodological research on large-scale underwater acoustic data curation.

Once the MarineSet data are released on Zenodo, the recommended approach for users is to download the **pre-curated dataset** rather than rerunning the complete HK-means pipeline.

## Data Availability

The final MarineSet data publication is currently **work in progress**.

The curated MarineSet data will be made publicly available through Zenodo. A DOI and direct download instructions will be added to this repository once the Zenodo record has been published.

> **Data release: Work in progress**
>
> The MarineSet data are currently being prepared for publication on Zenodo.
> The Zenodo DOI will be added to this README once the data release is available.
>
> **Zenodo DOI:** *To be added*

Until the Zenodo release is available, the repository contains the code used to construct MarineSet and the code required to retrieve selected recordings from the underlying NOAA data source.

## Reproducing the MarineSet data download

The repository provides two related routes for obtaining acoustic data:

1. **AIS-based selection and download**, using the AIS curation pipeline.
2. **HK-means-based selection and download**, using the cluster curation pipeline.

These should not be confused with directly downloading the final MarineSet release.

The final Zenodo release will contain the **already-curated MarineSet samples**. Re-running the complete curation pipeline is primarily intended for reproduction of the methodology rather than routine dataset access.

### Step 1: Clone the repository

Clone the repository:

```bash
git clone https://github.com/hildeingvildhummel/MarineSet.git
cd MarineSet
```

The curation code is located in:

```text
curationCode/
```

### Step 2: Set up Google Cloud authentication

The MarineSet curation code retrieves source audio from the NOAA Passive Bioacoustic Google Cloud Storage bucket.

The relevant bucket is:

```text
noaa-passive-bioacoustic
```

Google Cloud authentication must therefore be configured before the download scripts are executed.

The AIS pipeline explicitly refers to the Google Cloud authentication instructions in `main.py`.

The repository does not contain authentication credentials. Users must configure their own Google Cloud authentication according to the requirements of the NOAA data source and Google Cloud Storage.

### Step 3: AIS-based download

If you want to reproduce the AIS-based selection, move to:

```bash
cd curationCode/AISCuration
```

The pipeline is controlled through:

```text
config.yaml
```

The default configuration can be used directly, or copied and modified for a different experiment.

Run:

```bash
python main.py
```

The script performs the following operations:

```text
1. Load the configuration
2. Load the AIS information
3. Create or load the curated AIS selection
4. Save the AIS selection
5. Initialize the AIS audio extractor
6. Identify the corresponding NOAA recordings
7. Download/process the selected recordings
```

If the AIS selection has already been generated, the existing selection can be reused rather than performing the AIS curation again.

To explicitly rerun the AIS curation:

```bash
python main.py --run-curation
```

The resulting acoustic recordings are written to the output directory specified in the configuration.

### Step 4: Direct interaction with the NOAA cloud storage

The lower-level Google Cloud functionality is implemented in:

```text
curationCode/AISCuration/GoogleCloudConnection.py
```

The same functionality is also used by the cluster-based curation code.

The module provides:

```python
get_file_names(...)
```

for identifying `.flac` and `.wav` files in a source folder, and:

```python
download_file(...)
```

for downloading a specific file.

A file can therefore be retrieved once its full Google Cloud Storage object path is known.

The code also provides:

```python
download_folder(...)
```

for downloading files belonging to a specified source folder.

The download functionality includes a date check and is designed around the NOAA recordings used for MarineSet.

### Step 5: Cluster-based download

The cluster-based download procedure is implemented in:

```text
curationCode/ClusterCuration/Download_Selection.py
```

This code connects the selected samples produced by the cluster-based curation procedure back to the original NOAA recordings.

The procedure identifies the source recording corresponding to a selected timestamp and retrieves the relevant acoustic data from Google Cloud Storage.

The relevant functionality includes:

```python
get_file_names(...)
download_file(...)
read_audio_section(...)
```

The audio is read from the original recording after identifying the appropriate source file and temporal location.

The cluster-based download workflow is therefore:

```text
HK-means-selected samples
        │
        ▼
Selected timestamps
        │
        ▼
Identify corresponding NOAA recording
        │
        ▼
Query NOAA Google Cloud Storage
        │
        ▼
Download source recording
        │
        ▼
Extract selected acoustic section
```

### Step 6: Running the complete cluster-cura­tion pipeline

The complete cluster-based pipeline can be launched using:

```bash
cd curationCode/ClusterCuration
python main.py --config config.yaml --dest <download_directory> --save <model_name>
```

The arguments are:

```text
--config    Path to the YAML configuration file
--dest      Local directory for downloaded data
--save      Name/path used for saving the HK-means model
```

However, **running this command is not recommended if your only goal is to obtain MarineSet**.

The complete pipeline includes the computationally expensive HK-means training stage. It may therefore require substantial computational resources and considerable processing time.

For normal dataset use, the preferred workflow is:

```text
Zenodo
  │
  ▼
Download pre-curated MarineSet
  │
  ▼
Use MarineSet for experiments
```

rather than:

```text
NOAA
  │
  ▼
Download large-scale source data
  │
  ▼
Extract Conformer embeddings
  │
  ▼
Train HK-means
  │
  ▼
Perform sampling
  │
  ▼
Download selected recordings
```

The second workflow is intended for reproducing or extending the curation procedure.

## MarineSet dataset statistics

MarineSet was constructed using two complementary curation strategies.

### AIS-based curation

The AIS-based curation resulted in:

- **968.2 hours** of acoustic data;
- **6,540 unique vessels**; and
- recordings from **28 hydrophones**.

### HK-means-based curation

The cluster-based curation resulted in:

- **2,031.8 hours** of acoustic data; and
- recordings from **43 hydrophones**.

The two curation strategies provide complementary approaches to selecting representative underwater acoustic recordings.

## Downstream datasets

The repository also contains the code used to prepare the labeled datasets used for downstream evaluation.

These datasets are divided into two categories:

### Ship-type classification

- DeepShip
- ShipsEar

### Marine mammal call classification

- Watkins Marine Mammal Sound Database
- DCLDE 2026

The corresponding scripts are located in:

```text
MarineNet/classification_data/
```

The directory currently contains:

```text
MarineNet/classification_data/
├── DCLDE2026_download.py
├── DCLDE2026_split.py
├── Watkins.py
├── ship_data.py
├── shipsEar_train.csv
├── shipsEar_val.csv
└── shipsEar_test.csv
```

## Ship-type dataset splits

### DeepShip

DeepShip contains four ship categories:

- Cargo
- Passengership
- Tanker
- Tug

The split implemented in:

```text
MarineNet/classification_data/ship_data.py
```

is based on the recording date.

The split boundary is:

```text
2017-12-01
```

Recordings before this date are assigned to the training set, while recordings on or after this date are assigned to the test set.

Thus, the DeepShip split is **temporal rather than random**.

This is important when comparing results, as the model is evaluated on recordings from a later period than those used for training.

### ShipsEar

ShipsEar uses explicit train, validation, and test split files:

```text
MarineNet/classification_data/shipsEar_train.csv
MarineNet/classification_data/shipsEar_val.csv
MarineNet/classification_data/shipsEar_test.csv
```

These files define the samples assigned to each split.

The repository also contains code in:

```text
MarineNet/classification_data/ship_data.py
```

for organizing the ShipsEar audio according to these splits.

## Marine mammal call dataset splits

### Watkins Marine Mammal Sound Database

The Watkins split used in this repository follows the split used by the **BEANS benchmark**.

The split originates from:

> Hagiwara et al., *BEANS: The Benchmark of Animal Sounds*, arXiv:2210.12300.

The BEANS benchmark uses the Watkins Marine Mammal Sound Database with a **6:2:2 train/validation/test split**, with stratification across labels.

Reference:

```text
https://arxiv.org/pdf/2210.12300
```

The corresponding split files are:

```text
annotations.train.csv
annotations.valid.csv
annotations.test.csv
```

The repository script:

```text
MarineNet/classification_data/Watkins.py
```

reads these annotation files and organizes the audio into:

```text
Data/watkins_split/
├── train/
├── val/
└── test/
```

Within each split, audio files are further organized by their class label.

The script therefore preserves the train/validation/test partition specified by the BEANS benchmark rather than generating a new random split.

If the Watkins dataset is used, the BEANS paper should be cited.

### DCLDE 2026

The DCLDE 2026 split is created using:

```text
MarineNet/classification_data/DCLDE2026_split.py
```

The script reads the window metadata and reconstructs the recording datetime from the original recording path.

Recordings are then sorted chronologically.

The split is designed to allocate approximately:

```text
80% → training
20% → testing
```

The important distinction is that the split is performed at the **recording level**, rather than independently for individual windows.

This means that windows originating from the same recording are kept together in the same split. Consequently, a recording does not contribute windows to both the training and test sets.

The resulting metadata files are:

```text
train.csv
test.csv
```

and the corresponding audio windows are organized into:

```text
train/
test/
```

This chronological recording-level split reduces the possibility of temporal leakage between training and testing.

## Dataset split summary

| Dataset | Task | Training split | Validation split | Test split | Split strategy |
|---|---|---|---|---|---|
| DeepShip | Ship type | Before 2017-12-01 | — | On/after 2017-12-01 | Temporal |
| ShipsEar | Ship type | `shipsEar_train.csv` | `shipsEar_val.csv` | `shipsEar_test.csv` | Predefined split |
| Watkins | Marine mammal calls | BEANS train split | BEANS validation split | BEANS test split | Stratified 6:2:2 |
| DCLDE 2026 | Marine mammal calls | ~80% | — | ~20% | Chronological, recording-level |

## MarineNet

The repository also contains **MarineNet**, a baseline model for underwater acoustic representation learning.

The MarineNet code is located in:

```text
MarineNet/
```

and contains:

```text
MarineNet/
├── classification_data/
├── support/
├── Classification.py
└── MarineNet.py
```

MarineNet is provided as a baseline for evaluating representations learned from the MarineSet data.

The main focus of this repository, however, is the **MarineSet dataset and the large-scale data curation methodology** rather than MarineNet itself.

## Publication

The associated publication describing MarineSet is currently in preparation.

> **Publication:** *To be added*

A full citation and DOI will be added once the publication is available.

## Citation

If you use MarineSet, please cite the associated MarineSet publication once it is available.

```bibtex
@article{MarineSet,
  title   = {To be added},
  author  = {Hummel, Hilde},
  journal = {To be added},
  year    = {2026},
  doi     = {To be added}
}
```

If you use the Watkins Marine Mammal Sound Database split provided in this repository, please also cite the BEANS benchmark:

```bibtex
@article{hagiwara2022beans,
  title         = {BEANS: The Benchmark of Animal Sounds},
  author        = {Hagiwara, Masato and Hoffman, Benjamin and Liu, Jen-Yu and Cusimano, Maddie and Effenberger, Felix and Zacarian, Katie},
  journal       = {arXiv preprint arXiv:2210.12300},
  year          = {2022},
  doi           = {10.48550/arXiv.2210.12300}
}
```

## Acknowledgements

MarineSet uses acoustic data originating from the NOAA Passive Bioacoustic Data Collection.

We thank the organizations and researchers responsible for collecting, maintaining, and making the underlying acoustic and AIS data available.

## License

Please refer to the license and usage conditions of the original data sources before redistributing or using the underlying NOAA recordings.

The code in this repository is provided for research purposes.

## Contact

For questions regarding MarineSet or the associated curation code, please open an issue in this repository or contact the authors.
