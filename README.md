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

## Data Availability

The final MarineSet data publication is currently **work in progress**.

The curated MarineSet data will be made publicly available through Zenodo. A DOI and direct download instructions will be added to this repository once the Zenodo record has been published.

> **Data release: Work in progress**
>
> The MarineSet data are currently being prepared for publication on Zenodo.
> The Zenodo DOI will be added to this README once the data release is available.
>
> **Zenodo DOI:** *To be added*

Until the Zenodo release is available, the repository contains the code used to construct MarineSet and the code required to retrieve selected recordings from the underlying NOAA data source (for instructions see below).

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

The dataset originates from:

> Irfan, M., Jiangbin, Z., Ali, S., Iqbal, M., Masood, Z., & Hamid, U. (2021). **DeepShip: An underwater acoustic benchmark dataset and a separable convolution based autoencoder for classification**. *Expert Systems with Applications, 183*, 115270. https://doi.org/10.1016/j.eswa.2021.115270

The DeepShip dataset consists of 47 hours and 4 minutes of real-world underwater recordings from 265 ships belonging to the four ship categories listed above. The recordings were collected using infrastructure from Ocean Networks Canada.

The original paper and dataset information can be found at:

- [DeepShip paper](https://doi.org/10.1016/j.eswa.2021.115270)
- [DeepShip GitHub repository](https://github.com/irfankamboh/DeepShip)

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

The ShipsEar dataset originates from:

> Santos-Domínguez, D., Torres-Guijarro, S., Cardenal-López, A., & Peña-Giménez, A. (2016). **ShipsEar: An underwater vessel noise database**. *Applied Acoustics, 113*, 64–69. https://doi.org/10.1016/j.apacoust.2016.06.008

ShipsEar is an underwater vessel noise database containing recordings of different vessel types and background noise collected in the Atlantic Ocean near the coast of Spain.

The original paper can be found at:

- [ShipsEar paper](https://doi.org/10.1016/j.apacoust.2016.06.008)

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

The data originates from:

> Palmer, K., et al. (2025). **A Public Dataset of Annotated *Orcinus orca* Acoustic Signals for Detection and Ecotype Classification**. *Scientific Data*. https://doi.org/10.1038/s41597-025-05281-5

The dataset contains annotated acoustic recordings of killer whales (*Orcinus orca*), as well as recordings and annotations of other marine mammal species and abiotic sounds. The data were compiled from multiple sources and recording deployments across Alaska, British Columbia, and Washington.

The dataset is publicly available through the NOAA National Centers for Environmental Information (NCEI):

- [DCLDE 2026 dataset](https://doi.org/10.25921/15ey-mh50)

The associated dataset publication can be found at:

- [A Public Dataset of Annotated *Orcinus orca* Acoustic Signals for Detection and Ecotype Classification](https://doi.org/10.1038/s41597-025-05281-5)

For the DCLDE 2026 classification, only the calls are extracted from the complete dataset. The extraction is performed by defining the call from the annotations and extract 5 second windows. The call is randomly placed within the window, ensuring the complete call is present. The windowed data can be downloaded using: 

```text
MarineNet/classification_data/DCLDE2026_download.py
```

Since multiple calls from various species can be present within the extracted 5 second windows, this classfication is referred to as a multilabel setting. A window is considered multilabel if the intersection of the window length and annotation length / min( window length, annotation length) is greater than 0.3. Each annotation is treated individually. 

Next, the data is split based on a chronological split similar to the Deepship split. Here, ~80\% of the data was used for training and the remaining ~20\% for testing. The time boundary was found at 2013-11-22 13:36:00.

The DCLDE 2026 split is created using:

```text
MarineNet/classification_data/DCLDE2026_split.py
```

The script reads the window metadata and reconstructs the recording datetime from the original recording path.

The important distinction is that the split is performed at the **recording level**, rather than independently for individual windows.

This means that windows originating from the same recording are kept together in the same split. Consequently, a recording does not contribute windows to both the training and test sets.

The corresponding audio windows are organized into:

```text
train/
test/
```

This chronological recording-level split reduces the possibility of temporal leakage between training and testing.

## Dataset split summary

| Dataset | Task | Training split | Validation split | Test split | Split strategy |
|---|---|---|---|---|---|
| DeepShip | Ship type | Before 2017-12-01 | — | On/after 2017-12-01 | Temporal |
| ShipsEar | Ship type | `shipsEar_train.csv` | `shipsEar_val.csv` | `shipsEar_test.csv` | Stratified split |
| Watkins | Marine mammal calls | BEANS train split | BEANS validation split | BEANS test split | Predefined split |
| DCLDE 2026 | Marine mammal calls | Before 2013-11-22 12:36:00 | — | After 2013-11-22 13:36:00 | Temporal |

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

## Pretrained HK-Means Model

The pretrained [HK-Means model](https://huggingface.co/hildehummel/HKmeans) used in this study is available on Hugging Face.

The [Conformer model](https://huggingface.co/hildehummel/Conformer_UATR) used to generate the embeddings for training the HK-Means model is also available on Hugging Face.

## Acknowledgements

MarineSet uses acoustic data originating from the NOAA Passive Bioacoustic Data Collection.

We thank the organizations and researchers responsible for collecting, maintaining, and making the underlying acoustic and AIS data available.

## License

Please refer to the license and usage conditions of the original data sources before redistributing or using the underlying NOAA recordings.

The code in this repository is provided for research purposes.

## Contact

For questions regarding MarineSet or the associated curation code, please open an issue in this repository or contact the authors.
