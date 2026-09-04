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

The dataset contains associated metadata, including information about recording windows, hydrophones, acoustic annotations, AIS information, and direction-of-arrival (DOA) information where available.

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
