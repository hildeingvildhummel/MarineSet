# MarineSet

Research code and data accompanying **MarineSet**, a curated large-scale dataset for underwater acoustics.

MarineSet was developed to support machine learning research in passive acoustic monitoring (PAM), with a particular focus on large-scale data curation and self-supervised representation learning for underwater sound.

This repository contains:

- Code for the automatic curation of large-scale underwater acoustic recordings.
- MarineSet audio samples and associated metadata.
- Direction-of-arrival (DOA) information.
- Automatic Identification System (AIS) information.
- Code for training and evaluating the MarineNet baseline model.

---

## Overview

Passive acoustic monitoring systems can collect large quantities of underwater audio over extended periods. These recordings contain information about vessels, marine mammals, environmental processes, and other acoustic events.

MarineSet addresses this problem through the automatic curation of large collections of underwater acoustic recordings. The dataset combines complementary curation strategies to construct a diverse collection of recordings suitable for machine learning and self-supervised representation learning.

The curation pipeline includes approaches based on:

- **AIS information**, to associate acoustic recordings with vessel activity.
- **Acoustic representation learning and clustering**, to identify acoustically diverse samples from large collections of unlabeled recordings.

The resulting dataset can be used for developing and evaluating machine learning methods for underwater acoustics, including self-supervised learning, transfer learning, ship-radiated noise classification, and marine bioacoustics.

---

## Dataset

MarineSet was constructed from long-term underwater acoustic recordings collected from hydrophone deployments in U.S. waters.

The source recordings span multiple years and hydrophone locations, providing substantial variation in:

- Geographic location
- Recording period
- Season
- Time of day
- Vessel activity
- Acoustic environment

Two complementary curation strategies were used to construct MarineSet.

### AIS-Based Curation

AIS information is used to identify vessel activity associated with underwater acoustic recordings.

Recordings are matched with nearby vessel activity using the geographic relationship between hydrophones and AIS positions. AIS positions are interpolated to obtain vessel locations at regular time intervals, which are then used to associate vessel activity with acoustic recordings.

The AIS-based curation resulted in:

- **968.2 hours** of curated audio
- **6,540 unique vessels**
- **28 hydrophones**

### Acoustic Clustering-Based Curation

A second curation strategy is based on the acoustic content of the recordings.

Audio recordings are converted into learned acoustic representations and subsequently clustered using hierarchical k-means (HK-means). The hierarchical clustering structure is used to select acoustically diverse recordings from the larger collection.

The clustering-based curation resulted in:

- **2,031.8 hours** of curated audio
- **43 hydrophones**

The HK-means hierarchy uses the following numbers of clusters:

```text
6000 → 400 → 40 → 10
