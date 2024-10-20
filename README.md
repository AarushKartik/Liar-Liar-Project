
# Liar-Liar Project

## Overview

This project provides a comprehensive analysis of the Liar-Liar dataset, focusing on identifying misinformation in political discourse. The repository includes scripts and modules for preprocessing the dataset and executing various analytical tasks. This README serves as a guide to understanding the functionalities of the key components in the repository, specifically highlighting data_processing.py and model_training.py.

## Table of Contents
- [Project Name](#project-name)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
    - [data_processing.py](#main_droid_fullpy)
    - [model_training.py](#main_images_fullpy)

## Getting Started

### Prerequisites

Ensure you have the following prerequisites installed:
- Python 3.7 or higher
- Required Python packages (listed in `requirements.txt`)

### Installation

Clone the repository:
```bash
git clone https://${GIT_USERNAME}:${GIT_PAT}@github.com/AarushKartik/Liar-Liar-Project.git
cd Liar-Liar-Project

```

## Usage

### data_processing.py

'data_processing.py' is the primary script for preparing the Liar-Liar dataset for analysis. It handles data cleaning, preprocessing, and feature extraction. Key components include:

- **Data Cleaning**: Removes duplicates and irrelevant information to ensure data quality.
- **Data Loading**: Reads the Liar-Liar dataset and formats it for analysis.
- **Feature Extraction**: Generates features such as sentiment scores and readability metrics to enhance model training.

### model_training.py

`main_images.py` is responsible for image processing tasks such as feature extraction, image matching, and other computer vision techniques. This script can be used independently or as part of the larger SLAM system.

- **Feature Extraction**: Detects and describes keypoints in the images.
- **Image Matching**: Matches features between images to find correspondences.
- **Visualization**: Provides tools for visualizing the processed images and matches.

## Scripts and Modules

### droid_slam

The `droid_slam` module contains the core components of the SLAM system, including:
- `droid_net.py`: Defines the neural network architecture used for feature extraction and matching.
- `factor_graph.py`: Implements the factor graph used for pose estimation and optimization.
- `data_readers`: Contains utilities for reading and preprocessing various types of input data (e.g., RGB-D images, IMU data).


