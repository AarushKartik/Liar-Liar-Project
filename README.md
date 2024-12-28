
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
    - [data_processing.py](#data_processing_py)
    - [model_training.py](#model_training_py)
  - [Paper and Presentation Preview](#paper-and-presentation-preview)

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

### LightGBM Installation

Building LightGBM: 
```bash
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
cmake -B build -S .
cmake --build build -j4
```

## Usage

### data_processing.py

'data_processing.py' is the primary script for preparing the Liar-Liar dataset for analysis. It handles data cleaning, preprocessing, and feature extraction. Key components include:

- **Data Cleaning**: Removes duplicates and irrelevant information to ensure data quality.
- **Data Loading**: Reads the Liar-Liar dataset and formats it for analysis.
- **Feature Extraction**: Generates features such as sentiment scores and readability metrics to enhance model training.

### model_training.py

`model_training.py` is responsible for training machine learning models on the Liar-Liar dataset to classify political statements based on truthfulness. This script can be used for model training and evaluation.

- **Model Training**: Trains a Long Short-Term Memory (LSTM) model for fake news detection.
- **Evaluation**: Provides accuracy, precision, recall, and F1-score metrics.
- **Visualization**: Displays model performance through confusion matrices and classification reports.

## Paper and Presentation Preview

### Paper Preview

You can view a preview of the paper for this project by clicking the link below:

[![PDF Paper Preview](https://raw.githubusercontent.com/AarushKartik/Liar-Liar-Project/main/assets/paper_thumbnail.png)](https://github.com/AarushKartik/Liar-Liar-Project/raw/main/Aarush_Liar_Liar_Paper.pdf)

This link will open the PDF paper directly in your browser.

### Presentation Preview

You can view a preview of the presentation for this project by clicking the link below:

[![PDF Presentation Preview](https://raw.githubusercontent.com/AarushKartik/Liar-Liar-Project/main/assets/presentation_thumbnail.png)](https://github.com/AarushKartik/Liar-Liar-Project/raw/main/Aarush_Liar_Liar_Presentation.pdf)

This link will open the PDF presentation directly in your browser.

