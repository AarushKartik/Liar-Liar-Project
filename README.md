
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



