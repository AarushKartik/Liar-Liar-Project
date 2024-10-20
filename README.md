
# Site Walk

## Overview

This project provides an implementation of a visual SLAM system and image processing pipelines. The repository contains various scripts and modules designed for simultaneous localization and mapping (SLAM) and image analysis tasks. This README provides a comprehensive guide to understanding the functionality and usage of the main components in the repository, specifically focusing on `main_droid_full.py` and `main_images_full.py`.

## Table of Contents
- [Project Name](#project-name)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
    - [main_droid_full.py](#main_droid_fullpy)
    - [main_images_full.py](#main_images_fullpy)
  - [Scripts and Modules](#scripts-and-modules)
    - [droid_slam](#droid_slam)
    - [image_processing](#image_processing)

## Getting Started

### Prerequisites

Ensure you have the following prerequisites installed:
- Python 3.7 or higher
- Required Python packages (listed in `requirements.txt`)

### Installation

Clone the repository:
```bash
git clone https://${GIT_USERNAME}:${GIT_PAT}@github.com/nexterarobotics/sitewalk.git
cd sitewalk
```

## Usage

### main_droid.py

`main_droid.py` is the primary script for running the full SLAM system. This script handles the initialization, processing, and optimization steps required for visual SLAM. Below is an overview of its key components:

- **Initialization**: Sets up the necessary parameters and data structures.
- **Data Loading**: Reads input data and preprocesses it.
- **SLAM Processing**: Executes the SLAM pipeline, including feature extraction, matching, and pose estimation.
- **Optimization**: Performs bundle adjustment and other optimization techniques to refine the SLAM results.

### main_images.py

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

### To Build the Docker Container
```bash
docker build -f docker/Dockerfile.droid \
             -t sitewalk-droid:2.0 \
             --build-arg GIT_USERNAME=GIT_USERNAME \
             --build-arg GIT_PAT=GIT_PAT \
             --build-arg AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
             --build-arg AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
             --build-arg AWS_DEFAULT_REGION=AWS_DEFAULT_REGION \
             .

docker build -f docker/Dockerfile.image \
             -t sitewalk-image:2.0 \
             --build-arg GIT_USERNAME=GIT_USERNAME \
             --build-arg GIT_PAT=GIT_PAT \
             --build-arg AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
             --build-arg AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
             --build-arg AWS_DEFAULT_REGION=AWS_DEFAULT_REGION \
             .
```

### To Run the Docker Container
```bash
docker run -it \
    --gpus all \
    --shm-size=8g \
    sitewalk-droid:2.0 \
    bash -c "chmod +x /sitewalk/batch_run_droid.sh && \
             /sitewalk/batch_run_droid.sh \
             --site_walk_id 2391"

docker run -it \
    sitewalk-image:2.0 \
    bash -c "chmod +x /sitewalk/batch_run_image.sh && \
             /sitewalk/batch_run_droid.sh \
             --site_walk_id 2391 \
	     --pcd_sub_id 6"
```

### AWS ECR - Uploading Docker Images

To upload a Docker image to AWS ECR, follow these steps:

1. **Set Up AWS CLI:**
   Ensure AWS CLI is installed on your machine.

2. **Configure AWS CLI:**
   Use `aws configure` to set up your credentials.

3. **Authenticate Docker to ECR:**
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 521337707473.dkr.ecr.us-east-1.amazonaws.com
   ```

4. **Tag the Docker Image with both Version and Latest:**
   ```bash
   docker tag sitewalk-droid:[version] [your-aws-account-id].dkr.ecr.[region].amazonaws.com/site-walk:[version]
   docker tag sitewalk-droid:[version] [your-aws-account-id].dkr.ecr.[region].amazonaws.com/site-walk:latest
   ```

   ```bash
   docker tag sitewalk-image:[version] [your-aws-account-id].dkr.ecr.[region].amazonaws.com/site-walk-image-upload:[version]
   docker tag sitewalk-image:[version] [your-aws-account-id].dkr.ecr.[region].amazonaws.com/site-walk-image-upload:latest
   ```

5. **Push the Image:**
   ```bash
   docker push [your-aws-account-id].dkr.ecr.[region].amazonaws.com/site-walk:[version]
   docker push [your-aws-account-id].dkr.ecr.[region].amazonaws.com/site-walk:latest
   ```

   ```bash
   docker push [your-aws-account-id].dkr.ecr.[region].amazonaws.com/site-walk-image-upload:[version]
   docker push [your-aws-account-id].dkr.ecr.[region].amazonaws.com/site-walk-image-upload:latest
   ```

These steps will allow you to upload your container images directly to AWS ECR for further use in your deployments.

## Getting Started

To set up your environment and ensure you have all necessary tools and dependencies, follow the steps outlined below.

### System Update and Python Installation

First, update your system's package index and install Python 3 and pip:

```bash
sudo apt update
sudo apt install python3-pip
python3 -m pip install --upgrade pip
```

### Monitoring Tools

#### CPU Monitoring

Install `bpytop` for monitoring CPU usage:

```bash
sudo apt install snapd
sudo snap install bpytop
bpytop
```

#### GPU Monitoring

Install `nvitop` for monitoring GPU usage:

```bash
sudo apt install pipx
sudo apt install python3-venv
pipx run nvitop
```

### Essential Tools Installation

#### Ninja Build System

Install the Ninja build system:

```bash
sudo wget -qO /usr/local/bin/ninja.gz https://github.com/ninja-build/ninja/releases/latest/download/ninja-linux.zip
sudo gunzip /usr/local/bin/ninja.gz
sudo chmod a+x /usr/local/bin/ninja
ninja --version
```

#### pybind11

Install `pybind11` for building Python C++ extensions:

```bash
git clone https://github.com/pybind/pybind11.git 
cd pybind11 
mkdir build 
cd build 
cmake .. 
sudo make -j$(nproc) install
```

### Repository Setup

#### Clone the Repository

Clone the project repository:

```bash
git clone https://yourusername:yourtoken@github.com/nexterarobotics/sitewalk.git
cd sitewalk
```

### Virtual Environments

Create and activate virtual environments for the project.

#### DROID Virtual Environment

Create and set up the virtual environment for SLAM:

```bash
python3 -m venv droid
source droid/bin/activate
python3 -m pip install --upgrade pip
pip install --upgrade pip
```

### PyTorch Installation

Install PyTorch with CUDA support.

#### PyTorch (CUDA 12.1)

```bash
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

#### PyTorch (CUDA 11.8)

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```


#### DROID Repository Requirements

Install the dependencies for the DROID repository:

```bash
pip install -r requirements/requirements.droid.txt
pip install PyYAML==5.1
pip install pytorch-lightning==1.3.5
```

### DROID Repository Setup

Clone and set up the DROID SLAM repository:

```bash
git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git && \
    cd DROID-SLAM && \
    python3 setup.py install && \
    cd ..
```

#### Image Processing Virtual Environment

Create and set up the virtual environment for image processing:

```bash
python3 -m venv image
source image/bin/activate
python3 -m pip install --upgrade pip
pip install --upgrade pip
```

#### Image Repository Requirements

Install the dependencies for the image processing repository:

```bash
pip install -r requirements/requirements.image.txt
```

### Additional Tools and Configurations

#### Install ffmpeg

Install `ffmpeg` for video processing:

```bash
sudo apt-get install ffmpeg
```

#### AWS CLI Configuration

Install and configure the AWS CLI:

```bash
sudo snap install aws-cli --classic
aws configure
```

#### Download Weights

Download the pre-trained ResNet50 weights for blurriness detection:

```bash
aws s3 cp s3://didge-cv-models/ResNet50/blur_detection/best_val_acc.pth weights/
aws s3 cp s3://didge-cv-models/LoFTR/outdoor_ds.ckpt loftr/weights/
aws s3 cp s3://didge-cv-models/DROID-SLAM/droid.pth weights/
```

#### Path and Environment Configuration

```bash
export PATH=$PATH:/snap/bin
conda config --set auto_activate_base false
```
