# Track 2: Weather forecasting models

This track centers on assessing and extending the capabilities of state-of-the-art weather models. Participants are encouraged to develop new evaluation frameworks and apply models to a variety of downstream forecasting tasks to identify strengths and gaps in current approaches.

## Materials

This repository contains all the materials and information for track 2 of the hackathon. Participants will find everything they need to build their projects, including:

- Implementations of state-of-the-art weather forecasting models
- Introduction of possible downstream tasks to evaluate
- Access to 2018 data from the ERA5 dataset

## Docker image
We provide the setup of a docker image that can be used to develop and work on the hackathon task.

## Local install of the makani library
We will load the SFNO from the open-source [makani](https://github.com/NVIDIA/makani) codebase. Please install the codebasse in editable mode in your loaded docker image. Please make sure you install the version v0.1.1, as the newest version will have issues with the provided checkpoint. 

```bash
git clone git@github.com:NVIDIA/makani.git
cd makani
git checkout v0.1.1
pip install -e .
```

This will allow you to use the SFNO implementation, load the checkpoints and use some functionality to derive the cosine zenith angle used as additional static variables.