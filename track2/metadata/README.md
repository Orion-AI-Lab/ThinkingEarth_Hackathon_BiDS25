# Metadata


This folder stores additional data essential for our models, including:

- Training Statistics: Statistical data derived from the training set, which is applied to the test data for normalization or other preprocessing steps.

- Static Variables: Static input variables that are added to the model's feature set. This could include geographical information or other unchanging data points that are relevant for the model's predictions.

This setup ensures that all necessary auxiliary data is organized and easily accessible for both model training and inference.

Find the respective website where to download the data here:

- SFNO: [Checkpoints and Statistics](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/sfno_73ch_small)

- Pangu-Weather:

- Aurora:

# Structure

For each model, please create an individual folder. The dataloader will look for files in the respective folder. 