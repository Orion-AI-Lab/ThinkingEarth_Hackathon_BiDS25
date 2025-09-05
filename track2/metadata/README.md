# Data

This folder serves to store the data required for the setup, please include your local downloads of the ERA5 data as well as additional metdata here.

We recommend to structure the data in this folder as follows: 

```
data
├── ...
├── era5                                 # folder for era5 data
│   ├── 2018                             # 2018 data
│      ├── 73varQ                        # data comprising 73 variables, including the q variables
│      ├── 73varR                        # data comprising 73 variables, including the q variables
│   ├── static                             # static era5 data
├── sfno                                 # sfno-related data, i.e., checkpoints and stats
```

This setup ensures that all necessary auxiliary data is organized and easily accessible for both model training and inference.

Find the respective website where to download the data here:

- Hackathon 2018 data: [HuggingFace](https://huggingface.co/datasets/franzigrkn/thinking_earth_hackathon_bids2025/tree/main/2018)

- SFNO: [Checkpoints and Statistics](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/sfno_73ch_small)

- Pangu-Weather: TBD

- Aurora: TBD

# Structure

For each model's specific data, please create an individual folder. The dataloader will look for files in the respective folder. 