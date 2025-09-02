import os
import logging
from typing import Optional, List, Tuple, Union

import torch
import json
import numpy as np
from torch.utils.data import Dataset
import h5py as h5
import datetime as dt

from aurora import Batch, Metadata


class dataloader_era5(Dataset):
    """
    A dataloder to load the ERA5 data provided in the daata files for test data 2018.
    """

    def __init__(self, 
                data_path: str,
                metadata_path: str,
                in_channels: List[int],
                out_channels: List[int],
                stats_mean_path: str = None,
                stats_std_path: str = None,
                normalize: bool = True):
        """
        Args:
            data_path (string): Path to the h5 file with ERA5 data.
            in_channels (list[int]): XXX
            out_channels (list[int]): XXX
        """
        self.data_path = data_path
        self.stats_mean_path = stats_mean_path
        self.stats_std_path= stats_std_path
        self.metadata_path = metadata_path
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.mean, self.std = self.get_stats()
        self.metadata = self.get_metadata()

    def get_stats(self):
        if self.stats_mean_path is None and self.stats_std_path is None:
            self.stats_mean_path = "track2/metadata/{model}/global_means.npy"
            self.stats_std_path = "track2/metadata/{model}/global_stds.npy"
        mean = np.load(self.stats_mean_path, allow_pickle=True)
        std = np.load(self.stats_std_path, allow_pickle=True)

        return mean, std

    def get_metadata(self):
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)
        return metadata

    def get_channel_list(self):
        return self.metadata['coords']['channel']

    def get_data(self, date, model):
        # date input has to has format: "%Y-%M-%DT%h:%m:%s"
        # example: '2011-11-04T00:05:23'

        # convert input data from string to datetime objectm timezone utc
        datetime = dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc)
        print(f"Date: {datetime}")

        # convert to timestamps - this is
        timestamp = datetime.timestamp()

        print(f"Reading input file from {self.data_path}...")
        with h5.File(self.data_path, 'r') as fin:
            entry_key = "fields"
            data_handle = fin[entry_key]
            print(f"Shape of data_handle: {data_handle.shape}")
            # find start and end index in timestamps
            timestamps = fin["timestamp"][...]
            if len(np.where(timestamps == timestamp)[0]) == 0:
                raise ValueError(f"Timestamp {datetime} not found in file.")
            else:
                assert len(np.where(timestamps == timestamp)[0])==1, f"Several timepoints refer to the input data. Length: {len(np.where(timestamps == timestamp)[0])}"
                index = np.where(timestamps == timestamp)[0][0]
                data_timestamp = timestamps[index]
            # extract data
            if model=="aurora":
                data = torch.from_numpy(data_handle[index-1:index+1, ...])
                print(f"Shape of data: {data.shape}")
            else:
                data = data_handle[index, ...]
                print(f"Shape of data: {data.shape}")

        # channel names
        channel_list = self.get_channel_list()
        print(f"Channel list: {channel_list}")

        if model=="aurora":
            # FOR NOW: Random static data
            # randomly generate a numpy array of dimension (8,721,1440)
            static_data = torch.from_numpy(np.random.rand(3,721,1440)).float()
            static_data = static_data = static_data
            print(f"Shape of static data: {static_data.shape}")
            # time
            time = np.array(data_timestamp)
            time = time.astype("datetime64[s]").tolist()
            print(f"Time: {time}")
            # convert to torch
            pressure_levels = np.array([50,100,150,200,250,300,400,500,600,700,850,925,1000])
            # build batch to return
            batch = Batch(
                surf_vars={
                    # First select the first two time points: 00:00 and 06:00. Afterwards, `[None]`
                    # inserts a batch dimension of size one.
                    "2t": data[:2,channel_list.index('t2m'),:,:][None],
                    "10u": data[:2,channel_list.index('u10m'),:,:][None],
                    "10v": data[:2,channel_list.index('v10m'),:,:][None],
                    "msl": data[:2,channel_list.index('msl'),:,:][None],
                },
                static_vars={
                    # The static variables are constant, so we just get them for the first time.
                    "z": static_data[0, :, :],
                    "slt": static_data[1, :, :],
                    "lsm": static_data[2, :, :],
                },
                atmos_vars={
                    "t": data[:2,channel_list.index('t50'):channel_list.index('t50')+13,:,:][None],
                    "u": data[:2,channel_list.index('u50'):channel_list.index('u50')+13,:,:][None],
                    "v": data[:2,channel_list.index('v50'):channel_list.index('v50')+13,:,:][None],
                    "q": data[:2,channel_list.index('q50'):channel_list.index('q50')+13,:,:][None],
                    "z": data[:2,channel_list.index('z50'):channel_list.index('z50')+13,:,:][None],
                },
                metadata=Metadata(
                    lat=torch.from_numpy(np.array(self.metadata['coords']['lat'])),
                    lon=torch.from_numpy(np.array(self.metadata['coords']['lon'])),
                    # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
                    # `datetime.datetime`s. Note that this needs to be a tuple of length one:
                    # one value for every batch element. Select element 1, corresponding to time
                    # 06:00.
                    time=(time,),
                    atmos_levels=tuple(int(level) for level in pressure_levels),
                ),
            )
            output=batch
        elif model =="pangu":
            # get indices in channel_list for surface and atmospheric variables
            surf_ids = [channel_list.index(var) for var in ['msl', 'u10m', 'v10m', 't2m']]
            pangu_z_list = ["z1000", "z925", "z850", "z700", "z600", "z500", "z400", "z300", "z250", "z200", "z150", "z100", "z50"]
            pangu_q_list = ["q1000", "q925", "q850", "q700", "q600", "q500", "q400", "q300", "q250", "q200", "q150", "q100", "q50"]
            pangu_t_list = ["t1000", "t925", "t850", "t700", "t600", "t500", "t400", "t300", "t250", "t200", "t150", "t100", "t50"]
            pangu_u_list = ["u1000", "u925", "u850", "u700", "u600", "u500", "u400", "u300", "u250", "u200", "u150", "u100", "u50"]
            pangu_v_list = ["v1000", "v925", "v850", "v700", "v600", "v500", "v400", "v300", "v250", "v200", "v150", "v100", "v50"]
            z_ids = [channel_list.index(var) for var in pangu_z_list]
            q_ids = [channel_list.index(var) for var in pangu_q_list]
            t_ids = [channel_list.index(var) for var in pangu_t_list]
            u_ids = [channel_list.index(var) for var in pangu_u_list]
            v_ids = [channel_list.index(var) for var in pangu_v_list]
            # extract upper_data and surface_data
            surface_data = data[surf_ids, :, :].astype(np.float32)
            upper_data = np.stack((data[z_ids, :, :], data[q_ids, :, :], data[t_ids, :, :], data[u_ids, :, :], data[v_ids, :, :]), axis=0).astype(np.float32)
            # return
            output = upper_data, surface_data
        elif model == "sfno":
            # FOR NOW: randomly generate a numpy array of dimension (8,721,1440)
            rand_data = np.random.rand(2,721,1440)
            # concatenate data with rand_data
            final_data = np.concatenate((data, rand_data), axis=0)
            final_data = np.expand_dims(final_data, axis=0)
            # convert to torch and make it float
            final_data = torch.from_numpy(final_data).float()
            print(f"Shape of final_data: {final_data.shape}")
            # return
            output = final_data

        # normalize variables
        if self.normalize:
            data = (data - self.mean) / self.std

        return output

