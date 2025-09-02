#import torch_harmonics
import torch
import json
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
from typing import Optional, Dict, Any

#from torch_harmonics.examples.models import SphericalFourierNeuralOperatorNet as SFNO
from makani.models.networks.sfnonet import SphericalFourierNeuralOperatorNet as SFNO
from torch_harmonics.examples import PdeDataset

def prepend_prefix_to_state_dict(
    state_dict: Dict[str, Any],
    prefix: str,
) -> None:
    r"""Append the prefix to states in state_dict in place.

    ..note::
        Given a `state_dict` from a local model, a DP/DDP model can load it by applying
        `prepend_prefix_to_state_dict(state_dict, "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.

    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = list(state_dict.keys())
    for key in keys:
        newkey = prefix + key
        state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata if any.
    if hasattr(state_dict, "_metadata"):
        keys = list(state_dict._metadata.keys())
        for key in keys:
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.
            if len(key) >= 0:
                newkey = prefix + key
                state_dict._metadata[newkey] = state_dict._metadata.pop(key)

def pop_prefix_from_state_dict(
    state_dict: Dict[str, Any],
    prefix: str,
) -> None:
    r"""Append the prefix to states in state_dict in place.

    ..note::
        Given a `state_dict` from a local model, a DP/DDP model can load it by applying
        `prepend_prefix_to_state_dict(state_dict, "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.

    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = list(state_dict.keys())
    for key in keys:
        # find prefix part in key and remove it
        if key.startswith(prefix):
            newkey = key[len(prefix):]
            state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata if any.
    if hasattr(state_dict, "_metadata"):
        keys = list(state_dict._metadata.keys())
        for key in keys:
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.
            if len(key) >= 0:
                newkey = prefix + key
                state_dict._metadata[newkey] = state_dict._metadata.pop(key)

def main():

    # set seed
    torch.manual_seed(1404)
    torch.cuda.manual_seed(1404)

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device.index)

    # 1 hour prediction steps
    dt = 1*3600
    dt_solver = 150
    nsteps = dt//dt_solver
    dataset = PdeDataset(dt=dt, nsteps=nsteps, dims=(256, 512), device=device, normalize=True)
    # There is still an issue with parallel dataloading. Do NOT use it at the moment
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, persistent_workers=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, persistent_workers=False)

    # load config
    config_path = "/bids_weather_forecasting_hackathon/checkpoints/sfno_checkpoints/sfno_73ch_small_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    #print(f"Config: {config}")

    nlat = config['img_shape_x']
    nlon = config['img_shape_y']
    print(f"num lat: {nlat}")
    print(f"num lon: {nlon}")

    model = partial(SFNO, 
        img_size=(nlat, nlon),  
        grid=config["data_grid_type"],
        num_layers=config['num_layers'], 
        scale_factor=config['scale_factor'],
        inp_chans=config["N_in_channels"],
        out_chans=config["N_out_channels"],
        embed_dim=config['embed_dim'], 
        big_skip=True, 
        pos_embed=config["pos_embed"], 
        use_mlp=config["use_mlp"], 
        normalization_layer=config["normalization_layer"]
    )

    model = model()
    print(f"Model: {type(model)}")
    print(f"Model: {model}")

    # load checkpoint
    ckpt_path = "/bids_weather_forecasting_hackathon/checkpoints/sfno_checkpoints/checkpoints/sfno_73ch_small_training_checkpoints_best_ckpt_mp0.tar"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state"]
    pop_prefix_from_state_dict(state_dict, "module.model.")
    # load state dict
    model.load_state_dict(state_dict, strict=True)

if __name__ == "__main__":
    #import torch.multiprocessing as mp
    #mp.set_start_method('forkserver', force=True)

    main()