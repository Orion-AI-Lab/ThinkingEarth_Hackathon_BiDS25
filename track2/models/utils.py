from typing import Optional, Dict, Any

import datetime as dt
import numpy as np

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

def get_date_from_timestamp(timestamp):
    return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)

def compute_zenith_angle(timestamp, lat_grid, lon_grid):
    # import
    from makani.makani.third_party.climt.zenith_angle import cos_zenith_angle

    # convert to datetimes:
    datetime = get_date_from_timestamp(timestamp)
    # remove offset information
    datetime = datetime.replace(tzinfo=None)
    
    # compute the corresponding zenith angles
    cos_zenith = cos_zenith_angle(datetime, lon_grid, lat_grid).astype(np.float32)

    return cos_zenith