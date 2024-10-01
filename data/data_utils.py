from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datetime import datetime, date, timedelta
import glob
import os

from torchvision import transforms
import torch
import xarray as xr
import numpy as np
import pandas as pd
import einops

from helpers.misc_helpers import calculate_mode

def load_patches(dates: Union[List[str],List[date]],
                 source_dir: str,
                 combined_patches: bool = False) -> List[xr.Dataset]:
    """returns a list of patches available for given dates

    Args:
        date (datetime.date): patches of date to load
        source_dir (str): path to source dir
        combined_patches (bool): If True take file mask for combined (seviri + era5 cube + dardar) patches
    Returns:
        (list[xr.Dataset]): list of patches 
    """
    fmask = "patch*.nc" if combined_patches is False else "combined_patch*.nc"

    # convert datetimes to strings if dates are given as datetimes
    if all(isinstance(d, date) for d in dates):
        dates = [d.strftime("%Y_%m_%d") for d in dates] # type: ignore

    # loop through all date subdirectories
    fnames = []
    for datestring in dates:
        date_fnames =  glob.glob(os.path.join(source_dir,datestring,fmask)) # type: ignore
        fnames += date_fnames
        if len(fnames) == 0:
            print(f"No patches available for {datestring} in source dir {source_dir}")
            return None  # type: ignore
           
    patches = [xr.load_dataset(fname) for fname in fnames]
    print(f"loaded {len(patches)} patches")
    return patches
    

def get_overpass_direction(patch: xr.Dataset) -> str:
    """return direction of satellite overpass

    example: SE means the satellite flies from North to South and West to East
    """
    if (patch.y_nearest[0] > patch.y_nearest[-1]) & (patch.x_nearest[0] >= patch.x_nearest[-1]):
        return "SW"
    elif (patch.y_nearest[0] > patch.y_nearest[-1]) & (patch.x_nearest[0] < patch.x_nearest[-1]):
        return "SE"
    elif (patch.y_nearest[0] < patch.y_nearest[-1]) & (patch.x_nearest[0] < patch.x_nearest[-1]):
        return "NE"
    elif (patch.y_nearest[0] < patch.y_nearest[-1]) & (patch.x_nearest[0] >= patch.x_nearest[-1]):
        return "NW"
    else: 
        raise ValueError(f"Double Check patch with sensing start time:{patch.sensing_start.values[0]}")

def sort_overpass_indices(overpass_indices: np.ndarray, overpass_direction: str):
    """Sorts overpass indices in seviri patch according to overpass direction

    Args:
        overpass_indices (np.ndarray): n_profiles, 2. containing x,y indices where in the patch there was a satellite overpass
        overpass_direction (str): ["SW","SE","NE","NW"]

    Returns:
        _type_: _description_
    """

    assert overpass_direction in ["SW","SE","NE","NW"], "Provide valid overpass direction"
    
    if overpass_direction=="NE":
        #NE - y: ascending x: descending (in indices space → i.e. the opposite in cartesian space, cause flipped x axis in seviri)
        overpass_indices=overpass_indices[np.lexsort((-overpass_indices[:, 0],overpass_indices[:, 1]))] # from the docs, it should be last, first to get the sort order
    
    elif overpass_direction=="NW":
        #NW - y: ascending x: ascending (in indices space → i.e. the opposite in cartesian space, cause flipped x axis in seviri)
        overpass_indices=overpass_indices[np.lexsort((overpass_indices[:, 0],overpass_indices[:, 1]))]
    
    elif overpass_direction=="SW":
        #SW - y: descending; x: ascending (in indices space → i.e. the opposite in cartesian space, cause flipped x axis in seviri)
        overpass_indices=overpass_indices[np.lexsort((overpass_indices[:, 0],-overpass_indices[:, 1]))]
    elif overpass_direction=="SE":
        #SE - y: descending; x: desscending (in indices space → i.e. the opposite in cartesian space, cause flipped x axis in seviri)
        overpass_indices=overpass_indices[np.lexsort((-overpass_indices[:, 0],-overpass_indices[:, 1]))]
        
    return overpass_indices

def combine_seviri_and_integrated_era5_channels(patch: xr.Dataset, variables: List):
        """Generate combined array (n_channels, y, x) of seviri and integrated era5 variables. Both lay on a 64x64 grid, but have different coord 
        definitions in the xr.Dataset

        Args:
            patch: xr.Dataset of single patch
            variables: list of variables with the seviri channels first e.g. ["ch4","ch5", "era5_ciwc", "era5_clwc"] 

        Returns:
            np.ndarray: concatenated array of both seviri and era5 channels (n_channels, y, x), with the first indices being the seviri channes
        """

        # get era and seviri channels
        era_channels = [ch for ch in variables if "era5" in ch]
        seviri_channels = [ch for ch in variables if ch not in era_channels]
        
        # get arrays
        era5_patch_data = patch[era_channels].isel(era5_time=0).to_array().values
        seviri_patch_data = patch[seviri_channels].to_array().values
        
        return np.concatenate((seviri_patch_data, era5_patch_data),axis=0)  

def get_patch_meta_data(patch:xr.Dataset) -> np.ndarray:
    """Return meta data of patch in 1-dim np.array
    
    order of meta data: [latitude, longitude, month, day_night, hour, land_water_mask]
    
    
    """
    lat = patch.latitude.values.mean()
    lon = patch.longitude.values.mean()
    month = patch.sensing_stop.dt.month.values[0]
    day_night_flag = calculate_mode(patch.day_night_flag.values,0)
    time_of_day = patch.sensing_stop.dt.round("1h").dt.hour.values[0]
    lwm = calculate_mode(patch.land_water_mask.values,0)
    
    return np.array([lat,lon,month,day_night_flag,time_of_day,lwm])

def get_height_level_range(min_height=1680, max_height=16980, step=60, reverse=True):
    height_range = np.arange(min_height,max_height+step,step)
    if reverse:
        return np.flip(height_range)
    else:
        return height_range
    
class UnNormalize(transforms.Normalize):
    def __init__(self,mean,std,*args,**kwargs):
        new_mean = [-m/s for m,s in zip(mean,std)]
        new_std = [1/s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)

def get_horizontal_cloud_coverage(cloud_profile, cloud_thres=0):
    """calculates relative horizontal cloud cover for one profile"""
    return (cloud_profile.sum(dim=1)>cloud_thres).sum() / cloud_profile.shape[0]

def horizontal_cover_by_level(cloud_profile, n_level_aggregation=16):
    """returns relative cloud coverage per level
    
    args:
        cloud_profile (torch.tensor): cloud profile (along dardar overpass) with shape n_pixels, n_height_levels
        n_level_aggregation (int): defines how many levels to aggregate for calculation
    """
    sliced_profile = cloud_profile.unfold(1,n_level_aggregation, n_level_aggregation)
    x = torch.empty(1,sliced_profile.shape[1])
    for i in range(sliced_profile.shape[1]):
        slice_ = sliced_profile[:,i,:]
        x[0,i] = get_horizontal_cloud_coverage(slice_)
    return x


def get_overpass_data(y_hat, dardar, overpass_mask, target_transform, meta_data=None, meta_data_filter=None):
    """_summary_

    Args:
        y_hat (torch.tensor): predictions (B x C x Z x Y x X)
        dardar (torch.tensor): dardar (B x C x Z x Y x X)
        overpass_mask (torch.tensor): (B x 1 x 1 x Y x X) also squeezed tensor works
        target_transform (): target transformation
        meta_data (torch.tensor): meta data
        meta_data_filter (tuple, optional): indicating filtering of meta data. Defaults to None.

    Returns:
        torch.tensor: concatenated overpasses (overpass pixel x C x Z)
    """

    y_hat_profile_data_list = []
    dardar_profile_data_list = []

    n_targets = dardar.shape[1]
    out_channels = dardar.shape[2]

    if meta_data_filter is not None:
        # filter for matching idxs
        idx_range = torch.where(meta_data[:,meta_data_filter[0]]==meta_data_filter[1])[0]
    else:
        # use all idxs
        idx_range = range(y_hat.shape[0])

    # in case there is no data return empty tensor
    if len(idx_range)== 0:
        y_hat = torch.empty(size=(0,n_targets,out_channels),device=y_hat.device)
        dardar = torch.empty(size=(0,n_targets,out_channels),device=y_hat.device)
        

    # create overpasses
    else:
        for idx in idx_range:
            y_hat_profile = torch.masked_select(y_hat[idx], overpass_mask[idx].bool())
            y_hat_profile = einops.rearrange(y_hat_profile, '(c z overpass) -> overpass c z',c=n_targets, z=out_channels).squeeze() # squeeze so shape is (overpass, z) for single target

            dardar_profile = torch.masked_select(dardar[idx], overpass_mask[idx].bool())
            dardar_profile = einops.rearrange(dardar_profile, '(c z overpass) -> overpass c z',c=n_targets, z=out_channels).squeeze()

            y_hat_profile_data_list.append(y_hat_profile)
            dardar_profile_data_list.append(dardar_profile)

        dardar_profile_data = torch.concat(dardar_profile_data_list)
        y_hat_profile_data = torch.concat(y_hat_profile_data_list)

        # convert to original scale → this is what we mainly work with in the evaluation
        dardar = target_transform.inverse_transform(dardar_profile_data)
        y_hat = target_transform.inverse_transform(y_hat_profile_data)
    
    return y_hat, dardar

