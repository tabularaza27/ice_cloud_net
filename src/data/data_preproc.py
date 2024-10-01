import argparse
import json
import os
from datetime import datetime, date, timedelta
import glob
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import multiprocessing as mp

from scipy.spatial import cKDTree

import xarray as xr
import numpy as np
import satpy
import pandas as pd
from pyproj import CRS, Transformer

def add_timestamp_coord(ds):
    """add time timestamp as coordinate to dataset"""
    base_time = ds.base_time.values  # reference time
    time_deltas = pd.to_timedelta(ds.dtime.values, 's')  # dtime
    pixel_time = base_time + time_deltas  # timestamp
    ds = ds.assign_coords({'time': pixel_time})
    return ds

def adapt_dardar_nice(ds):
    """adapt dardar nice to comply to variable names etc. of dardar cloud files

    Args:
        ds (xr.Dataset): _description_
    """

    # adapt Nice to Dardar cloud data
    ds = ds.rename(name_dict={"lat":"latitude","lon":"longitude","ta":"temperature","nightday_flag":"day_night_flag"})

    # convert time coord from iterative index do datetime
    ds = add_timestamp_coord(ds)

    return ds

def read_dardar_one_day(date, source_dir):
    """

    Args:
        date (datetime.date):
        source_dir:

    Returns:

    """
    datestring = date.strftime("%Y_%m_%d")

    if "DARDAR_NICE" in source_dir:
        source_dir += f"/{date.year}"
        preproc_func = adapt_dardar_nice
    else:
        preproc_func = None

    if not os.path.isdir(os.path.join(source_dir, datestring)):
        print(f"No DARDAR data downloaded for {datestring}")
        return None

    fnames = sorted(glob.glob(os.path.join(source_dir, datestring, "*")))
    print(f"load {len(fnames)} dardar files for {datestring}")
    dar_ds = xr.open_mfdataset(fnames, preprocess=preproc_func)

    return dar_ds


def add_cartesian_coordinates(dar_ds, target_proj):
    """adds cartesian coordinate variables to dardar dataset

    Args:
        dar_ds:
        target_proj:

    Returns:

    """
    # source proj of DARDAR coordinates (lat/lon)
    source_proj = CRS.from_string('EPSG:4326')

    dar_lons = dar_ds.longitude.values
    dar_lats = dar_ds.latitude.values

    # transform lon/lat to cartesian coordinates [m]
    transformer = Transformer.from_crs(source_proj, target_proj, always_xy=True)
    dar_x, dar_y = transformer.transform(dar_lons, dar_lats)

    dar_ds = dar_ds.assign(x=xr.DataArray(dar_x, dims="time", coords={'time': dar_ds.time.values}),
                           y=xr.DataArray(dar_y, dims="time", coords={'time': dar_ds.time.values}))

    return dar_ds


def preproc_dardar(dar_ds, roi, variables, target_proj_source_file, height_levels=(0, 25080)):
    """

    Args:
        roi (list): [lon_min, lon_max, lat_min, lat_max]
        target_proj_source_file:
        variables:
        dar_ds:
        height_levels (tuple): (min_height, max_height)
    """

    # load projection
    with open(target_proj_source_file, "r") as json_file:
        data = json.load(json_file)
    target_proj = CRS.from_json_dict(data)

    dar_ds[["longitude", "latitude", "day_night_flag"]].load()

    # crop to region of interest
    dar_ds = dar_ds.where((dar_ds.longitude >= roi[0]) & (dar_ds.longitude <= roi[1]) & (dar_ds.latitude >= roi[2]) & (
            dar_ds.latitude <= roi[3]), drop=True)

    # calculate vertically integrated variables (IWP)
    height_arr = dar_ds.height
    dar_ds["iwp"] = np.sum(dar_ds.iwc[:, height_arr >= 0], 1) * 60

    # select variables
    dar_ds = dar_ds[variables]

    # select height levels
    dar_ds = dar_ds.sel(height=slice(height_levels[1], height_levels[0]))

    # add cartesian coordinates
    dar_ds = add_cartesian_coordinates(dar_ds, target_proj)

    return dar_ds


def get_dardar_segments(dar_ds):
    """

    Args:
        dar_ds:

    Returns:
        list[tuple]: list of tuples containing start and end indices of dardar profiles along time dim
    """
    time_diff = dar_ds.time.diff(dim="time") / np.timedelta64(1, 's')  # get time diff between profiles in seconds
    segment_end_idxs = np.where(time_diff.values > 2)[
        0]  # in seconds, 2 is here somewhat random as the normal time diff between dardar obs is 0.16, but just to have some buffer

    segment_idxs = get_segment_indices(segment_end_idxs)

    return segment_idxs


def get_segment_indices(segment_end_idxs):
    """start/end idxs along time dim"""
    segment_idxs = [(None, segment_end_idxs[0] + 1)]
    for idx, i in enumerate(segment_end_idxs):
        try:
            segment_idxs.append((i + 2, segment_end_idxs[idx + 1] + 1))
        except IndexError:
            segment_idxs.append((i + 2, None))
    return segment_idxs


def calculate_mode(arr, axis):
    """calculates mode in a given array"""
    vals, counts = np.unique(arr, return_counts=True, axis=axis)
    max_occurance = np.nanargmax(counts)
    agg = vals[max_occurance]
    return agg


def bin_dardar_to_seviri_grid(dar_ds, ds_seviri, dardar_variables):
    """
    Args:
        dar_ds:
        ds_seviri:
        dardar_variables:

    Returns:

    """
    cat_variables = deepcopy(dardar_variables["categorical"])
    cont_variables = deepcopy(dardar_variables["continuous"])

    ### find nearest neighbour ###

    dar_x = dar_ds.x.values
    dar_y = dar_ds.y.values

    # find nearest neighbour seviri pixel
    sev_x = ds_seviri.x.values
    sev_y = ds_seviri.y.values

    # create meshgrid of seviri coordinates
    sev_xx, sev_yy = np.meshgrid(sev_x, sev_y)

    # Reshape the coordinates to match the format expected by cKDTree
    sev_xx_flat = sev_xx.ravel()
    sev_yy_flat = sev_yy.ravel()
    sev_coords = np.column_stack((sev_xx_flat, sev_yy_flat))

    # Build the KDTree from the coordinates in the second dataset
    tree = cKDTree(sev_coords)

    # Query the nearest neighbor for each (x, y) pair in the first dataset
    distances, indices = tree.query(np.column_stack((dar_x, dar_y)))

    # Retrieve the corresponding values from the seviri dataset using the indices
    x_nearest = sev_xx_flat[indices]
    y_nearest = sev_yy_flat[indices]

    # Add variables to dardar ds
    dar_ds = dar_ds.assign(
        x_nearest=xr.DataArray(x_nearest, dims="time", coords={'time': dar_ds.time.values}).astype(dtype=np.float64),
        y_nearest=xr.DataArray(y_nearest, dims="time", coords={'time': dar_ds.time.values}).astype(dtype=np.float64))

    # add nearest neighbour variables to cont variables
    cont_variables += ["x_nearest", "y_nearest"]

    # rename dimension time to synthetic dimension (increasing index) - profiles with same index belong to same seviri pixel
    # rename time variable to xy and assign xy values to it. Unfortunatley I have to make strings of the xy tuples
    time = dar_ds["time"].values
    xy_nearest = list(zip(x_nearest, y_nearest))

    dar_ds["time"] = [str(xy) for xy in
                      xy_nearest]  # a bit dirty, but I don't need the time dimension here as all dardar profiles were already matched to the seviri observation
    dar_ds = dar_ds.rename({"time": "xy"})

    # add time as variable
    dar_ds = dar_ds.assign(time=xr.DataArray(time, dims="xy", coords={'xy': dar_ds.xy.values}))

    # incremental list of profiles in same seviri pixel
    index = -1
    previous_value = None
    result = np.empty_like(dar_ds.xy.values)

    for i, value in enumerate(dar_ds.xy.values):
        if value != previous_value:
            index += 1
        result[i] = index
        previous_value = value

    dar_ds["xy"] = result
    dar_ds["xy"] = dar_ds["xy"].astype("int")

    ### groupby over profiles with same index ###
    # aggregate categorical variable by mode
    dar_ds[cat_variables].load()
    dar_ds_binned_cat = dar_ds[cat_variables].groupby("xy").reduce(calculate_mode, dim='xy')

    # aggregate continuous variables 
    cloud_variables_3d = [var for var in cont_variables if var in ["iwc","icnc_5um","icnc_100um","reffcli"]] 
    
    # In Cloud Means
    cloud_mask_2d = dar_ds["iwp"] != 0
    cloud_mask_3d = dar_ds["iwc"] != 0 

    dar_ds_iwp_incloud = dar_ds[["iwp"]].where(cloud_mask_2d).groupby("xy").mean(dim="xy").fillna(0)
    dar_ds_incloud = dar_ds[cloud_variables_3d].where(cloud_mask_3d).groupby("xy").mean(dim="xy").fillna(0)

    # rename variables with appendix _incloud
    dar_ds_iwp_incloud = dar_ds_iwp_incloud.rename(dict(iwp="iwp_incloud"))
    dar_ds_incloud = dar_ds_incloud.rename({f"{var}": f"{var}_incloud" for var in cloud_variables_3d})

    # todo replace nans - all cloud free regions are set to nan
    #dar_ds_iwp.iwp = dar_ds_iwp.iwp.fillna(0)
    #dar_ds_iwc.iwc = dar_ds_iwc.iwc.fillna(0)

    # Grid means
    dar_ds_binned_cont = dar_ds[cont_variables].groupby("xy").mean(dim="xy")
    dar_ds_binned_time = dar_ds["time"].groupby("xy").mean(dim="xy")  # somehow disappears in groupby in previous line

    # combine datasets
    dar_ds_binned = xr.merge((dar_ds_incloud, dar_ds_iwp_incloud, dar_ds_binned_cont, dar_ds_binned_time, dar_ds_binned_cat), combine_attrs="no_conflicts")

    return dar_ds_binned

# The following 2 functions have been adapted from amell et al. 2022
# DOI: 10.5194/amt-15-5701-2022

def _resolve_serialisation(ds: xr.Dataset) -> xr.Dataset:
    """Cannot save to disk the SEVIRI datasets as netCDF. TypeErrors are raised
    for attributes coming from attributes prepared by Satpy. Remove these
    attributes from the satpy variables (checking that they are common in all
    variables) and store them as attributes for the dataset:

    Args:
        ds: The dataset with serialisation problems

    Returns the described dataset
    """
    # Variables
    channels = ["ch{:}".format(i) for i in range(1, 12)]
    # Attributes to remove from all variables
    attrs_remove = ['sun_earth_distance_correction_applied',
                    'sun_earth_distance_correction_factor', 'platform_name', 'sensor',
                    'georef_offset_corrected', 'raw_metadata', 'start_time', 'end_time',
                    'resolution', 'calibration', 'modifiers', '_satpy_id', 'ancillary_variables']

    for var in channels:
        for key in attrs_remove:
            _ = ds[var].attrs.pop(key, None)

    # Remove two other attributes, but check they are common in all variables
    orbital_parameters_ch1 = ds[channels[0]].attrs["orbital_parameters"]
    del ds[channels[0]].attrs["orbital_parameters"]
    area_proj4_string_ch1 = ds[channels[0]].attrs["area"].proj4_string
    del ds[channels[0]].attrs["area"]
    for var in channels[1:]:
        assert ds[var].attrs["orbital_parameters"] == orbital_parameters_ch1
        del ds[var].attrs["orbital_parameters"]
        assert ds[var].attrs["area"].proj4_string == area_proj4_string_ch1
        del ds[var].attrs["area"]

    # Set these last two attributes as global to the dataset
    ds.attrs["proj4_string"] = area_proj4_string_ch1
    for key, value in orbital_parameters_ch1.items():
        ds.attrs[key] = value

    return ds

def read_seviri_native_file(filepath: str) -> xr.Dataset:
    """Reads a SEVIRI MSG Level 1.5 file in native format and returns it with
    all channels except the HRV channel loaded as an xarray Dataset

    Args:
        filepath: the full file path to the native file

    Returns the described xarray Dataset."""
    scn = satpy.Scene(filenames=[filepath], reader='seviri_l1b_native')

    # Load all channels except HRV
    with warnings.catch_warnings():
        # Ignore warning messages that come from satpy
        warnings.filterwarnings("ignore")
        scn.load(["IR_016", "IR_039", "IR_087", "IR_097", "IR_108",
                  "IR_120", "IR_134", "VIS006", "VIS008", "WV_062", "WV_073"])

        assert scn.all_same_area
        assert scn.all_same_proj

        # Get the coordinates
        x_coord = scn["IR_016"].x.values.astype(np.float32)
        y_coord = scn["IR_016"].y.values.astype(np.float32)

        # Build dataset
        ds = xr.Dataset(
            data_vars={
                "ch1": (["y", "x"],
                        scn["VIS006"].values.astype(np.float32),
                        scn["VIS006"].attrs),
                "ch2": (["y", "x"],
                        scn["VIS008"].values.astype(np.float32),
                        scn["VIS008"].attrs),
                "ch3": (["y", "x"],
                        scn["IR_016"].values.astype(np.float32),
                        scn["IR_016"].attrs),
                "ch4": (["y", "x"],
                        scn["IR_039"].values.astype(np.float32),
                        scn["IR_039"].attrs),
                "ch5": (["y", "x"],
                        scn["WV_062"].values.astype(np.float32),
                        scn["WV_062"].attrs),
                "ch6": (["y", "x"],
                        scn["WV_073"].values.astype(np.float32),
                        scn["WV_073"].attrs),
                "ch7": (["y", "x"],
                        scn["IR_087"].values.astype(np.float32),
                        scn["IR_087"].attrs),
                "ch8": (["y", "x"],
                        scn["IR_097"].values.astype(np.float32),
                        scn["IR_097"].attrs),
                "ch9": (["y", "x"],
                        scn["IR_108"].values.astype(np.float32),
                        scn["IR_108"].attrs),
                "ch10": (["y", "x"],
                         scn["IR_120"].values.astype(np.float32),
                         scn["IR_120"].attrs),
                "ch11": (["y", "x"],
                         scn["IR_134"].values.astype(np.float32),
                         scn["IR_134"].attrs)
            },
            coords={
                "x": (["x"], x_coord),
                "y": (["y"], y_coord),
                "sensing_start": (["sensing_start"], [scn.start_time]),
                "sensing_stop": (["sensing_stop"], \
                                 [pd.to_datetime(filepath.split('/')[-1].split('-')[-2].split('Z')[0],
                                                 format="%Y%m%d%H%M%S.%f").to_pydatetime()])
            },
            attrs={"source_file": os.path.basename(filepath)}
        )

    return ds


### create patches for training ###

def get_seviri_grid_idx(x, y, sev_x, sev_y):
    """returns indices of x,y values in seviri native grid

    returns
        tuple: (x_index,y_index)

    """
    return np.where(sev_x == x)[0][0], np.where(sev_y == y)[0][0]


def calculate_quadratic_bounding_box(x, y, patch_size=32):
    """calculate quadratic bounding box around given coordinates"""
    half_length = int(patch_size / 2)
    assert isinstance(half_length,int)
    x_min = x + (half_length-1) # order of x is from + (east) → - (west)
    x_max = x - half_length
    y_min = y - (half_length-1)
    y_max = y + half_length
    return x_min, x_max, y_min, y_max


def create_patches(dar_ds, ds_seviri, patch_size=64, verbose=False):
    """
        Args:
            patch_size:
            ds_seviri:
            dar_ds (xarray.Dataset): binned dardar data with dimension (xy, height)

    """
    dar_profile_idx = 0
    patches = []
    # patch_meta = []  # stores coordinate of bounding box of patch

    # seviri coord arrays
    sev_x = ds_seviri.x.values
    sev_y = ds_seviri.y.values

    while (dar_profile_idx < dar_ds.xy.size):
        if verbose:
            print("profile index", dar_profile_idx)

        y_ = dar_ds.y_nearest.values[dar_profile_idx]
        x_ = dar_ds.x_nearest.values[dar_profile_idx]

        x_idx, y_idx = get_seviri_grid_idx(x_, y_, sev_x, sev_y)

        x_min_idx, x_max_idx, y_min_idx, y_max_idx = calculate_quadratic_bounding_box(x_idx, y_idx,
                                                                                      patch_size=patch_size)

        # calculate coordinate values in seviri coords
        x_min = sev_x[x_min_idx]
        x_max = sev_x[x_max_idx]
        y_min = sev_y[y_min_idx]
        y_max = sev_y[y_max_idx]

        if verbose:
            print(x_min, x_max, y_min, y_max)

        seviri_patch = ds_seviri.sel(x=slice(x_max, x_min), y=slice(y_min, y_max))
        dar_patch = dar_ds.where((dar_ds.y_nearest >= y_min) & (dar_ds.y_nearest < y_max) & (
                dar_ds.x_nearest >= x_min) & (dar_ds.x_nearest < x_max), drop=True)

        # merge both dataset into one xr.Dataset
        patch = xr.merge((seviri_patch, dar_patch), combine_attrs="no_conflicts")

        assert patch.x.size == patch.y.size == patch_size, f"Patch has sizes ({patch.x.size},{patch.y.size}), should be {patch_size}"

        patches.append(patch)
        # patch_meta.append((x_min, x_max, y_min, y_max))

        # shift dardar profile index (i.e. center of next patch)
        # now we do it very simply by just shifting the bounding box by a fraction, resulting in small overlap
        dar_profile_idx += int(patch.xy.size * 4 / 5)
    print(f"Created {len(patches)} patches with patch size {patch_size}")

    return patches

def create_patches_pipeline(d: date,
                            seviri_observation_times: dict,
                            DARDAR_SOURCE_DIR: str = "/net/n2o/wolke_scratch2/kjeggle/DARDAR_CLOUD/v3",
                            SEVIRI_SOURCE_DIR = "/net/n2o/wolke_scratch2/kjeggle/SEVIRI",
                            SEVIRI_PROJECTION_SOURCE = "/net/n2o/wolke_scratch2/kjeggle/SEVIRI/seviri_proj.json",
                            TRAINING_DATA_DIR = "/net/n2o/wolke_scratch2/kjeggle/VerticalCloud/DataSmallDomain/TrainingData",
                            ROI = [-30,30,-30,30], # lon_min, lon_max, lat_min, lat_max
                            PATCH_SIZE = 64,
                            DARDAR_LEVELS = (1680,18060),
                            DARDAR_VARIABLES = {"continuous": ["iwp","iwc","effective_radius","u_velocity","v_velocity","temperature","time","longitude","latitude"], 
                                                "categorical": ["day_night_flag","land_water_mask","DARMASK_Simplified_Categorization"]}
                            ):
    print(f" ###### {d} ######")
    """Create patches pipeline

    preprocesses DARDAR and SEVIRI data and co-locates them, and saves to directory

    Returns:
        _type_: _description_
    """

    # reading dardar
    dar_ds = read_dardar_one_day(d, source_dir=DARDAR_SOURCE_DIR)
    
    # For some dates no DARDAR data exists
    if not dar_ds:
        d += timedelta(days=1)
        print("No DARDAR data available for this date")
        return
    
    # preproc dardar    
    dar_ds = preproc_dardar(dar_ds, 
                            roi=ROI, 
                            variables=DARDAR_VARIABLES["continuous"]+DARDAR_VARIABLES["categorical"],
                            target_proj_source_file=SEVIRI_PROJECTION_SOURCE,
                            height_levels=DARDAR_LEVELS)
    
    # discard if < 100 observations exist
    if dar_ds.time.size < 100:
        print(f"only {dar_ds.time.size} profiles exist for dardar data, discard it and continue with next")
        d += timedelta(days=1)
        return
    
    # generating dardar segments
    segment_idxs = get_dardar_segments(dar_ds)
    
    # looping through available dardar segments available on this day #
    all_patches = []    
    for segment in segment_idxs:
        segment_start_idx, segment_end_idx = segment
        segment_dar_ds = dar_ds.isel(time=slice(segment_start_idx,segment_end_idx))
        segment_dar_ds = dar_ds.isel(time=slice(segment_start_idx,segment_end_idx))

        # get segment start/end time
        start_time = pd.to_datetime(segment_dar_ds.time.values.min())
        end_time = pd.to_datetime(segment_dar_ds.time.values.max())
        print(f"Segment start: {start_time} , end: {end_time}")

        # get available seviri files ( check if overlap between dardar and seviri files exist )
        #available_seviri_obs = get_MSG15_ids(start_time,end_time,satellite="MSG2")
        available_seviri_obs = {file: obs_times for file, obs_times in seviri_observation_times.items() if check_time_overlap(start_time, end_time, obs_times["timestamp_start"], obs_times["timestamp_stop"])}
        available_seviri_obs = dict(sorted(available_seviri_obs.items())) # sort by key
        #pprint(available_seviri_obs)
        
        if len(available_seviri_obs.keys()) == 0:
            print("no seviri files available for this segment")
            continue
        
        ### loop through seviri obs available for this segment ###
        for seviri_file, seviri_file_meta in available_seviri_obs.items():
            print(seviri_file)
            print(f"Seviri file: {seviri_file_meta['timestamp_start']} - {seviri_file_meta['timestamp_stop']}")
            try:
                ds_seviri = read_seviri_native_file(f"{SEVIRI_SOURCE_DIR}/{seviri_file}.nat")
                ds_seviri = _resolve_serialisation(ds_seviri)
                # todo resolve serialization already here
            except FileNotFoundError:
                print(f"{seviri_file} has not been downloaded yet, skip for now and continue with next")
                continue
            # coordinates to float64 as otherwise averageing of the floats at a late point is not precise
            ds_seviri["x"] = ds_seviri.x.astype(np.float64)#
            ds_seviri["y"] = ds_seviri.y.astype(np.float64)

            # crop dardar data along time dim → temporal colocation with seviri observation
            seviri_start_time = pd.to_datetime(ds_seviri.sensing_start.values[0])
            seviri_end_time = pd.to_datetime(ds_seviri.sensing_stop.values[0])
            dar_ds_seviri_obs = segment_dar_ds.sel(time=slice(seviri_start_time,seviri_end_time))
            
            # discard if < x observations exist
            if dar_ds_seviri_obs.time.size < 10:
                print(f"only {dar_ds_seviri_obs.time.size} profiles exist for this seviri image, discard it and continue with next")
                continue
            
            # bin dardar data to seviri grid
            dar_ds_binned = bin_dardar_to_seviri_grid(dar_ds_seviri_obs, ds_seviri, DARDAR_VARIABLES)

            patches = create_patches(dar_ds_binned, ds_seviri, patch_size=PATCH_SIZE)
            all_patches.append(patches)
    
    # save patches to disk
    patches_flat = sum(all_patches, [])
    print(f"save {len(patches_flat)} patches to disk")
    save_training_data(patches_flat,d, TRAINING_DATA_DIR)
    
    return 1

def save_training_data(patches, d, TRAINING_DATA_DIR="/net/n2o/wolke_scratch2/kjeggle/VerticalCloud/DataSmallDomain/TrainingData"):
    date_str = datetime.strftime(d,"%Y_%m_%d")
    save_dir = os.path.join(TRAINING_DATA_DIR,date_str)
    if not os.path.isdir(save_dir):  os.makedirs(save_dir)
    for idx, patch in enumerate(patches):
        # quick n dirty removal of time parameters and orbital parameters → include time parameters in _resolve_serialisation
        for i in range(1,12):
            try:
                del patch[f"ch{i}"].attrs['orbital_parameters']
            except KeyError:
                pass
            try:
                del patch[f"ch{i}"].attrs['time_parameters']
                #print(f"deleted time_parameters attribute from ch{i}") 
            except KeyError:
                pass
        patch.to_netcdf(os.path.join(save_dir,f"patch_{idx}.nc"))
        #print("saved", os.path.join(save_dir,f"patch_{idx}.nc"))

def check_time_overlap(start_time1, end_time1, start_time2, end_time2):
    if start_time1 <= start_time2 <= end_time1 or start_time2 <= start_time1 <= end_time2:
        return True
    return False

def date_iterator(start_date, end_date):
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += timedelta(days=1)

def get_seviri_observation_times(seviri_file: str):
    end_time = datetime.strptime(seviri_file.split("-")[-2],"%Y%m%d%H%M%S.%f000Z")
    start_time = end_time - timedelta(seconds=12.6*60) # 12.6 minutes
    
    return {"timestamp_start": start_time, "timestamp_stop": end_time}


if __name__ == '__main__':
    ### create patches for training IceCloudNet ###
    # i.e. co-locating SEVIRI and DARDAR-Nice data
    # chopping into patches suitable for training ML models

    # example: python data_preproc.py 20100101 20101231 8
    # creates patches for 1 year with 8 parallel threads
    # for more details on preproc logic check create_patches_pipeline

    parser = argparse.ArgumentParser(description="Create patches for IceCloudNet Training")
    
    # define directory locations
    SEVIRI_SOURCE_DIR = "/path/to/unzipped_SEVIRI_files"
    DARDAR_SOURCE_DIR = "/path/to/DARDAR_Nice_files"
    TRAINING_DATA_DIR = "./path/to/dir/where/patches/will/be/stored"
    SEVIRI_PROJECTION = "./seviri_proj.json"

    PATCH_SIZE = 128
    ROI = [-32,32,-32,32], # lon_min, lon_max, lat_min, lat_max

    # Start date: positional argument for date_string in format YYYYMMDD
    parser.add_argument(
        "dt_start", 
        type=lambda d: datetime.strptime(d, "%Y%m%d").date(), 
        help="Date in format YYYYMMDD"
    )

    # End date: positional argument for date_string in format YYYYMMDD
    parser.add_argument(
        "dt_end", 
        type=lambda d: datetime.strptime(d, "%Y%m%d").date(), 
        help="Date in format YYYYMMDD"
    )

    # Number of threads for parallel execution
    parser.add_argument(
        "n_workers", 
        type=int, 
        help="number of threads"
    )
    
    args = parser.parse_args()

     
    # Define the start date and end date
    start_date = args.dt_start
    end_date = args.dt_end
    n_workers = args.n_workers

    # get all downloaded seviri files and add start stop times
    all_seviri_files = glob.glob(os.path.join(SEVIRI_SOURCE_DIR,"MSG*.nat"))
    seviri_observation_times = {sev_file.split("/")[-1].split(".nat")[0]: get_seviri_observation_times(sev_file) for sev_file in all_seviri_files}
   
    pipeline_dict = dict(seviri_observation_times=seviri_observation_times,
                        TRAINING_DATA_DIR=TRAINING_DATA_DIR,
                        DARDAR_SOURCE_DIR =DARDAR_SOURCE_DIR,
                        SEVIRI_SOURCE_DIR =SEVIRI_SOURCE_DIR,
                        SEVIRI_PROJECTION_SOURCE =SEVIRI_PROJECTION,
                        ROI = ROI, # lon_min, lon_max, lat_min, lat_max
                        PATCH_SIZE = PATCH_SIZE,
                        DARDAR_LEVELS = (1680,20580),
                        DARDAR_VARIABLES = {"continuous": ["iwp","iwc","reffcli","icnc_5um","icnc_100um","temperature","time","longitude","latitude"], 
                                            "categorical": ["day_night_flag","land_water_mask","iteration_flag","land_water_mask","clm"]})
    # create pool and run in pipeline in parallel
    pool = mp.Pool(n_workers)
    for d in date_iterator(start_date, end_date):
        pool.apply_async(create_patches_pipeline, kwds=dict(d=d, **pipeline_dict))
        
        #ROI=[-30,30,40,60],
        #TRAINING_DATA_DIR="/net/n2o/wolke_scratch2/kjeggle/VerticalCloud/DataSmallDomainNorth/TrainingData")

    pool.close()
    pool.join()
