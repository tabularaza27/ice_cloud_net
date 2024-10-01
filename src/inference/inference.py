import argparse
import os
import gc
import glob
import io
import itertools
from datetime import datetime
from collections import OrderedDict

import xarray as xr
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from distributed import Client
import dask.array as da
from pyorbital import astronomy

from einops import rearrange
from omegaconf import OmegaConf

import socket
import sys

from data.data_utils import load_patches, sort_overpass_indices, get_overpass_direction, get_height_level_range
from helpers.misc_helpers import *
from helpers.comet_helpers import get_patch_ids, load_experiment

from helpers.comet_helpers import get_patch_ids, load_experiment, get_trained_model, get_trained_experiment, get_dm

from data.data_module import LogTransform, LogTransform2D, VerticalCloudDataModule

from model.DiscriminatorModel import AutoencoderKL


def load_discriminator_model(model_conf_filepath,model_checkpoint_dir, epoch=None):
    # Model with Discriminator
    # base confif
    model_conf = OmegaConf.load(model_conf)
    # load model

    model = AutoencoderKL(**model_conf["model"]["params"])
    
    try:
        if epoch:
            # load specific epoch
            try:
                path = glob.glob(os.path.join(model_checkpoint_dir, f"epoch={epoch}*.ckpt"))[0]
            except IndexError:
                raise ValueError(f"no checkpoint available for epoch {epoch}")
        else:
            # load last available checkpoint
            paths = glob.glob(os.path.join(model_checkpoint_dir, "*.ckpt"))
            paths.sort(key=os.path.getmtime)
            path = paths[-1]
        print(f"load model from path: {path}")
        pretrained_state_dict = torch.load(path)["state_dict"]
        print("loaded model from disk")
    except IndexError:
        # load from comet
        raise ValueError(f"no checkpoint available in {model_checkpoint_dir}")
    if "module._forward_module." in next(iter(pretrained_state_dict.keys())):
        pretrained_state_dict = OrderedDict([(key.split("module._forward_module.")[-1],val) for key,val in pretrained_state_dict.items()])
    model.load_state_dict(pretrained_state_dict)
    model.out_channels = model_conf["model"]["params"]["unetconfig"]["params"]["out_channels"]
    model.meta_data_embedding=model.unet.meta_data_embedding
    return model, model_conf

def load_seviri_data(d, data_dir, patch_size=128):
    input_dir = os.path.join(data_dir,"SeviriWholeAreaInput")
    ds_seviri = xr.open_dataset(os.path.join(input_dir, f"seviri_timeseries_{d}.nc"))
    ds_seviri = add_meta_seviri(ds_seviri, os.path.join(data_dir))

    # crop ds that it fits into 128x128 patches
    n_xdirection = np.floor(ds_seviri.x.size / patch_size)
    n_ydirection = np.floor(ds_seviri.y.size / patch_size)
    
    x_offset=(ds_seviri.x.size-(n_xdirection*patch_size))/2
    x_min = int(x_offset)
    x_max = int(ds_seviri.x.size - x_offset)
    
    y_offset=(ds_seviri.y.size-(n_ydirection*patch_size))/2
    y_min = int(y_offset)
    y_max = int(ds_seviri.y.size - y_offset)
    
    ds_seviri = ds_seviri.isel(x=slice(x_min,x_max),y=slice(y_min,y_max))

    return ds_seviri

def add_meta_seviri(seviri_ds, aux_dir):
    """add meta data to seviri

    map calipso lwm that is used during training to seviri

    Args:
        seviri_ds (_type_): _description_
        aux_dir (_type_): _description_

    Returns:
        _type_: _description_
    """
    seviri_aux = xr.open_dataset(os.path.join(aux_dir, "seviri_aux_latlon.nc"),decode_times=False)
    seviri_ds = xr.merge([seviri_ds, seviri_aux[["lsm","satzen"]]],join="left").drop_vars(["lon0"])

    # mask  | seviri | dardar
    # coast | 1      | 2
    # sea   | 2      | 7
    # land  | 3      | 1
    
    seviri_ds["land_water_mask"] = seviri_ds.lsm.where(seviri_ds.lsm!=1,3).where(seviri_ds.lsm!=2,7).where(seviri_ds.lsm!=3,1)
    seviri_ds.land_water_mask.attrs = {'units': '1',
                                   'long_name': 'Land Water Mask from Calipso',
                                   'comment': 'Land Water Mask from Calipso files, indicating the surface type at the laser footprint 0=shallow ocean 1=land 2=coastlines 3=shallow inland water 4=intermittent water 5=deep inland water 6=continental ocean 7=deep ocean',
                                   'valid_range': np.array([0, 7], dtype=np.int16),
                                   'flag_values': np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int16),
                                   'flag_meanings': 'shallow_ocean land coastlines shallow_inland_water intermittent_water deep_inland_water continental_ocean deep_ocean'}

    return seviri_ds

def load_model(data_dir,
               model_conf_filepath,
               model_checkpoint_dir, 
               checkpoint_epoch=None):
    """initiate model and data module

    Args:
        data_dir (str): path to training patches
        model_conf_filepath (str): path to model config file
        model_checkpoint_dir (int): path to model checkpoint dir
        checkpoint_epoch (int, optional): load model at epoch checkpoint. Defaults to None → load latest model

    Returns:
        _type_: _description_
    """
    
    train_patch_ids = []
    val_patch_ids = []

    #load data module
    dm_overwrite_hparams = {"era5_variables": ["temperature"],
                             "data_dir": data_dir,
                             "dataloader_num_workers":4,
                             "train_patch_ids":train_patch_ids, 
                             "val_patch_ids":val_patch_ids,
                             "load_online":True,
                             "target_variable": ['iwc', 'icnc_5um'],
                             "seviri_channels": ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ch9', 'ch10', 'ch11'],
                             "fold_to_level_thickness":4,
                             "target_transform": LogTransform2D(constants=[1,1],scalers=[1e7,1e-2]),
                             "batch_size":4}

    dm = VerticalCloudDataModule(**dm_overwrite_hparams)
    dm.setup(stage="fit")
    
    model,conf = load_discriminator_model(model_conf_filepath=model_conf_filepath,
                                          model_checkpoint_dir=model_checkpoint_dir,
                                          epoch=checkpoint_epoch)
    model.prediction_heads = model.unet.prediction_heads

    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))

    return model, dm

def prep_input_data(ds_seviri, dm, patch_size=128):
    """create input data with the same shape as patches the model was trainded on 

    Args:
        ds_seviri (_type_): _description_
        dm (_type_): _description_
        patch_size (int, optional): _description_. Defaults to 128.

    Returns:
        tuple: input_data, input_data_chunks, meta_data_chunks, meta_data_input, height_levels

    """
    seviri_channels = dm.seviri_channels
    meta_data_helpers = ["lat","lon", "land_water_mask"]

    # create height level array
    level_thickness=60
    if dm.fold_to_level_thickness:
        level_thickness *= dm.fold_to_level_thickness
    height_levels  = get_height_level_range(min_height=dm.data_val.height_levels[1],max_height=dm.data_val.height_levels[0]-240,step=level_thickness)

    # create tensors
    input_data = torch.tensor(ds_seviri[seviri_channels].to_array().values)
    input_data = rearrange(input_data,'c t h w -> t c h w') # change channel and time dims
    input_data = dm.data_val.transform(input_data)

    meta_data_input = torch.tensor(ds_seviri[meta_data_helpers].to_array().values)
    meta_data_input = rearrange(meta_data_input,'c t h w -> t c h w') # change channel and time dims
    
    # create chunks with size of patches 
    chunk_size = (input_data.shape[0], len(dm.seviri_channels), patch_size, patch_size) # n_timesteps, seviri_channels, patch_size, patch_size
    chunks = []

    meta_data_chunks = (input_data.shape[0], 6)
    meta_data_chunks = []
    
    # Loop through the original tensor and create chunks
    for i in range(0, input_data.shape[-2], patch_size):
        for j in range(0, input_data.shape[-1], patch_size):
            # Extract a chunk from the original tensor
            chunk = input_data[:,:,i:i+chunk_size[-2], j:j+chunk_size[-1]]
            meta_data_chunk = get_seviri_meta_data_chunks(meta_data_input[:,:,i:i+chunk_size[-2], j:j+chunk_size[-1]],ds_seviri.sensing_stop)
            # Append the chunk to the list
            chunks.append(chunk)
            meta_data_chunks.append(meta_data_chunk)
        # print("\n")
    
    # Stack the chunks to create a new tensor
    # Stack the chunks to create a new tensor
    input_data_chunks = torch.stack(chunks, dim=0)
    meta_data_chunks = torch.stack(meta_data_chunks, dim=0)
    print("input data chunks shape",input_data_chunks.shape) # n patches, n timesteps, h, w
    print("meta data chunks shape", meta_data_chunks.shape) # n patches, n timesteps, 6

    return input_data, input_data_chunks, meta_data_chunks, meta_data_input, height_levels

def get_seviri_meta_data_chunks(meta_data_ts: torch.tensor, dt: xr.DataArray):
    """create patch meta data

    order of meta data: [latitude, longitude, month, day_night, hour, land_water_mask]
    
    Args:
        meta_data_ts: (timesteps, 3, 128, 128) 2nd dim: lat, lon, lwm
        dt: time dimension of seviri ds
    Returns:
        torch.tensor: timesteps, 6
    """
    # latlon
    latlon = torch.mean(meta_data_ts[:,:2],dim=(-1,-2))
    lwm = torch.mode(meta_data_ts[:,2].flatten(start_dim=-2,end_dim=-1)).values.unsqueeze(1)
    
    month = torch.tensor(dt.dt.month.values).unsqueeze(1)
    
    hour = torch.tensor(dt.dt.round("1h").dt.hour.values).unsqueeze(1)
    
    lat, lon = torch.mean(latlon,dim=0).numpy()
    
    szas = [get_sza(dt, lat, lon) for dt in dt]
    
    night_day_flag = (torch.tensor(szas) >= 80).long().unsqueeze(1)
    
    meta_data_chunk = torch.concatenate([latlon,month,night_day_flag,hour,lwm],dim=1)
    
    return meta_data_chunk

def get_sza(dt, lat, lon):
    """ get solar zenith angle
    
    source:
    https://user.eumetsat.int/s3/eup-strapi-media/Optimal_Cloud_Analysis_OCA_Release_1_Product_Users_Guide_5120a10382.pdf

    Args:
        dt (xr.DataArray): datetime dataarray
        lat (float): _description_
        lon (float): _description_

    Returns:
        np.ndarray: solar zenith angles
    """
    year = dt.dt.year
    month = dt.dt.month
    day = dt.dt.day
    hour = dt.dt.hour
    minute = dt.dt.minute
    # pixel location (decimal degree)

    b = datetime(int(year),int(month),int(day),int(hour),int(minute),0,0)
    # print('processing date:',b)
    # print('location: latitude=%.1f, longitude=%.1f' % (lat,lon) )
    
    sol_zen = np.arccos(astronomy.cos_zen(b,lon,lat))*180./np.pi
    # print('SZA=%.1f degrees' % sol_zen) 

    return sol_zen

def get_predictions(model, dm, seviri_patches, meta_data_patches, patch_size=128):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    y_hat_chunks = torch.empty(size=(seviri_patches.shape[0],seviri_patches.shape[1], model.prediction_heads, model.out_channels, patch_size,patch_size)).squeeze() # n patches, n timesteps, prediction heads, height levels, patch_size, patch_size
    
    model.eval()
    for i,patch in enumerate(zip(seviri_patches,meta_data_patches)):
        #print("input shape", p.shape,i)
        sev, md = patch

        with torch.no_grad():
            # if device == "cpu":
            y_hat_p = model(sev.to(device),md.to(device))     
            y_hat_p = dm.target_transform.inverse_transform(y_hat_p)#.cpu())
            #print("output shape", y_hat_p.shape)
        #print(y_hat_p.shape)
        y_hat_chunks[i] = y_hat_p

    print("predictions shape", y_hat_chunks.shape)

    return y_hat_chunks

def stich_patches_to_full_domain(input_data_full_domain, input_data_patches, model, y_hat_patches):
    patch_size = y_hat_patches.shape[-1]
    chunk_size = input_data_patches.shape[1:]
    
    # create empty full image target tensor
    target_size = (input_data_full_domain.shape[0],model.prediction_heads, model.out_channels, input_data_full_domain.shape[-2], input_data_full_domain.shape[-1]) # n_time steps, height level, H, W
    full_image_prediction = torch.empty(target_size)
    # create empty seviri input tensor → used for testing stich back
    full_image = torch.empty(input_data_full_domain.shape)
    
    # loop through chunks and fill target tensor (needs a lot of memory)
    chunk_id = 0
    for i in range(0, input_data_full_domain.shape[-2], patch_size):
        # print("i",i)
        for j in range(0, input_data_full_domain.shape[-1], patch_size):
            # print("j",j)
            full_image[:,:, i:i+chunk_size[-2], j:j+chunk_size[-1]] = input_data_patches[chunk_id] # just for control that tensors are stiched back together correctly
            full_image_prediction[:,:,:, i:i+chunk_size[-2], j:j+chunk_size[-1]] = y_hat_patches[chunk_id]
            chunk_id+=1

    if torch.isnan(input_data_full_domain).any():
        print(f"seviri input data contains {torch.isnan(input_data_full_domain).sum()} nan values → predictions will be nan as well")
        assert torch.isclose(full_image, input_data_full_domain,atol=0,equal_nan=True).all(), f"seviri input data ({input_data_full_domain}) does not equal stiched control image of seviri ({full_image.shape})"
    else:
        assert (full_image==input_data_full_domain).all(), f"seviri input data ({input_data_full_domain}) does not equal stiched control image of seviri ({full_image.shape})"

    # # change z and time dims to comply with order in  ds
    full_image_prediction = rearrange(full_image_prediction,'t pred_targets z h w -> pred_targets z t h w') 

    print("full image prediction shape", full_image_prediction.shape)

    return full_image_prediction

def split_by_chunks(dataset):
    """dask utility"""
    chunk_slices = {}
    for dim, chunks in dataset.chunks.items():
        slices = []
        start = 0
        for chunk in chunks:
            if start >= dataset.sizes[dim]:
                break
            stop = start + chunk
            slices.append(slice(start, stop))
            start = stop
        chunk_slices[dim] = slices
    for slices in itertools.product(*chunk_slices.values()):
        selection = dict(zip(chunk_slices.keys(), slices))
        yield dataset[selection]

def create_filepath(ds, prefix='parallel_test', root_path="./"):
    """
    Generate a filepath when given an xarray dataset
    """
    start = ds.time[0].dt.strftime("%Y%m%d:%H").data
    end = ds.time[-1].dt.strftime("%Y%m%d:%H").data
    filepath = f'{root_path}/ice_cloud_net_{prefix}_{start}.nc'
    return filepath

def create_icecloudnet_ds(ds_seviri, height_levels, full_image_prediction):
    # use seviri ds as base
    ds = ds_seviri

    # add height variable
    ds = ds.assign_coords(height=height_levels)

    # create icecloudnet variables
    level_thickness=240
    iwp = (torch.sum(full_image_prediction[0],dim=0)*level_thickness).detach() # calculate iwp
    iwc = full_image_prediction[0].detach()
    icnc_5um = full_image_prediction[1].detach()

    # rearrange dimensions
    iwc = rearrange(iwc, "height time y x -> time y x height")
    icnc_5um = rearrange(icnc_5um, "height time y x -> time y x height")

    # create hourly chunks
    iwc = da.from_array(iwc.numpy(), chunks=(4,*iwc.shape[1:]))
    iwp = da.from_array(iwp.numpy(), chunks=(4,*iwp.shape[1:]))
    icnc_5um = da.from_array(icnc_5um.numpy(), chunks=(4,*icnc_5um.shape[1:]))

    # add to dataset
    ds["iwc"] = (("sensing_stop","y","x","height"),iwc)
    ds["iwp"] = (("sensing_stop","y","x"),iwp)
    ds["icnc_5um"] = (("sensing_stop","y","x","height"),icnc_5um)

    # rename time dimension
    ds = ds.rename({"sensing_stop":"time"})

    # add attributes
    ds.icnc_5um.attrs = dict(units="m-3", long_name="number concentration of ice crystals larger than 5um in air")
    ds.iwc.attrs = dict(units="kg m-3", long_name="mass concentration of frozen water in air")
    ds.iwp.attrs = dict(units="kg m-2", long_name="integrated mass concentration of frozen water in air")
    ds.lon.attrs = dict(units="degree_east", standard_name="longitude")
    ds.lat.attrs = dict(units="degree_north",standard_name="latitude")
    ds.time.attrs = dict(standard_name="time", long_name="reference time", description="time of seviri observation (sensing stop time)")
    ds.height.attrs = dict(standard_name="height", units="m")
    ds.x.attrs = dict(standard_name="projection_x_coordinate",units="m")
    ds.y.attrs = dict(standard_name="projection_y_coordinate",units="m")

    ds["georef_offset_corrected"] = ds.georef_offset_corrected.assign_attrs(dict(units=1))

    ds_attrs = dict(title="IceCloudNet predictions of IWC and ICNC from Meteosat SEVIRI input",
        version="v1.0",
        creation_date=datetime.now().date().strftime("%Y-%m-%d"),
        source="File produced by IceCloudNet v0.1. Input files: " + ds.attrs['source_file'][:32] + "*.nat",
        author="Kai Jeggle (kai.jeggle@env.ethz.ch)",
        contact="Kai Jeggle (https://orcid.org/0000-0002-3098-9484)",
        references="tbd",
        institution="Institute for Atmosphere and Climate - ETH Zurich",
        history="File produced by IceCloudNet v0.1. Input files: " + ds.attrs['source_file'][:32] + "*.nat",
        conventions="CF-1.6 where applicable", 
        license = "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        proj4_string=ds.attrs["proj4_string"],
        )
    
    ds.attrs = ds_attrs

    # drop lower height levels if they exist
    ds = ds.sel(height=slice(16800,3800))

    return ds

def save_icecloudnet_ds(ds, target_dir, comp_level, helper_vars = ["lat","lon"]):
    # create directory for day 
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # save files in parallel using chunking and dask
    client = Client()

    # iwc
    datasets = list(split_by_chunks(ds[["iwc","iwp",*helper_vars]]))
    paths = [create_filepath(ds,"iwc", target_dir) for ds in datasets]
    #assert len(datasets) == ds.sizes["time"]/ds.iwc.chunksizes["time"][0], f"number of data sets ({len(datasets)}) does not equal time_dim/chunksize ({ds.sizes['time']}/{ds.iwc.chunksizes['time'][0]})"

    encoding=dict(iwc={"zlib":True,"complevel":comp_level,"_FillValue": 0},
              iwp={"zlib":True,"complevel":comp_level,"_FillValue": 0})

    xr.save_mfdataset(datasets=datasets, paths=paths,encoding=encoding) 

    # icnc
    datasets = list(split_by_chunks(ds[["icnc_5um",*helper_vars]]))
    paths = [create_filepath(ds,"icnc", target_dir) for ds in datasets]
    #assert len(datasets) == ds.sizes["time"]/ds.icnc_5um.chunksizes["time"][0], f"number of data sets ({len(datasets)}) does not equal time_dim/chunksize ({ds.sizes['time']} {ds.icnc_5um.chunksizes['time'][0]})"
    encoding=dict(icnc_5um={"zlib":True,"complevel":comp_level,"_FillValue": 0})
    xr.save_mfdataset(datasets=datasets, paths=paths,encoding=encoding)

def run(d, timestep_slice, model_conf_filepath, model_checkpoint_dir):
    start_time = datetime.now()

    # load seviri data
    ds_seviri = load_seviri_data(d,DATA_DIR)
    ds_seviri = ds_seviri.isel(sensing_stop=slice(*timestep_slice))
    print("loaded seviri")

    # load model and data module
    model, dm = load_model(DATA_DIR, 
                           model_conf_filepath=model_conf_filepath,
                           model_checkpoint_dir=model_checkpoint_dir,
                           checkpoint_epoch=143)
    print("loaded model")

    # prep input data
    input_data, input_data_chunks, meta_data_chunks, meta_data_input, height_levels = prep_input_data(ds_seviri, dm)
    print("created input data")

    # inference
    y_hat_patches = get_predictions(model, dm, input_data_chunks, meta_data_chunks)
    print("predictions made")

    full_image_prediction= stich_patches_to_full_domain(input_data, input_data_chunks, model, y_hat_patches)
    print("stiched predictions to full image")

    # delete tensors to free up space
    del y_hat_patches 
    del meta_data_input
    del meta_data_chunks
    del input_data
    del input_data_chunks

    gc.collect()

    icn_ds = create_icecloudnet_ds(ds_seviri, height_levels, full_image_prediction)
    print("created ice cloud net ds")

    target_dir = os.path.join(DATA_DIR,"WholeAreaPredictions",d)
    save_icecloudnet_ds(icn_ds, target_dir, comp_level=3)
    print("saved data sets")

    print(f"run time {datetime.now()-start_time}")

if __name__ == "__main__":
    # Define directory paths as global variables → adapt to your setup
    DATA_DIR = "/path/to/data/" 
    # must have the following subdirectories `WholeAreaPredictions` and files
    # - WholeAreaPredictions (to save IceCloudNet predictions)
    # - SEVIRIWholeAreaInput (seviri data for the whole domain)
    # - TrainingData (containing patches)
    # - data_stats.csv
    # - seviri_aux_latlon.nc

    # Define path to model config
    model_conf = "/path/to/model/config"

    # Define path to model checkpoint
    model_checkpoint_dir = "/path/to/model/checkpoint"

    parser = argparse.ArgumentParser(description="Process a date string and a timestep slice.")
    # Positional argument for timestep_slice as a tuple (smallest, largest)
    parser.add_argument(
        "timestep_slice", 
        type=lambda ts: eval(ts), 
        help="A tuple of smallest and largest timestep (e.g., '(0, 10)')"
    )

    # Parse the arguments
    args = parser.parse_args()
    
    # Access and print the parsed arguments
    print(f"Date: {args.date_string.strftime('%Y%m%d')}")
    print(f"Timestep slice: {args.timestep_slice}")
    
    run(args.date_string.strftime('%Y%m%d'), args.timestep_slice)
