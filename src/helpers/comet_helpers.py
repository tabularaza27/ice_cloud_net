import glob
import json
import io
import os
import re
from pprint import pprint
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import OrderedDict

# from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn.functional as F
import comet_ml

import matplotlib.pyplot as plt
import holoviews as hv

from data.data import VerticalCloudDataModule, LogTransform
from model.LightningModel import VerticalCloudCubeModel

COMET_API_KEY = "Rrwj142Sk080T0Qth3KNdPQg5"
COMET_WORKSPACE = "tabularaza27"
COMET_PROJECT = "verticalcloud"

dm_hparams_config = {'data_dir': str,
 'start_date': datetime.date,
 'end_date': datetime.date,
 'height_levels': tuple,
 'daytime_only': bool,
 'era5_variables': list,
 'target_transform': LogTransform,
 'seviri_channels': list,
 'batch_size': int,
 'prediction_type': str,
 'target_variable': list,
 'shuffle': bool,
 'load_online': bool,
 'dataloader_num_workers': int,
 'fold_to_level_thickness': int,
 'randomly_mask_ir_channels': float,
 'seviri_masking_constant': int}

model_hparams_config = {'seviri_channels': list,
 'out_channels': int,
 'prediction_heads': int,
 'zero_constant': np.float64,
 'criterion': Callable,
 'criterion_kwargs': dict,
 'auto_encoding': bool,
 'binary_prediction': bool,
 'residual': bool,
 'target_variable': str,
 'era5_variables': list,
 'cloud_thres': np.float64,
 'target_transform': LogTransform,
 'lr': float,
 'dim_mults': tuple,
 'final_relu': bool,
 'extended_final_conv': bool,
 'rff_params': dict,
 'additional_logging': bool,
 'clamp_output': bool,
 'meta_data_embedding': bool}

dm_overwrite_hparams = {"era5_variables": ["temperature"],
                         "data_dir": "",
                         "dataloader_num_workers":4,
                         "train_patch_ids":None, 
                         "val_patch_ids":None,
                         "load_online":True,
                         "target_transform": LogTransform(scaler=1e7),
                         "batch_size":64}
model_overwrite_hparams = {"criterion": F.l1_loss,"final_relu":True,"dim_mults":(1,2,4),"out_channels":256}

def load_experiment(experiment_name:str, project_name:str="verticalcloud") -> comet_ml.api.APIExperiment:
    """returns experiment object"""
    comet_api_endpoint = comet_ml.api.API(api_key=COMET_API_KEY)
    experiment = comet_api_endpoint.get("{}/{}/{}".format(COMET_WORKSPACE, project_name, experiment_name))

    return experiment

def get_asset_id(experiment_asset_dict, asset_key, asset_value):
    asset_id = next((asset["assetId"] for asset in experiment_asset_dict if asset[asset_key] == asset_value), None)
    return asset_id

def get_patch_ids(experiment_name: str) -> Tuple[List[str],List[str]]:
    experiment = load_experiment(experiment_name)
    experiment_assets = experiment.get_asset_list()

    # get train/val dates
    asset_id = get_asset_id(experiment_assets, "fileName", "val_patch_ids")
    val_patch_ids = experiment.get_asset(asset_id, return_type="json")

    asset_id = get_asset_id(experiment_assets, "fileName", "train_patch_ids")
    train_patch_ids = experiment.get_asset(asset_id, return_type="json")

    return train_patch_ids, val_patch_ids

def get_hparams(experiment, hparams_config,overwrite_params,print_hparams=False):
    hparams = dict()
    for param in experiment.get_parameters_summary():
        p_name = param["name"]
        if p_name not in hparams_config.keys():
            continue
        elif param["valueCurrent"] == "iwc":
            param_value = "iwc"
        elif param["valueCurrent"] == "icnc_5um":
            param_value = "icnc_5um"
        elif param["valueCurrent"] in ["None",'null']:
            param_value = None
        elif hparams_config[p_name] in (list,tuple,dict):
            param_value_str = param["valueCurrent"]
            param_value = json.loads(param_value_str)
        elif hparams_config[p_name] == bool:
            if param["valueCurrent"] == "false":
                param_value = False
            else:
                param_value = True
        elif hparams_config[p_name] == datetime.date:
            param_value = datetime.strptime(param["valueCurrent"],"%Y-%m-%d").date()
        elif hparams_config[p_name] in [int,float,str,np.float64]:
            param_value = hparams_config[p_name](param["valueCurrent"])
        elif hparams_config[p_name] == LogTransform:
            args = ["constant", "scaler"]
            args_dict = dict()
            input_string = param["valueCurrent"]
            for arg in args:
                match = re.search(fr'{arg}=(\d+\.*)', input_string)
                # Check if the pattern was found
                if match:
                    # Extract the matched value
                    args_dict[arg] = np.float64(match.group(1))
            param_value=LogTransform(**args_dict)
    
        else:
            print(f"No value for {p_name} available/possible to parse → set manually")
            param_value = "SetManually"
        hparams[p_name] = param_value

    # set params to overwrite
    print(overwrite_params)
    for key,val in overwrite_params.items():
        print(f"Set {key} manually with value {val}")
        hparams[key]=val
        
    if print_hparams:
        print("-------")
        print("hparams")
        print("-------")
        pprint(hparams,depth=1)
    return hparams

def get_trained_model(experiment, model_hparams):
    vertical_cloud_model = VerticalCloudCubeModel(**model_hparams)
    exp_key = experiment.get_metadata()["experimentKey"]
    try:
        # load local
        path = glob.glob(os.path.join("/cluster/work/climate/kjeggle/model_checkpoints", exp_key, "*.ckpt"))[0]
        pretrained_state_dict = torch.load(path)["state_dict"]
        print("loaded model from disk")
    except IndexError:
        # load from comet
        asset_id = experiment.get_model_asset_list("ice_net")[0]["assetId"]
        model_binary = experiment.get_asset(asset_id, return_type="binary")
        pretrained_state_dict = torch.load(io.BytesIO(model_binary))
        print("loaded model from comet")
    vertical_cloud_model.load_state_dict(pretrained_state_dict)
    
    return vertical_cloud_model

def get_data_module(experiment, dm_hparams):
    return VerticalCloudDataModule(**dm_hparams)

def get_dm(experiment_id, dm_overwrite_params,dm_hparams_config=dm_hparams_config):
    experiment = load_experiment(experiment_name=experiment_id)
    print("loaded experiment from comet")
    
    dm_hparams = get_hparams(experiment, 
                             print_hparams=True,
                             hparams_config=dm_hparams_config,
                             overwrite_params=dm_overwrite_params)
    
    dm = VerticalCloudDataModule(**dm_hparams)
    dm.setup(stage="fit")
    
    return dm 

def get_trained_experiment(experiment_id,
                           model_overwrite_params,
                           model_hparams_config=model_hparams_config):

    experiment = load_experiment(experiment_name=experiment_id)
    print("loaded experiment from comet")
       
    model_hparams = get_hparams(experiment, 
                                hparams_config=model_hparams_config, 
                                overwrite_params=model_overwrite_params,
                                print_hparams=False)
    
    model = VerticalCloudCubeModel(**model_hparams)

    # load local
    exp_key = experiment.get_metadata()["experimentKey"]
    try:
        path = glob.glob(os.path.join("/cluster/work/climate/kjeggle/model_checkpoints", exp_key, "*.ckpt"))[0]
        pretrained_state_dict = torch.load(path)["state_dict"]
        print("loaded model from disk")
    except IndexError:
        # load from comet
        asset_id = experiment.get_model_asset_list("ice_net")[0]["assetId"]
        model_binary = experiment.get_asset(asset_id, return_type="binary")
        if torch.cuda.is_available():
            pretrained_state_dict = torch.load(io.BytesIO(model_binary))
        else:
            pretrained_state_dict = torch.load(io.BytesIO(model_binary),map_location=torch.device('cpu'))
        print("loaded model from comet")
    if "module._forward_module." in next(iter(pretrained_state_dict.keys())):
        pretrained_state_dict = OrderedDict([(key.split("module._forward_module.")[-1],val) for key,val in pretrained_state_dict.items()])
    
    model.load_state_dict(pretrained_state_dict)
    
    return experiment, model

def log_image_to_comet(exp, p, image_name, log_kwargs=dict(overwrite=False)):
    fpath = f"/scratch/{image_name}.png"
    #hvplot
    if "holoviews" in str(type(p)):
        hv.save(p,fpath)
    # matplotlib
    else:
    # elif isinstance(p,plt.Figure):
        plt.savefig(fpath,dpi="figure")
        if isinstance(p,plt.Figure):
            plt.close(p)
        else:
            # seaborn
            plt.close(p.fig)
        
    exp.log_image(fpath, image_name,**log_kwargs)