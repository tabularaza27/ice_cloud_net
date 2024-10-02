import os
from datetime import datetime, date
import glob
from typing import Any, Callable, List, Optional, Tuple, Union

import xarray as xr
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
import pytorch_lightning as pl
from sklearn.base import BaseEstimator, TransformerMixin
import einops

from data.data_utils import load_patches, get_overpass_direction, sort_overpass_indices, combine_seviri_and_integrated_era5_channels, get_patch_meta_data

class VerticalCloudParentDataset(VisionDataset):
    """Dataset to predict singular columns from seviri and era5 data

    ToDo:
        Add meta data to identify each patch (date, x/y coordinates)
        Add other data like topography, viewing angle, land water mask
        Currently not very efficient with for loop through all patches -> vectorize    
        Add saving tensors to disk -> possibility to load precomputed tensors directly instead of patches
    """
    def __init__(self,
                 root: str,
                 date_strings: List[str],
                 load_online: bool = False,
                 seviri_channels: List[str] = ["ch9","ch10"],
                 target_variable: str ="iwc",
                 era5_variables: List[str]= ["temperature","u_velocity","v_velocity"],
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 era5_transform: Optional[Callable] = None,
                 height_levels: Optional[Tuple[int, int]] = (16980, 1680),
                 patches: Optional[List[xr.Dataset]] = None,
                 patch_ids: Optional[List[str]] = None,
                 daytime_only: bool = False,
                 fold_to_level_thickness = None,
                 drop_cloud_free_profiles: bool = False,
                 load_aux_patch: bool = True,
                 randomly_mask_ir_channels: float = 0,
                 seviri_masking_constant: int = -9
                 ) -> None:
        super().__init__(root, 
                         transform=transform, 
                         target_transform=target_transform)
        
        self.seviri_aux_data = xr.open_dataset(os.path.join(os.path.split(root)[0],"seviri_aux.nc"),decode_times=False) # todo make dynamic# os.path.join(root, "seviri_aux.nc"))
        self.used_patch_indices = []

        self.load_online = load_online
        self.date_strings = date_strings
        self.seviri_channels = seviri_channels
        self.target_variable = target_variable
        self.era5_variables = era5_variables
        self.era5_transform = era5_transform
        self.height_levels = height_levels
        self.patches = patches
        self.daytime_only = daytime_only
        self.fold_to_level_thickness = fold_to_level_thickness
        self.drop_cloud_free_profiles = True
        self.load_aux_patch = load_aux_patch
        self.randomly_mask_ir_channels = randomly_mask_ir_channels # randomly mask ir channels during day
        self.seviri_masking_constant = seviri_masking_constant # int to mask seviri values after normalization, i.e. vis channels during night
        
        # check for vis channels → they need to be masked for night time observations
        try:
            self.vis_ch_idx = [self.seviri_channels.index(ch) for ch in ['ch1','ch2','ch3']]
            self.max_vis_ch_idx = max(self.vis_ch_idx) # use max index of vis channels to select vis channels in __getitem__()
        except ValueError:
            self.vis_ch_idx = []
            self.max_vis_ch_idx = None

        if self.load_online:
            if patch_ids is None:
                # load and filter patch_ids based on given date strings
                self.patch_ids = self._get_patch_ids()
            else:
                # use given patch_ids (already filtered)
                self.patch_ids = patch_ids
                print(f"loaded {len(patch_ids)} patch ids")
        else:
            self.seviri_data, self.era5_data, self.dardar_data, self.overpass_indices, self.meta_data = self._load_data() # seviri_data, era5_data, dardar_data, overpass_indices

            assert not np.isnan(self.seviri_data).any(), "Seviri data contains nan values"
            assert not np.isnan(self.era5_data).any(), "ERA5 data contains nan values"
            assert not np.isnan(self.dardar_data).any(), "DARDAR data contains nan values"
            assert not np.isnan(self.overpass_indices).any(), "Overpass indices contains nan values"

    def _load_data(self) -> Tuple[Any, Any, Any, Any, Any]:
        """Create Input/Target tensors

        Used when loading all patches into memory

        Returns:
            Tuple[Any, Any, Any, Any]: seviri, era5, dardar, overpass_indices, meta_data
        """
        raise NotImplementedError

    def _load_single_patch(self, patch_id: str) -> Tuple[Any, Any, Any, Any, Any]:
        """Create Input/Target tensors

        Used when loading single patch online during experiment

        Returns:
            Tuple[Any, Any, Any, Any, Any]: seviri, era5, dardar, overpass_indices, meta_data
        """
        raise NotImplementedError

    def _get_patch_ids(self) -> List: 
        """Retrieve patch ids for all patches given self.date_strings
        
        patch id is defined as `date_string/patch_N.nc`
        """
        # loop through date directories, load each patch and check if it has to be filtered
        # todo: I could have a list with preloaded ids to avoid looping and loading all patches every experiment
        patch_ids = []
        filtered_patch_counter = 0
        for date_string in self.date_strings:
            date_patch_ids = glob.glob(os.path.join(self.root, date_string,"*nc"))
            # loop throug each patch and check if it has to be filtered
            for p in date_patch_ids:
                patch = xr.open_dataset(p)
                
                # slice height levels as this influences the filtering conditions
                if self.height_levels is not None:
                    patch = patch.sel(height=slice(*self.height_levels))
                # check filtering conditions
                if self._filter_out_patch(patch):
                    filtered_patch_counter+=1
                else:
                    p_id = os.path.join(*p.split("/")[-2:]) # only use last 2 path components as id
                    patch_ids.append(p_id)
                    
        print(f"loaded {len(patch_ids)} patch ids")
        print(f"filtered out {filtered_patch_counter}/{len(patch_ids)+filtered_patch_counter} patch ids")

        return patch_ids
           
    def _filter_out_patch(self, patch: xr.Dataset) -> bool:
        """Returns True for patches to be filrered our, else False

        Args:
            patch (xr.Dataset): _description_
        """
        # filter out patch if nan values exist in era5 data -> currently brute force approach, think about interpolating in the future or filter only columns
        
        # filter for era5 variables (currently along dardar track)
        if self.era5_variables is not None:
            if patch[self.era5_variables[0]].isnull().any().values == True:
                return True

        # patches with nan values in integrated era5 variables  # todo dont have variable name hardcoded  
        if "era5_ciwc_cirrus" in self.seviri_channels:
            if patch["era5_ciwc_cirrus"].isnull().any().values == True:
                return True

        # filter out patches with nan dardar values
        if patch[self.target_variable].isnull().any().values == True:
            return True
        
        # filter out patches with nan seviri values
        if patch[self.seviri_channels[:1]].isnull().any().to_array().any().values == True:
            return True
        
        # filter out night time (1) observations if daytime_only
        if self.daytime_only:
            if (patch.day_night_flag == 1).any():
                return True

        # filter out patches without any dardar data
        if self.drop_cloud_free_profiles:
            if (patch["iwc"]==0).all().values == True:
                return True
        
        return False

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (seviri_patch, era5_column, dardar_column, overpass_mask, meta_data, index) 
        """
        if self.load_online:
            patch_id = self.patch_ids[index]
            seviri, era5, dardar, overpass_index, meta_data = self._load_single_patch(patch_id)
        else:
            seviri, era5, dardar, overpass_index, meta_data = self.seviri_data[index], self.era5_data[index], self.dardar_data[index], self.overpass_indices[index], self.meta_data[index]
        
        # get nighttime idxs
        night_time = meta_data[3]
        seviri = torch.from_numpy(seviri) # torchvision transform is used for seviri -> requires torch.Tensor as input
            
        # set vis channels to zero for night time idx
        #if (night_time == 1) and (self.max_vis_ch_idx is not None):
        #    seviri[:self.max_vis_ch_idx+1] = torch.clamp(seviri[:self.max_vis_ch_idx+1],-1,-1)

        if self.transform is not None: 
            # seviri = torch.nan_to_num(seviri, nan=0) # todo remove once there are no nan values in seviri anymore
            seviri = self.transform(seviri)

        # set vis channels to -99 for night time idx
        if (night_time == 1) and (self.max_vis_ch_idx is not None) and (self.seviri_masking_constant is not None):
            seviri[:self.max_vis_ch_idx+1] = torch.clamp(seviri[:self.max_vis_ch_idx+1],self.seviri_masking_constant,self.seviri_masking_constant)

        # mask ir channels with probability `randomly_mask_ir_channels` during they day and if visible channels are available
        # rationale is that the model then focuses to learn from vis channels
        if (night_time == 0) and (self.max_vis_ch_idx is not None):
            if np.random.random(1) <= self.randomly_mask_ir_channels:
                seviri[self.max_vis_ch_idx+1:] = torch.clamp(seviri[self.max_vis_ch_idx+1:],self.seviri_masking_constant,self.seviri_masking_constant)

        if self.era5_transform is not None:
            era5 = self.era5_transform(era5)

        if self.fold_to_level_thickness:
            dardar = torch.mean(torch.tensor(dardar).unfold(-3,self.fold_to_level_thickness,self.fold_to_level_thickness),dim=-1) # grid mean

        if self.target_transform is not None:
            dardar = self.target_transform(dardar) 

        return seviri, era5, dardar, overpass_index, meta_data, index

    def __len__(self) -> int:
        if self.load_online:
            return len(self.patch_ids)
        else:
            return self.seviri_data.shape[0]


class VerticalCloudSingleColumnDataset(VerticalCloudParentDataset):
    """Dataset to predict singular columns from seviri and era5 data"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def _load_data(self):
        if self.patches is None:
            self.patches = load_patches(self.date_strings, self.root) # load patches as xarray datasets -> List[xr.Dataset]

        seviri_data = []
        era5_data = []
        dardar_data = []
        overpass_indices = []

        filtered_patch_counter = 0
        for i, patch in enumerate(self.patches):
            # select height
            if self.height_levels is not None:
                patch = patch.sel(height=slice(*self.height_levels))

            # filter out "bad" patches
            if self._filter_out_patch(patch):
                filtered_patch_counter+=1
                continue
            else:
                self.used_patch_indices.append(i)

            # seviri data
            seviri_patch_data = patch[self.seviri_channels].to_array().values
            seviri_patch_data = np.repeat(seviri_patch_data[np.newaxis, :, :, :], patch.xy.size, axis=0) # n_profiles, n_channels, x, y
            seviri_data.append(seviri_patch_data)
        
            # era5 data
            era5_patch_data = np.transpose(patch[self.era5_variables].to_array().values,(1,0,2)) # n_profiles, n_variables, n_height_levels
            era5_data.append(era5_patch_data)
            
            # dardar data (target)
            dardar_patch_data = patch[self.target_variable].values # n_profiles, n_height_levels
            dardar_data.append(dardar_patch_data)

            # overpass - as indices in patch grid 
            x_indices, _ , y_indices = np.where((patch.x_nearest==patch.x)&(patch.y_nearest==patch.y)) # _ are the indices of the xy vector 
            overpass_patch_indices = np.column_stack((x_indices, y_indices)) # n_profiles, 2
            overpass_indices.append(overpass_patch_indices)
        print(f"filtered out {filtered_patch_counter} bad quality patches")

        seviri_data = np.concatenate(seviri_data)
        era5_data = np.concatenate(era5_data)
        dardar_data = np.concatenate(dardar_data)
        overpass_indices = np.concatenate(overpass_indices)

        assert seviri_data.shape[0] == era5_data.shape[0] == dardar_data.shape[0] == overpass_indices.shape[0], f"shapes of data does not match seviri {seviri_data.shape}, era5 {era5_data.shape}, dardar {dardar_data.shape}, overpass {overpass_indices.shape}"

        return seviri_data, era5_data, dardar_data, overpass_indices

class VerticalCloudCubeDataset(VerticalCloudParentDataset):
    """Dataset to predict dardar from seviri and aggregated era5 data"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _load_data(self):
        if self.patches is None:
            self.patches = load_patches(self.date_strings, self.root) # load patches as xarray datasets -> List[xr.Dataset]
        else:
            print("patches already loaded")

        seviri_data = []
        era5_data = []
        dardar_data = []
        overpass_indices = []
        meta_data = []

        filtered_patch_counter = 0

        if len(self.patches[0][self.target_variable].coords) == 2:
            height_resolved = True
        else:
            height_resolved = False

        for i, patch in enumerate(self.patches):

            # add auxiliary data to each patch (could be done in patch generation)
            y_min=patch.y.min()
            y_max=patch.y.max()
            x_min=patch.x.min()
            x_max=patch.x.max()

            aux_patch = self.seviri_aux_data.sel(y=slice(y_min,y_max),x=slice(x_max,x_min))[["lsm","alt","pixel_area","satzen"]] # todo double check aux data specs lon0 and offset correction
            patch = xr.merge([patch, aux_patch])

            # calulate cirrus/mixed iwp if necessary (could be done in patch generation):
            if "cirrus_iwp" in self.target_variable and "cirrus_iwp" not in patch.data_vars:
                cirrus_iwc = patch["iwc"].where(patch.temperature<235.15,other=0)
                mixed_iwc = patch["iwc"].where(patch.temperature>=235.15,other=0) 
                patch["cirrus_iwp"] = np.sum(cirrus_iwc, 1) * 60
                patch["mixed_iwp"] = np.sum(mixed_iwc, 1) * 60


            # select height
            if self.height_levels is not None:
                patch = patch.sel(height=slice(*self.height_levels))

            # filter out "bad" patches
            if self._filter_out_patch(patch):
                filtered_patch_counter+=1
                continue

            self.used_patch_indices.append(i)

            # seviri data
            if "era5_ciwc_cirrus" in self.seviri_channels:
                # quickfix of treating era5 integrated values as seviri channel. -> can use same architecture for this
                seviri_patch_data = combine_seviri_and_integrated_era5_channels(patch, self.seviri_channels)
            else:
                seviri_patch_data = patch[self.seviri_channels].to_array().values # n_channels, x, y
            seviri_data.append(np.expand_dims(seviri_patch_data,0))

            # era5 data - take mean across whole patch as era data is much coarser than seviri and almost the same across whole patch
            era5_patch_data = patch[self.era5_variables].to_array().mean(dim="xy").values #  n_era5_variables, n_height_levels
            era5_data.append(np.expand_dims(era5_patch_data,0))
            
            # overpass - as indices in patch grid 
            x_indices, _ , y_indices = np.where((patch.x_nearest==patch.x)&(patch.y_nearest==patch.y)) # _ are the indices of the xy vector 
            overpass_patch_indices = np.column_stack((x_indices, y_indices)) # n_profiles, 2
            overpass_direction = get_overpass_direction(patch)
            overpass_patch_indices = sort_overpass_indices(overpass_patch_indices, overpass_direction)
            overpass_mask = np.zeros(shape=(patch.x.size,patch.y.size)) # x,y
            overpass_mask[overpass_patch_indices[:,1],overpass_patch_indices[:,0]] = 1 # set dardar observations to 1. x coords (along columns) y coords (along rows)
            overpass_indices.append(np.expand_dims(overpass_mask,0)) # actually this is 2d map of the overpass and not only the indices; todo: naming

            # dardar data (target)
            if isinstance(self.target_variable, list):
                n_target_vars = len(self.target_variable)
                dardar_overpass = patch[self.target_variable].to_array().values.T # n_profiles, n_variables
            else:
                n_target_vars = 1
                dardar_overpass = patch[self.target_variable].values

            if height_resolved:
                dardar_patch_data = np.zeros(shape=(patch.x.size,patch.y.size,patch.height.size)) # initialize cube x, y, height
                dardar_patch_data[overpass_patch_indices[:,1],overpass_patch_indices[:,0]] = dardar_overpass # set dardar overpass value in cube
            else:
                # use integrated variable, i.e. one vertical level
                dardar_overpass = np.expand_dims(dardar_overpass,1) # n_profiles, n_variables
                dardar_patch_data = np.zeros(shape=(patch.x.size,patch.y.size,n_target_vars)) # initialize cube x, y, height
                dardar_patch_data[overpass_patch_indices[:,1],overpass_patch_indices[:,0]] = dardar_overpass # set dardar overpass value on plane
            
            dardar_data.append(np.expand_dims(dardar_patch_data,0))

            # load meta data
            md = get_patch_meta_data(patch)
            meta_data.append(np.expand_dims(md,0))

        print(f"filtered out {filtered_patch_counter} bad quality patches")

        seviri_data = np.concatenate(seviri_data)
        era5_data = np.concatenate(era5_data)
        dardar_data = np.concatenate(dardar_data)
        overpass_indices = np.concatenate(overpass_indices)
        meta_data = np.concatenate(meta_data)

        assert seviri_data.shape[0] == era5_data.shape[0] == dardar_data.shape[0] == overpass_indices.shape[0] == meta_data.shape[0], f"shapes of data does not match seviri {seviri_data.shape}, era5 {era5_data.shape}, dardar {dardar_data.shape}, overpass {overpass_indices.shape}, meta data {meta_data.shape}"

        return seviri_data, era5_data, dardar_data, overpass_indices, meta_data

    def _load_single_patch(self, patch_id):
        patch_path = os.path.join(self.root, patch_id)
        patch = xr.open_dataset(patch_path)

        if len(patch[self.target_variable].coords) == 2:
            height_resolved = True
        else:
            height_resolved = False

        # height_resolved = True if "iwc" in self.target_variable else False # True for iwc, False for iwp

        # add auxiliary data to each patch (could be done in patch generation)
        # todo check how long this merge takes 
        y_min=patch.y.min()
        y_max=patch.y.max()
        x_min=patch.x.min()
        x_max=patch.x.max()

        if self.load_aux_patch:
            aux_patch = self.seviri_aux_data.sel(y=slice(y_min,y_max),x=slice(x_max,x_min))[["lsm","alt","pixel_area","satzen"]] # todo double check aux data specs lon0 and offset correction
            patch = xr.merge([patch, aux_patch])

        # calulate cirrus/mixed iwp if necessary (could be done in patch generation):
        if "cirrus_iwp" in self.target_variable and "cirrus_iwp" not in patch.data_vars:
            cirrus_iwc = patch["iwc"].where(patch.temperature<235.15,other=0)
            mixed_iwc = patch["iwc"].where(patch.temperature>=235.15,other=0) 
            patch["cirrus_iwp"] = np.sum(cirrus_iwc, 1) * 60
            patch["mixed_iwp"] = np.sum(mixed_iwc, 1) * 60


        # select height
        if self.height_levels is not None:
            patch = patch.sel(height=slice(*self.height_levels))

        # seviri data
        if "era5_ciwc_cirrus" in self.seviri_channels:
            # quickfix of treating era5 integrated values as seviri channel. -> can use same architecture for this
            seviri_patch_data = combine_seviri_and_integrated_era5_channels(patch, self.seviri_channels)
        else:
            seviri_patch_data = patch[self.seviri_channels].to_array().values # n_channels, x, y

        # era5 data - take mean across whole patch as era data is much coarser than seviri and almost the same across whole patch
        era5_patch_data = patch[self.era5_variables].to_array().mean(dim="xy").values #  n_era5_variables, n_height_levels
        
        # overpass - as indices in patch grid 
        x_indices, _ , y_indices = np.where((patch.x_nearest==patch.x)&(patch.y_nearest==patch.y)) # _ are the indices of the xy vector 
        overpass_patch_indices = np.column_stack((x_indices, y_indices)) # n_profiles, 2
        overpass_direction = get_overpass_direction(patch)
        overpass_patch_indices = sort_overpass_indices(overpass_patch_indices, overpass_direction)
        overpass_mask = np.zeros(shape=(patch.x.size,patch.y.size)) # x,y
        overpass_mask[overpass_patch_indices[:,1],overpass_patch_indices[:,0]] = 1 # set dardar observations to 1. x coords (along columns) y coords (along rows)

        # dardar data (target)
        
        # get swath data
        if isinstance(self.target_variable, list):
            n_target_vars = len(self.target_variable)
            dardar_overpass = patch[self.target_variable].to_array().values.T # n_profiles, n_variables (2d) 
            if height_resolved: 
                dardar_overpass = einops.rearrange(dardar_overpass,"z n c -> n z c") # n_profiles, height, n_variables (3d)
        else:
            n_target_vars = 1
            dardar_overpass = patch[self.target_variable].values
        
        # co-locate on grid    
        if height_resolved:
            if n_target_vars > 1:
                dardar_patch_data = np.zeros(shape=(patch.x.size,patch.y.size,patch.height.size,n_target_vars)) # initialize cube x, y, height
            else:
                dardar_patch_data = np.zeros(shape=(patch.x.size,patch.y.size,patch.height.size)) # initialize cube x, y, height, n_variables
            dardar_patch_data[overpass_patch_indices[:,1],overpass_patch_indices[:,0]] = dardar_overpass # set dardar overpass value in cube
        else:
            # use integrated variable, i.e. one vertical level
            if len(dardar_overpass.shape) == 1:
                dardar_overpass = np.expand_dims(dardar_overpass,1) # n_profiles, n_variables
            dardar_patch_data = np.zeros(shape=(patch.x.size,patch.y.size,n_target_vars)) # initialize cube x, y, height
            dardar_patch_data[overpass_patch_indices[:,1],overpass_patch_indices[:,0]] = dardar_overpass # set dardar overpass value on plane
        
        # permute dardar to correct shape
        # desired shape: C x Z X H X W
        if len(dardar_patch_data.shape) == 4:
            # height resolved & multple variables
            dardar_patch_data = einops.rearrange(dardar_patch_data,"H W Z C -> C Z H W")
        elif len(dardar_patch_data.shape) == 3:
            # either height resolved and 1 variable or 2d and multiple variables
            dardar_patch_data = einops.rearrange(dardar_patch_data,"H W C -> C H W")

        # get patch metadata
        meta_data = get_patch_meta_data(patch)

        return seviri_patch_data, era5_patch_data, dardar_patch_data, overpass_mask, meta_data


class VerticalCloudDataModule(pl.LightningDataModule):
    """Implement, this is boilerplate code

    Args:
        pl (_type_): _description_
    """
    def __init__(self, 
                 data_dir: str = "./",
                 load_online: bool = False,
                 start_date: Optional[date] = None,
                 end_date: Optional[date] = None,
                 seviri_channels: List[str] = ["ch9","ch10"],
                 target_variable: str ="iwc",
                 era5_variables: List[str]= ["temperature","u_velocity","v_velocity"],
                 target_transform: Union[Callable, None] = None,
                 height_levels: Tuple[int, int] = (16980, 1680),
                 batch_size: int = 16,
                 prediction_type: str = "cube", # ["cube", "column"],
                 daytime_only: bool = False,
                 train_patches: Optional[list] = None,
                 val_patches:Optional[list] = None,
                 test_patches: Optional[list] = None,
                 train_patch_ids: Optional[List[str]] = None,
                 val_patch_ids: Optional[List[str]] = None,
                 test_patch_ids: Optional[List[str]] = None,
                 seed: Optional[int] = 13,
                 dataloader_num_workers: int = 4,
                 shuffle: bool = False,
                 fold_to_level_thickness: Optional[int] = None,
                 randomly_mask_ir_channels: float = 0,
                 seviri_masking_constant: int = -9
                 ):
        super().__init__()

        if seed:
            pl.seed_everything(seed)

        if prediction_type == "cube":
            self.dataset_class = VerticalCloudCubeDataset
        elif prediction_type == "column":
            self.dataset_class = VerticalCloudSingleColumnDataset,
        else:
            raise ValueError("specify correct prediction type")

        self.load_online = load_online
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seviri_channels = seviri_channels
        self.era5_variables = era5_variables
        self.target_variable = target_variable
        self.height_levels = height_levels
        self.start_date = start_date
        self.end_date = end_date
        self.daytime_only = daytime_only
        self.target_transform = target_transform
        self.shuffle = shuffle
        self.dataloader_num_workers = dataloader_num_workers
        self.fold_to_level_thickness = fold_to_level_thickness
        self.randomly_mask_ir_channels = randomly_mask_ir_channels # randomly mask ir channels during day
        self.seviri_masking_constant = seviri_masking_constant # int to mask seviri values after normalization, i.e. vis channels during night

        # transform of seviri data using torch vision transforms
        self.transform = transforms.Normalize(mean=get_variable_stat(self.seviri_channels, "mean", stats_dir=self.data_dir, daytime=self.daytime_only), std=get_variable_stat(self.seviri_channels, "std", stats_dir=self.data_dir, daytime=self.daytime_only))
        # log transform target variable, add constant to handle 0 values (min in-cloud value/10) 
        self.target_transform = self.target_transform#get_variable_stat("iwc","min", stats_dir=self.data_dir)/10)  # type: ignore
        # standard scaler for each variable independently
        self.era5_transform = MultiFeatureScaler(means=get_variable_stat(self.era5_variables, "mean", stats_dir=self.data_dir), # type: ignore
                                                 stds=get_variable_stat(self.era5_variables, "std", stats_dir=self.data_dir)) # type: ignore
        
        # in case it exists, pass preloaded patches
        self.train_patches = train_patches
        self.val_patches = val_patches
        self.test_patches = test_patches

        # in case it exists, pass preloaded patch ids
        self.train_patch_ids = train_patch_ids
        self.val_patch_ids = val_patch_ids
        self.test_patch_ids = test_patch_ids

        self.data_train = None
        self.data_test = None
        self.data_val = None
        self.data_predict = None

        self.training_data_dir = "TrainingData" if "era5_ciwc_cirrus" not in self.seviri_channels else "TrainingDataERAIntegratedNN"

    def prepare_data(self):
        # download
        pass

    def setup(self, stage: str):
        # check if setup was already run
        if stage == 'fit' and self.data_train is not None:
            return
        if stage == 'test' and self.data_test is not None:
            return
        if stage is None and self.data_train is not None and self.data_test is not None:
            return

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # split train/val
            dates_to_load = self._get_dirs_to_load(stage=stage)
            self.train_date_strings = list(np.sort(np.random.choice(dates_to_load, int(0.8*len(dates_to_load)),replace=False))) # 80% train / 20% val
            self.val_date_strings = list(np.sort(np.array([d for d in dates_to_load if d not in self.train_date_strings])))
            
            self.data_train = self.dataset_class(root=os.path.join(self.data_dir, self.training_data_dir), # todo make dynamic
                                                               load_online=self.load_online, 
                                                               date_strings=self.train_date_strings,
                                                               seviri_channels = self.seviri_channels,
                                                               era5_variables = self.era5_variables,
                                                               target_variable = self.target_variable,
                                                               transform=self.transform,
                                                               era5_transform=self.era5_transform,
                                                               target_transform=self.target_transform,
                                                               height_levels=self.height_levels,
                                                               patches=self.train_patches,
                                                               patch_ids=self.train_patch_ids,
                                                               daytime_only=self.daytime_only,
                                                               fold_to_level_thickness=self.fold_to_level_thickness,
                                                               randomly_mask_ir_channels=self.randomly_mask_ir_channels,
                                                               seviri_masking_constant=self.seviri_masking_constant)            # type: ignore
            self.data_val = self.dataset_class(root=os.path.join(self.data_dir, self.training_data_dir), 
                                                              load_online=self.load_online,  
                                                              date_strings=self.val_date_strings,
                                                              seviri_channels = self.seviri_channels,
                                                              era5_variables = self.era5_variables,
                                                              target_variable = self.target_variable,
                                                              transform=self.transform,
                                                              era5_transform=self.era5_transform,
                                                              target_transform=self.target_transform,
                                                              height_levels=self.height_levels,
                                                              patches=self.val_patches,
                                                              patch_ids=self.val_patch_ids,
                                                              daytime_only=self.daytime_only,
                                                              fold_to_level_thickness=self.fold_to_level_thickness,
                                                              randomly_mask_ir_channels=0,
                                                              seviri_masking_constant=self.seviri_masking_constant)             # type: ignore

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            dates_to_load = self._get_dirs_to_load(stage=stage)
            self.data_test = self.dataset_class(root=os.path.join(self.data_dir, "TestData"), 
                                                              date_strings=list(dates_to_load),
                                                              seviri_channels = self.seviri_channels,
                                                              era5_variables = self.era5_variables,
                                                              target_variable = self.target_variable,
                                                              transform=self.transform,
                                                              era5_transform=self.era5_transform,
                                                              target_transform=self.target_transform,
                                                              height_levels=self.height_levels)       # type: ignore

        if stage == "predict":
            dates_to_load = self._get_dirs_to_load(stage=stage)
            self.data_predict = self.dataset_class(root=os.path.join(self.data_dir, "TestData"), 
                                                              date_strings=list(dates_to_load),
                                                              seviri_channels = self.seviri_channels,
                                                              era5_variables = self.era5_variables,
                                                              target_variable = self.target_variable,
                                                              transform=self.transform,
                                                              era5_transform=self.era5_transform,
                                                              target_transform=self.target_transform,
                                                              height_levels=self.height_levels)   # type: ignore

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size,shuffle=self.shuffle, pin_memory=False, num_workers=self.dataloader_num_workers) # type: ignore

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size,shuffle=False, pin_memory=False, num_workers=self.dataloader_num_workers) # type: ignore

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size,shuffle=False, pin_memory=False, num_workers=self.dataloader_num_workers) # type: ignore

    def predict_dataloader(self):
        return DataLoader(self.data_predict, batch_size=self.batch_size,shuffle=False, pin_memory=False, num_workers=self.dataloader_num_workers) # type: ignore
    
    def _get_dirs_to_load(self, stage) -> np.ndarray:
        """Determines which dates to load, each date is a subdirectory containing co-located patches of the given day

        Args:
            stage (_type_): _description_

        Returns:
            np.ndarray[str]: array of date strings %Y_%m_%d
        """
        if stage == "fit":
            sub_data_dir = "TrainingData"
        else:
            sub_data_dir = "TestData"

        # select subdirectories
        all_available_date_strings = np.array([os.path.basename(fpath) for fpath in glob.glob(os.path.join(self.data_dir,sub_data_dir,"*"))])
        all_available_dates = np.array([datetime.strptime(dstr, "%Y_%m_%d").date() for dstr in all_available_date_strings])
        if self.start_date and self.end_date:
            # select date range
            date_idxs_to_load = np.where((all_available_dates>=self.start_date)&(all_available_dates<=self.end_date))[0]
            return all_available_date_strings[date_idxs_to_load]
        else:
            # use all available data
            return all_available_date_strings
    
def get_variable_stat(var_names: Union[str,List[str]], 
                      stat: str, 
                      stats_dir: str ="/net/n2o/wolke_scratch2/kjeggle/VerticalCloud/Data",
                      daytime: bool = False) -> Union[float, List]:
    """Return precomputed statistics of dataset

    Args:
        var_names (Union[str,List]): seviri, era5, dardar variables
        stat (str): mean, min, max, median, var, std
        stats_file (str, optional): filepath to precomputed statistics. Defaults to /net/n2o/wolke_scratch2/kjeggle/VerticalCloud/Data".
        daytime (bool): if only daytime data is used
    Returns:
        Union[float, List]: _description_
    """

    fname = "data_stats_daytime" if daytime else "data_stats" # data_stats contains data from both day/night
    stats_df = pd.read_csv(os.path.join(stats_dir, f"{fname}.csv"),index_col=0)
    stats = stats_df.loc[stat, var_names] # type: ignore
    
    if isinstance(stats, pd.Series):
        # return stat for multiple variables as list
        stats = stats.to_list()
    
    return stats

class LogTransform(BaseEstimator, TransformerMixin):
    """Log Transform by first adding constant and then taking log10"""
    def __init__(self, constant=1, scaler=1):
        self.constant = constant # constant to be added to each value befor log transform
        self.scaler = scaler # each value is multiplied by scaler befor log transform and constant added

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X * self.scaler

        if isinstance(X, torch.Tensor):
            return torch.log10(X + self.constant)
        else:
            return np.log10(X + self.constant)

    def inverse_transform(self, X):
        if isinstance(X, torch.Tensor):
            return (torch.pow(10, X) - self.constant) / self.scaler
        else:
            return (np.power(10, X) - self.constant) / self.scaler

        # return (10 ** X - self.constant) / self.scaler
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.transform(*args, **kwargs)
    
class LogTransform2D(BaseEstimator, TransformerMixin):
    """Log Transform for tensor with dimensions (C x Z x H x W) with separate transforms for each channel"""
    def __init__(self, constants=None, scalers=None):
        self.constants = constants if constants is not None else [1] * 2  # Default constants
        self.scalers = scalers if scalers is not None else [1e7, 1e-2]  # Default scalers
        
        self.transforms = [LogTransform(constant=c,scaler=s) for c,s in zip(self.constants,self.scalers)] # type: ignore
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if len(X.shape)==5 or len(X.shape)==3:
            # (N X C x Z x H x W)
            # (overpass pixel x C x Z)
            channel_dim = 1
            
        else:
            # (C x Z x H x W)
            channel_dim = 0
        
        assert len(self.constants) == X.size(channel_dim), "Number of constants must match the number of channels"
        assert len(self.scalers) == X.size(channel_dim), "Number of scalers must match the number of channels"

        X_transformed = []
        for c in range(X.size(channel_dim)):
            if channel_dim == 1:
                X_c_transformed = self.transforms[c](X[:, c])
            else:
                X_c_transformed = self.transforms[c](X[c])
            X_transformed.append(X_c_transformed.unsqueeze(channel_dim))  # Add channel dimension back
        return torch.cat(X_transformed, dim=channel_dim)

    def inverse_transform(self, X):
        if len(X.shape)==5 or len(X.shape)==3:
            # (N X C x Z x H x W)
            # (overpass pixel x C x Z)
            channel_dim = 1
        else:
            # (C x Z x H x W)
            channel_dim = 0

        assert len(self.constants) == X.size(channel_dim), "Number of constants must match the number of channels"
        assert len(self.scalers) == X.size(channel_dim), "Number of scalers must match the number of channels"

        X_inverse_transformed = []
        for c in range(X.size(channel_dim)):
            if channel_dim == 1:
                X_c = self.transforms[c].inverse_transform(X[:, c])
            else:
                X_c = self.transforms[c].inverse_transform(X[c])
            X_inverse_transformed.append(X_c.unsqueeze(channel_dim))  # Add channel dimension back
        return torch.cat(X_inverse_transformed, dim=channel_dim)

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs) 
    
class CustomScaler(BaseEstimator, TransformerMixin):
    """Like StandardScaler, but fit() doesnt need to be executed but mean/std are set via arguments"""
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        return X * self.std + self.mean
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.transform(*args, **kwargs)

class MultiFeatureScaler(BaseEstimator, TransformerMixin):
    """Scale np.array of shape (n_features, n_height_levels)"""
    def __init__(self, means: List, stds: List):
        self.means = means
        self.stds = stds
        self.scalers = []

        if means is not None and stds is not None:
            self._fit_scalers()

    def _fit_scalers(self):
        self.scalers = []
        for i in range(len(self.means)):
            scaler = CustomScaler(mean=self.means[i], std=self.stds[i])
            self.scalers.append(scaler)

    def transform(self, X):
        transformed = []
        for i in range(len(self.scalers)):
            scaler = self.scalers[i]
            transformed.append(scaler.transform(X[i].reshape(-1, 1)))
        return np.array(transformed).squeeze()

    def inverse_transform(self, X):
        original = []
        for i in range(len(self.scalers)):
            scaler = self.scalers[i]
            original.append(scaler.inverse_transform(X[i].reshape(-1, 1)).flatten())
        return np.array(original).squeeze()
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.transform(*args, **kwargs)

