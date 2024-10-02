from model.ConvNextEncoder import ConvNextBlock
from evaluate.eval_plots import *
from helpers.comet_helpers import log_image_to_comet

import matplotlib.pyplot as plt

import numpy as np
import pytorch_lightning as pl
import torch
import einops
import pdb

y_hat_color = "#30A2DA"
dardar_color = "#FD654B"

class LogImagesCallback(pl.Callback):
    def __init__(self, log_every_n_epochs=1):
        self.log_every_n_epochs = log_every_n_epochs
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx): # todo remove dataloader_idx
        current_epoch = trainer.current_epoch
        if (batch_idx == 0) and (current_epoch % self.log_every_n_epochs == 0):
            createLogImages(pl_module, trainer.logger, batch, current_epoch, "val")
            
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        current_epoch = trainer.current_epoch
        if (batch_idx == 0) and (current_epoch % self.log_every_n_epochs == 0):
            createLogImages(pl_module, trainer.logger, batch, current_epoch, "train")

def createLogImages(pl_module, logger, batch, current_epoch, stage):
    seviri, era5, dardar, overpass_mask, meta_data, patch_idx = batch
    #dardar = dardar.permute(0,3,1,2)
    
    with torch.no_grad():
    # model = model.to(cuda0)
    #seviri = seviri.to(torch.device("cuda:0"))
        if pl_module.meta_data_embedding:
            y_hat = pl_module(seviri, meta_data) # todo add image=s, remove posterior
        else:
            y_hat = pl_module(seviri)
        
    seviri=seviri.cpu()
    era5=era5.cpu()
    dardar=dardar.cpu()
    overpass_mask=overpass_mask.cpu()
    y_hat=y_hat.cpu()

    if len(y_hat.shape) == 5:
        n_target_variables = y_hat.shape[1]
    else:
        n_target_variables = 1
    
    batch_size = np.clip(seviri.shape[0],1,32) # maximum 32 images
    
    
    # error on overpass → mask everythin else
    #if len(y_hat.shape) == 5:
    #    overpass_mask= overpass_mask.unsqueeze(1).unsqueeze(1)
    #else:
    #    overpass_mask= overpass_mask.unsqueeze(1)
    #y_hat_prime = y_hat * overpass_mask # masked cube
    #dardar = dardar * overpass_mask # dardar is already masked but in case target is in log space, set all non overpass indices to 0
    # loss = vertical_cloud_model.criterion(y_hat_prime, dardar,**vertical_cloud_model.criterion_kwargs)
    
    fig, ax = plt.subplots(batch_size, 2 + (2*n_target_variables), figsize=(24, 4*batch_size))
    
    for idx in range(batch_size):
        
        if n_target_variables==1:
            y_hat_profile =  np.array([torch.masked_select(y_hat[idx][i,:,:], overpass_mask[idx].bool()).numpy() for i in range(pl_module.out_channels)])
            dardar_profile =  np.array([torch.masked_select(dardar[idx][i,:,:], overpass_mask[idx].bool()).numpy() for i in range(pl_module.out_channels)])
            c = dardar_profile.sum(axis=0)
            integrated_prediction = (y_hat[idx].sum(dim=0).to(torch.device("cpu"))) / 60
        else:
            y_hat_profile= [np.expand_dims(np.array([torch.masked_select(y_hat[idx][var,i], torch.tensor(overpass_mask[idx]).bool()).numpy() for i in range(64)]),0) for var in range(n_target_variables)]
            dardar_profile= [np.expand_dims(np.array([torch.masked_select(dardar[idx][var,i], torch.tensor(overpass_mask[idx]).bool()).numpy() for i in range(64)]),0) for var in range(n_target_variables)]
            c = dardar_profile[0][0].sum(axis=0)
            integrated_prediction = (y_hat[idx].sum(dim=(0,1)).to(torch.device("cpu"))) / 60

        y_coords, x_coords = np.array(np.where(overpass_mask[idx]==1))
        # dardar_overpass_points = np.stack((x_coords, y_coords, c)).T        

        im = ax[idx][0].imshow(integrated_prediction,cmap="Greys_r")#,vmin=0,vmax=7)
        ax[idx][0].scatter(x_coords,y_coords, c=c)
        cbar = fig.colorbar(im, ax=ax[idx][0])
        ax[idx][0].set_title("full image prediction (integrated)")

        im = ax[idx][1].imshow(seviri[idx][4].to(torch.device("cpu")),cmap="Greys")#,vmin=-4,vmax=1)
        ax[idx][1].scatter(x_coords,y_coords, c=c)
        cbar = fig.colorbar(im, ax=ax[idx][1])
        ax[idx][1].set_title("Seviri (single channel)")

        if n_target_variables == 1:
            im = ax[idx][2].imshow(dardar_profile,aspect="auto")#,vmin=0,vmax=4)
            ax[idx][2].set_title("DARDAR profile")
            cbar = fig.colorbar(im, ax=ax[idx][2])

            im = ax[idx][3].imshow(y_hat_profile,aspect="auto")#,vmin=0,vmax=4)
            cbar = fig.colorbar(im, ax=ax[idx][3])
            ax[idx][3].set_title("predicted profile")
        else:
            # create profile plots for each target variable
            for ivar in range(n_target_variables):
                im = ax[idx][2+(ivar*2)].imshow(dardar_profile[ivar].squeeze(),aspect="auto")#,vmin=0,vmax=4)
                ax[idx][2+(ivar*2)].set_title("DARDAR profile")
                cbar = fig.colorbar(im, ax=ax[idx][2+(ivar*2)])

                im = ax[idx][3+(ivar*2)].imshow(y_hat_profile[ivar].squeeze(),aspect="auto")#,vmin=0,vmax=4)
                cbar = fig.colorbar(im, ax=ax[idx][3+(ivar*2)])
                ax[idx][3+(ivar*2)].set_title("predicted profile")

        

    #plt.tight_layout()
    fpath = f"/scratch/{stage}_sample_figs.png" # todo change back to /scratch
    plt.savefig(fpath,dpi="figure")
    logger.experiment.log_image(fpath,name=f"{stage}_sample_preds") # todo 
    plt.close(fig)

class SaveModelToComet(pl.Callback):
    def __init__(self, log_every_n_epochs=10):
        self.log_every_n_epochs = log_every_n_epochs
        
    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch % self.log_every_n_epochs == 0:
            torch.save(trainer.model.state_dict(), '/scratch/ice_net')
            trainer.logger.experiment.log_model('ice_net', '/scratch/ice_net',overwrite=True)

# todo log hparams on training start
class LogExperimentMeta(pl.Callback):
    """Logs metadata of experiment on training start
        - dm hparams
        - model hparams
        - train/val date strings
        - train/val patch ids if load_online
    """
    
    def __init__(self, dm_hparams, model_hparams):
        self.dm_hparams = dm_hparams
        self.model_hparams = model_hparams
        
    def on_fit_start(self, trainer, pl_module):
        # log hparams
        trainer.logger.log_hyperparams(self.dm_hparams)
        print("logged dm hparams")
        trainer.logger.log_hyperparams(self.model_hparams)
        print("logged model hparams")
        
        # log date strings
        trainer.logger.experiment.log_asset_data(trainer.datamodule.data_train.date_strings,name="train_dates",overwrite=True)
        trainer.logger.experiment.log_asset_data(trainer.datamodule.data_val.date_strings,name="val_dates",overwrite=True)
        print("logged date strings")
        
        # log patch ids
        if self.dm_hparams["load_online"]:
            trainer.logger.experiment.log_asset_data(trainer.datamodule.data_train.patch_ids,name="train_patch_ids",overwrite=True)
            trainer.logger.experiment.log_asset_data(trainer.datamodule.data_val.patch_ids,name="val_patch_ids",overwrite=True)
            print("logged patch ids")

class ControlDropout(pl.Callback):
    """sets stochastic depth dropout to True during training and False during validation/test"""

    def __init__(self):
        super()
    
    def on_train_epoch_start(self, trainer, pl_module) -> None:
        _set_dropout(trainer,stage="train")
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        _set_dropout(trainer,stage="val")

        
def _set_dropout(trainer, stage):

    if stage in ["train","fit"]:
        set_value = True
    else:
        set_value = False

    i = 0
    for name, layer in trainer.model.named_modules():
        if isinstance(layer, ConvNextBlock):
            layer.stochastic_depth = set_value
            if i==0:
                print("set stochastic depth to", set_value)
                i+=1

class ValidationEpochPlottingCallback(pl.Callback):
    def __init__(self):
        super()

    def on_validation_epoch_end(self, trainer, pl_module):
        # Gather data from all processes
        # Since pl_module.validation_data is a list of dictionaries, you need to handle it carefully
        #if not trainer.sanity_checking:
        #    pdb.set_trace()
        print("gather outputs")
        outputs = pl_module.all_gather(pl_module.validation_data)
        print("gathered outputs")
        #print("callback on device", outputs[0][0]["overpass_mask"].device)
        
        if trainer.is_global_zero and not trainer.sanity_checking:  # If it's the main process    
            print(type(outputs))
            print(len(outputs))
            print(type(outputs[0]))
            #print(outputs[0])
            print(len(outputs[0]))
            meta_data = torch.vstack([einops.rearrange(output["meta_data"].detach(), "d b c -> (d b) c") for output in outputs])
            overpass_mask = torch.vstack([einops.rearrange(output["overpass_mask"].detach(), "d b c z y x -> (d b) c z y x") for output in outputs])
            y_hat_overpass = torch.vstack([einops.rearrange(output["y_hat_overpass"].detach(), "d b c z -> (d b) c z") for output in outputs])
            y_hat_overpass_day = torch.vstack([einops.rearrange(output["y_hat_overpass_day"].detach(), "d b c z -> (d b) c z") for output in outputs])
            y_hat_overpass_night = torch.vstack([einops.rearrange(output["y_hat_overpass_night"].detach(), "d b c z -> (d b) c z") for output in outputs])
            dardar_overpass = torch.vstack([einops.rearrange(output["dardar_overpass"].detach(), "d b c z -> (d b) c z") for output in outputs])
            dardar_overpass_day = torch.vstack([einops.rearrange(output["dardar_overpass_day"].detach(), "d b c z -> (d b) c z") for output in outputs])
            dardar_overpass_night = torch.vstack([einops.rearrange(output["dardar_overpass_night"].detach(), "d b c z -> (d b) c z") for output in outputs])
            
            print("overpass mask shape", overpass_mask.shape)
            print("Shape of y_hat_overpass:", y_hat_overpass.shape)
            print("Shape of y_hat_overpass_day:", y_hat_overpass_day.shape)
            print("Shape of y_hat_overpass_night:", y_hat_overpass_night.shape)
            print("Shape of dardar_overpass:", dardar_overpass.shape)
            print("Shape of dardar_overpass_day:", dardar_overpass_day.shape)
            print("Shape of dardar_overpass_night:", dardar_overpass_night.shape)

    
            print("device of y_hat_overpass:", y_hat_overpass.device)
            print("device of y_hat_overpass_day:", y_hat_overpass_day.device)
            print("device of y_hat_overpass_night:", y_hat_overpass_night.device)
            print("device of dardar_overpass:", dardar_overpass.device)
            print("device of dardar_overpass_day:", dardar_overpass_day.device)
            print("device of dardar_overpass_night:", dardar_overpass_night.device)

            target_idxs = [0,1]
            targets = ["iwc","nice"]
            log_trans = [LogTransform(scaler=1e7), LogTransform(scaler=1e-2)]
            cloud_thres = 0

            if trainer.datamodule.fold_to_level_thickness:
                level_thickness = 60*trainer.datamodule.fold_to_level_thickness
            else:
                level_thickness = 60
            height_levels = get_height_level_range(trainer.datamodule.height_levels[1],trainer.datamodule.height_levels[0],level_thickness)
            height_levels = height_levels[-pl_module.out_channels:] # height levels has to have the same shape as predictions / cut off highest hight if necessary
            
            # save tensors
            torch.save(dict(meta_data=meta_data.cpu().detach(),
                            overpass_mask=overpass_mask.cpu().detach(),
                            y_hat=y_hat_overpass.cpu().detach(),
                            y_hat_day=y_hat_overpass_day.cpu().detach(),
                            y_hat_night=y_hat_overpass_night.cpu().detach(),
                            dardar=dardar_overpass.cpu().detach(),
                            dardar_day=dardar_overpass_day.cpu().detach(),
                            dardar_night=dardar_overpass_night.cpu().detach()),f"/cluster/work/climate/kjeggle/val_tensors/epoch={trainer.current_epoch}.pt")
            print("saved val tensors")
            for i,t,l in zip(target_idxs,targets,log_trans):
                print(i,t,l)
            
                plot_and_log(y_hat_overpass[:,i,:],dardar_overpass[:,i,:],trainer=trainer,suffix=f"{t}",target_variable=t,target_transform=l,height_levels=height_levels,cloud_thres=cloud_thres)
                plot_and_log(y_hat_overpass_day[:,i,:],dardar_overpass_day[:,i,:],trainer=trainer,suffix=f"{t}_day",target_variable=t,target_transform=l,height_levels=height_levels,cloud_thres=cloud_thres)
                plot_and_log(y_hat_overpass_night[:,i,:],dardar_overpass_night[:,i,:],trainer=trainer,suffix=f"{t}_night",target_variable=t,target_transform=l,height_levels=height_levels,cloud_thres=cloud_thres)    
            
           
def plot_and_log(y_hat, dardar, trainer, suffix="",target_variable="iwc",target_transform=LogTransform(scaler=1e7),height_levels=get_height_level_range(),cloud_thres=0):


    exp = trainer.logger.experiment

    if suffix != "":
        suffix = f"_{suffix}"

    if target_variable=="icnc_5um":
        target_variable="nice"
    
    level_thickness = np.abs(height_levels[0]-height_levels[1])
    

    # confusion matrix
    #fig = plt_confusion_matrix(y_hat, dardar, cloud_thres=cloud_thres)
    #log_image_to_comet(exp,fig, f"confusion_matrix_{cloud_thres}{suffix}")
    
    # cloud occurance per height level
    df, occurance_p = cloud_occurance_per_height_level(y_hat, dardar, height_levels=height_levels)
    log_image_to_comet(exp,occurance_p, f"cloud_occurance{suffix}")
    
    # metrics per level
    df, metric_p = metrics_per_level(y_hat, dardar, cloud_thres=cloud_thres, height_levels=height_levels, target_transform=target_transform,n_level_aggregation=int(height_levels.shape[0]/16)) # todo better heuristic for n_level_aggregation 
    log_image_to_comet(exp,metric_p,f"performance_metrics_levels{suffix}")
    
    result_fig1 = hv.Layout((metric_p.opts(backend_opts={"legend.title": ""}),occurance_p)).opts(backend_opts={"legend.title": ""},hspace=0.5,fig_inches=7)
    log_image_to_comet(exp,result_fig1,f"results_fig1{suffix}")

    # iwc per height level
    y_hat_height_iwc_df, y_hat_height_iwc_plt = iwc_per_height_df(y_hat,color=y_hat_color,plt_q10=False, height_levels=height_levels,target_variable=target_variable)
    dardar_height_iwc_df, dardar_height_iwc_plt = iwc_per_height_df(dardar,color=dardar_color,plt_q10=False, height_levels=height_levels,target_variable=target_variable)
    p = (y_hat_height_iwc_plt * dardar_height_iwc_plt).opts(fontscale=1.5)
    log_image_to_comet(exp,p, f"iwc_height{suffix}")
    # distributions
    fig = iwc_in_cloud_distribution(y_hat, dardar, cloud_thres=cloud_thres,target_variable=target_variable)
    log_image_to_comet(exp,fig, f"iwc_distributions{suffix}")

    # iwc vs iwc
    df, g = iwc_vs_iwc_plt(y_hat, dardar,target_variable=target_variable)
    log_image_to_comet(exp,g, f"iwc_vs_iwc{suffix}")
        


            



