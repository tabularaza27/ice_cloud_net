
import json
import os
from datetime import datetime,date
import sys
import glob

import comet_ml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch.nn.functional as F
from omegaconf import OmegaConf

from data.data_module import VerticalCloudDataModule, LogTransform, LogTransform2D
from model.LightningModel import VerticalCloudCubeModel
from model.DiscriminatorModel import AutoencoderKL
from model.losses import *
from helpers.callbacks import *
from helpers.comet_helpers import get_patch_ids, load_experiment


# train.py
def main(dm_hparams, model_hparams, gpus, num_nodes, exp_key, comet_logger, model_checkpoints_dir, ckpt_path=None, model_class=VerticalCloudCubeModel):
    print(comet_logger)
    # todo: load meta from previous experiment

    # init data module
    dm = VerticalCloudDataModule(**dm_hparams)
    
    # init model
    model = model_class(**model_hparams)
    model = model.float()

    # init callbacks
    # log meta data to comet
    log_meta_cb = LogExperimentMeta(dm_hparams=dm_hparams,model_hparams=model_hparams)

    # save checkpoint
    checkpoint_dir = os.path.join(model_checkpoints_dir,exp_key)
    checkpoint_cb = ModelCheckpoint(dirpath=checkpoint_dir,every_n_epochs=1,save_top_k=20,monitor="val/rec_loss")

    # save model to comet
    comet_model_cb = SaveModelToComet(log_every_n_epochs=1)

    # log images
    log_image_cb = LogImagesCallback(log_every_n_epochs=1)

    # log learning rate
    lr_cb = LearningRateMonitor(logging_interval='step')

    # control dropout - True during training, otherwise False. Don't pass this if you dont want to use dropout
    control_dropout_cb = ControlDropout()

    # validation callback 
    val_cb = ValidationEpochPlottingCallback()

    print("initiated loggers")

    # init trainer
    if gpus == 1:
        trainer = Trainer(
            accelerator="auto", 
            devices=gpus, 
            max_epochs=100,
            default_root_dir=model_checkpoints_dir, 
            logger=comet_logger,
            accumulate_grad_batches=2,
            callbacks=[log_image_cb,checkpoint_cb, comet_model_cb, log_meta_cb,control_dropout_cb, lr_cb, val_cb])
    else:
        if model_class == AutoencoderKL:
            gpu_strategy = "ddp"
        else:
            gpu_strategy = "ddp_find_unused_parameters_false"

        trainer = Trainer(
            accelerator="auto", 
            devices=gpus, #[2,3] 
            strategy=gpu_strategy,
            max_epochs=145,
            default_root_dir=model_checkpoints_dir, 
            logger=comet_logger,
            accumulate_grad_batches=1,
            sync_batchnorm=True,
            callbacks=[log_image_cb, checkpoint_cb, comet_model_cb, log_meta_cb,control_dropout_cb, lr_cb, val_cb]) #,checkpoint_cb
        

    # accelerator="gpu",
    # strategy="auto",
    # devices=gpus,
    # num_nodes=num_nodes,

    print("initiated trainer")
    
    #if (gpus > 1) and (ckpt_path is not None):
    #    # for distributed loading of ckpt, I got errors → just load weights
    #    model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    #    ckpt_path = None

    trainer.fit(model, dm, ckpt_path=ckpt_path)
    #val_dataloader = dm.val_dataloader()
    #print("len val dataloader", len(val_dataloader))
    #trainer.validate(model, val_dataloader)


if __name__ == '__main__':
    
    #### Adapt the following variables to your setup

    # gpus & nodes
    gpus = 4
    num_nodes = 1

    # init logger (todo: set as environ variables)
    COMET_API_KEY = "your comet api key"
    COMET_WORKSPACE = "comet workspace"
    COMET_PROJECT = "comet project"

    # define data directory and model config
    data_dir = "/path/to/data" # directory with training data 
    model_config = "disc_3d_cond-multipred" # filename of model config; todo edit 
    model_checkpoints_dir = "/path/to/model/checkpoints" # only necessary when loading saved model
    
    # using patch ids from previous experiment or load from disk
    load_ids_exp = None # "from_disk" / "comet_experiment_id" 
    
    # resuming previous experiment
    load_checkpoint_exp = None # comet experiment id 
    checkpoint_epoch = None # epoch to resume training from 

    ### Adapt over
    
    if load_ids_exp:
        # load pids from disk
        if load_ids_exp == "from_disk":
            with open(os.path.join(data_dir,"train_pids.json"), 'r') as file:
                train_patch_ids = json.load(file)
                print(len(train_patch_ids))
                train_patch_ids = list(np.random.choice(train_patch_ids,size=5000,replace=False))
            
            with open(os.path.join(data_dir,"test_pids.json"), 'r') as file: # val_pids
                val_patch_ids = json.load(file)
                print(len(val_patch_ids))
                val_patch_ids = list(np.random.choice(val_patch_ids,size=10239,replace=False))
        # load pids from experiment
        else:
            train_patch_ids, val_patch_ids = get_patch_ids(load_ids_exp)
    else:
        train_patch_ids=None
        val_patch_ids=None

    # dm hparams
    daytime_only = False
    if daytime_only:
        seviri_channels = [f"ch{i}" for i in range(1,11)]
    else:
        seviri_channels = [f"ch{i}" for i in range(4,12)]

    all_channels = True
    if all_channels:
        seviri_channels = [f"ch{i}" for i in range(1,12)]

    # seviri_channels += ["alt", "satzen"]

    fold_to_level_thickness = 4
    target_variable = ["iwc","icnc_5um"]#"iwc"# #"icnc_5um"#"iwc"# 
    dm_hparams = {
                    "data_dir": data_dir,
                    "start_date": date(2007,1,1),
                    "end_date": date(2009,12,31),
                    "height_levels": (16980, 1680), #(18060,1620), # (16980, 1680),#,#(18060,1680),#
                    "daytime_only": daytime_only,
                    "era5_variables": ["temperature"],
                    "target_transform": LogTransform2D(constants=[1,1],scalers=[1e7,1e-2]), #LogTransform(constant=1,scaler=1e7),#LogTransform2D(constants=[1,1],scalers=[1e7,1e-2]),#LogTransform(constant=1,scaler=1e-2), #LogTransform(constant=1,scaler=1e7), # todo try with 1e4 # LogTransform(5e-6),
                    "seviri_channels": seviri_channels,
                    "batch_size":16,
                    "prediction_type":"cube",
                    "target_variable":target_variable,#["cirrus_iwp","mixed_iwp"],#iwc
                    "shuffle":True,
                    "load_online": True,
                    "dataloader_num_workers": 8,
                    "train_patch_ids":train_patch_ids,
                    "val_patch_ids":val_patch_ids,
                    "fold_to_level_thickness": fold_to_level_thickness,
                    "seed": 13,
                    "randomly_mask_ir_channels": 0#0.15
                }

    if model_config:
        model_config_path = f".model_configs/{model_config}.yaml"
        conf = OmegaConf.load(model_config_path)
        model_hparams = conf["model"]["params"]
        model_class = AutoencoderKL
    else:
        model_hparams = {
                            "seviri_channels": seviri_channels,
                            "out_channels": 64,
                            "prediction_heads": 2,
                            "zero_constant": 0,# if dm_hparams["target_transform"] is None else dm_hparams["target_transform"](0),
                            "criterion": F.l1_loss,#torch.nn.HuberLoss(delta=5),#F.l1_loss,#SophisticatedReconstructionLoss(ssim_weight=0), # F.l1_loss # CombinedScalarBinaryLoss(dm_hparams["target_transform"],binary_weight=100,focal=True,focal_alpha=0.9),#F.l1_loss,#IceRegimeLoss(dm_hparams["target_transform"],cloud_thres=0),#CombinedScalarBinaryLoss(dm_hparams["target_transform"],binary_weight=100,focal=True,focal_alpha=0.9),#F.l1_loss,#CombinedScalarBinaryLoss(dm_hparams["target_transform"]),
                            "criterion_kwargs": {},
                            "auto_encoding": False,
                            "binary_prediction": False,
                            "residual": True,
                            "target_variable": dm_hparams["target_variable"],
                            "era5_variables": None, 
                            "cloud_thres":0, 
                            "target_transform":dm_hparams["target_transform"],
                            "lr":5e-5,
                            "lr_scheduler": None,#"cosine_annealing_wr",
                            "dim_mults":(1,2,4),
                            "final_relu": True,
                            "extended_final_conv":False,
                            "rff_params": None, 
                            "additional_logging":False,
                            "final_rff": None,#{"sigma":2**-3, "encoded_size":256}
                            "clamp_output": False,
                            "meta_data_embedding":True
                        }
        model_class = VerticalCloudCubeModel

    # get model checkpoint
    if load_checkpoint_exp:
        experiment = load_experiment(experiment_name=load_checkpoint_exp)
        exp_key = experiment.get_metadata()["experimentKey"] # type: ignore
        print(f"load experiement {load_checkpoint_exp} from checkpoint")
        
        if checkpoint_epoch:
            # load specific epoch
            try:
                ckpt_path = glob.glob(os.path.join(model_checkpoints_dir, exp_key, f"epoch={checkpoint_epoch}*.ckpt"))[0]
            except IndexError:
                raise ValueError(f"no checkpoint available for epoch {checkpoint_epoch}")
        else:
            # load last available checkpoint
            paths = glob.glob(os.path.join(model_checkpoints_dir, exp_key, "*.ckpt"))
            paths.sort(key=os.path.getmtime)
            ckpt_path = paths[-1]
        print(f"loaded {ckpt_path.split('/')[-1]}")
    else: 
        ckpt_path = None

    comet_logger = pl_loggers.CometLogger(
            api_key=COMET_API_KEY,
            workspace=COMET_WORKSPACE,
            project_name=COMET_PROJECT,
            auto_histogram_gradient_logging=True
        )
    
    exp_key = comet_logger.experiment.get_key()
    print("exp key", exp_key)

    # TRAIN
    main(dm_hparams, model_hparams, gpus, num_nodes, model_class=model_class, exp_key=exp_key, comet_logger=comet_logger, model_checkpoints_dir=model_checkpoints_dir, ckpt_path=ckpt_path)