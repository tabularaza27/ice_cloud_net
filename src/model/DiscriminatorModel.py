"""The code in this file was adapted form the stable-diffusion github repo

@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

source file: https://github.com/CompVis/stable-diffusion/blob/main/ldm/models/autoencoder.py
"""

import importlib
import torch
import numpy as np
from torchvision import transforms
import pytorch_lightning as pl

from data.data_utils import get_overpass_data
from data.data_module import LogTransform2D

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 unetconfig,
                 lossconfig,
                 embed_dim,
                 monitor=None,
                 ckpt_path=None,
                 ):
        super().__init__()
        self.lossconfig=lossconfig
        self.unetconfig=unetconfig
        self.unet = instantiate_from_config(unetconfig)
        if ckpt_path:
            self.load_pretrained_unet(ckpt_path)
        
        self.loss = instantiate_from_config(lossconfig)
        # necessary class attrs for logging via callbacks
        self.out_channels = self.unet.out_channels # needed for plotting callback
        self.meta_data_embedding=self.unet.meta_data_embedding
        self.target_transform = LogTransform2D(constants=[1,1],scalers=[1e7,1e-2]) # todo make dynamic (only used in validation data gathering)
        self.validation_data = []
        
    def load_pretrained_unet(self, ckpt_path):
        print(f"load pretrained unet from path {ckpt_path}")
        pretrained_state_dict = torch.load(ckpt_path)["state_dict"]
        pretrained_state_dict={k.split("ice_net.")[-1]: v for k, v in pretrained_state_dict.items()}
        self.unet.load_state_dict(pretrained_state_dict)

    def forward(self, input, emb=None):
        return self.unet(input, emb)
        
    def get_input(self, batch, split="train"):
        seviri, era5, dardar, overpass_mask, meta_data, patch_idx = batch
        
        # double check when which dtype occurs
        seviri = seviri.float() # this means float32, double() is float64,
        overpass_mask = overpass_mask.long()
        dardar = dardar.float()
        
        # add "channel" dimension to era5, double check era5 normalization
        if len(era5.shape)==2:
            era5 = era5.unsqueeze(1)
        
        # randomly rotate image
        if split == "train":
            rotation_angle = int(np.random.choice([0,90,180,270,-90,-180,-270]))
            seviri = transforms.functional.rotate(seviri, rotation_angle)
            overpass_mask = transforms.functional.rotate(overpass_mask, rotation_angle)
            if len(dardar.shape) == 5:
                # error in rotate function → thus this workaround
                # https://discuss.pytorch.org/t/problem-with-torchvision-functional-rotate/147740
                # for 2 target variables
                shape = dardar.shape
                dardar = transforms.functional.rotate(dardar.view(shape[0], shape[1] * shape[2], shape[3], shape[4]), rotation_angle).view(*shape)
            else:
                dardar = transforms.functional.rotate(dardar, rotation_angle) 

        # error on overpass → mask everythin else
        if len(dardar.shape) == 5:
            overpass_mask= overpass_mask.unsqueeze(1).unsqueeze(1)
        else:
            overpass_mask= overpass_mask.unsqueeze(1)

        return seviri, era5, dardar, overpass_mask, meta_data, patch_idx

    def training_step(self, batch, batch_idx, optimizer_idx):
        seviri, era5, dardar, overpass_mask, meta_data, patch_idx = self.get_input(batch, split="train")
        if self.unet.meta_data_embedding:
            reconstructions = self(seviri, meta_data)
        else:
            reconstructions = self(seviri, None)
        reconstructions = reconstructions * overpass_mask # mask reconstruction to overpass

        if self.loss.disc_conditional:
            if self.lossconfig["params"]["discriminator_3D"]:
                # in case we use whole cube as input for discriminator, passing all seviri data
                cond = seviri[:,3:] # use only IR channels
            else:
                # add seviri along over pass as condition
                cond = seviri * overpass_mask
        else:
            cond = None

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(dardar, reconstructions, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train", overpass_mask=overpass_mask, cond=cond)
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(dardar, reconstructions, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train", overpass_mask=overpass_mask, cond=cond)

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        seviri, era5, dardar, overpass_mask, meta_data, patch_idx = self.get_input(batch, split="val")
        if self.unet.meta_data_embedding:
            reconstructions = self(seviri, meta_data)
        else:
            reconstructions = self(seviri, None)
        reconstructions = reconstructions * overpass_mask # mask reconstruction to overpass

        # add seviri along over pass as condition
        # itodo: in case we use whole cube as input for discriminator, think about passing all seviri data
        if self.loss.disc_conditional:
            if self.lossconfig["params"]["discriminator_3D"]:
                # in case we use whole cube as input for discriminator, passing all seviri data
                cond = seviri[:,3:] # use only IR channels
            else:
                # add seviri along over pass as condition
                cond = seviri * overpass_mask
        else:
            cond = None

        aeloss, log_dict_ae = self.loss(dardar, reconstructions, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val", overpass_mask=overpass_mask, cond=cond)

        discloss, log_dict_disc = self.loss(dardar, reconstructions, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val", overpass_mask=overpass_mask, cond=cond)

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        #return self.log_dict

        y_hat = reconstructions

        # get overpass data (use default log transform)
        y_hat_overpass, dardar_overpass = y_hat_overpass, dardar_overpass = get_overpass_data(y_hat, dardar, overpass_mask, self.target_transform)
        y_hat_overpass_day, dardar_overpass_day = get_overpass_data(y_hat, dardar, overpass_mask, self.target_transform, meta_data=meta_data, meta_data_filter=(3,0))
        y_hat_overpass_night, dardar_overpass_night = get_overpass_data(y_hat, dardar, overpass_mask,  self.target_transform, meta_data=meta_data, meta_data_filter=(3,1))

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

        return {"overpass_mask":overpass_mask, 
                "meta_data": meta_data,
                "y_hat_overpass" : y_hat_overpass,
                "dardar_overpass": dardar_overpass,
                "y_hat_overpass_day" : y_hat_overpass_day,
                "dardar_overpass_day": dardar_overpass_day,
                "y_hat_overpass_night" : y_hat_overpass_night,
                "dardar_overpass_night": dardar_overpass_night,
                } # loss# ,"y_hat":y_hat,"dardar":dardar, "overpass_mask": overpass_mask}
    
    def validation_epoch_end(self, outputs) -> None:
        print("validation epoch end")
        self.validation_data = outputs
        print("set validation data")

    def configure_optimizers(self):
        lr = 5e-5 # TODO: for now manually set by me self.learning_rate
        opt_ae = torch.optim.Adam(self.unet.parameters(),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        if self.unet.prediction_heads > 1:
            return self.unet.unet.final_conv[0][1].weight
        else:
            return self.unet.unet.final_conv[1].weight
