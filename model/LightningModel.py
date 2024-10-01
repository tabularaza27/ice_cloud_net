import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from model.IWPNetV1 import IWPNetV1
from model.losses import TwoStageLoss, IceRegimeLoss, CombinedScalarBinaryLoss, SophisticatedReconstructionLoss

from data.data_module import LogTransform
from data.data_utils import get_overpass_data

from torchvision.ops.focal_loss import sigmoid_focal_loss

from torchvision import transforms

from torchmetrics import Accuracy, F1Score, MeanAbsoluteError, R2Score

class VerticalCloudCubeModel(pl.LightningModule):
    def __init__(self, 
                 seviri_channels, 
                 era5_variables,
                 out_channels=1,
                 prediction_heads=1,
                 target_variable="iwp",
                 binary_prediction=False,
                 criterion=F.l1_loss,
                 criterion_kwargs={},
                 zero_constant=None,
                 auto_encoding=False,
                 residual=False,
                 dim_mults=(1,2,4,8),
                 cloud_thres=0.000097, # 0.005 percentile in original scale,
                 target_transform=LogTransform(5e-6), # type: ignore
                 lr=1e-6,
                 lr_scheduler=None,
                 final_relu=True,
                 extended_final_conv=True,
                 rff_params=None,
                 additional_logging=False,
                 final_rff=None,
                 clamp_output=False,
                 meta_data_embedding=False
                ):
        super().__init__()
        self.seviri_channels = seviri_channels
        self.era5_variables = era5_variables
        self.out_channels = out_channels
        self.prediction_heads = prediction_heads
        self.criterion = criterion
        self.criterion_kwargs = criterion_kwargs
        self.binary_prediction = binary_prediction
        self.zero_constant = zero_constant
        self.auto_encoding = auto_encoding
        self.residual = residual
        self.dim_mults = dim_mults
        self.cloud_thres = cloud_thres
        self.target_transform = target_transform
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.final_relu = final_relu
        self.extended_final_conv = extended_final_conv
        self.rff_params = rff_params
        self.additional_logging = additional_logging
        self.final_rff=final_rff # final rff parameters or None
        self.meta_data_embedding = meta_data_embedding
        
        if clamp_output:
            self.clamp_output = (self.target_transform(cloud_thres), self.target_transform(1e-2))
        else:
            self.clamp_output = None
        
        if target_variable in ["iwc","icnc_5um","icnc_100um","icnc_5um_incloud","icnc_100um_incloud"]:
            self.ice_net = IWPNetV1(in_channels=len(self.seviri_channels),out_channels=self.out_channels,dim_mults = self.dim_mults,extended_final_conv=self.extended_final_conv,residual = self.residual,final_relu = self.final_relu, final_rff=self.final_rff,clamp_output=self.clamp_output, meta_data_embedding=self.meta_data_embedding)#, )
                
        if self.additional_logging:
            # classification metrics    
            self.train_acc = Accuracy(num_classes=2,multiclass=True)
            self.val_acc = Accuracy(num_classes=2,multiclass=True)
            self.train_f1 = F1Score(num_classes=2,multiclass=True)
            self.val_f1 = F1Score(num_classes=2,multiclass=True)
            # regression metrics
            self.train_mae_original = MeanAbsoluteError()
            self.val_mae_original = MeanAbsoluteError()
            self.train_mae_in_cloud = MeanAbsoluteError()
            self.val_mae_in_cloud = MeanAbsoluteError()
            self.train_mae_overpass = MeanAbsoluteError()
            self.val_mae_overpass = MeanAbsoluteError()
            self.train_mae_capped = MeanAbsoluteError()
            self.val_mae_capped = MeanAbsoluteError()
            self.train_mae_capped_original = MeanAbsoluteError()
            self.val_mae_capped_original = MeanAbsoluteError()
            self.train_r2 = R2Score()
            self.val_r2 = R2Score()
            self.train_r2_original = R2Score()
            self.val_r2_original = R2Score()

        self.validation_data = []
        
    def forward(self, image, era5=None):
        # todo rename era5 to something agnostic or meta_embed
        if era5 is not None:
            return self.ice_net(image, era5)
        else:
            return self.ice_net(image)#, None)
    
    def training_step(self, batch, batch_idx):
        stage="train"
        seviri, era5, dardar, overpass_mask, meta_data, patch_idx = batch
        
        # double check when which dtype occurs
        seviri = seviri.float() # this means float32, double() is float64,
        
        overpass_mask = overpass_mask.long()
        # dardar = dardar.permute(0,3,1,2) # permute to batch, channels, H, W
        dardar = dardar.float()

        # only use high clouds - todo remove
        if self.out_channels == 50:
            dardar=dardar[:,:50,:,:]
        
        # add "channel" dimension to era5, double check era5 normalization
        if len(era5.shape)==2:
            era5 = era5.unsqueeze(1)
        
        # randomly rotate image
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
        
        if self.era5_variables is None:
            era5 = None
    
        if self.binary_prediction:
            dardar = (dardar > self.zero_constant).float() # convert to binary tensor. cloud:1, cloud free:0
        
        # dardar_binary = (dardar > self.zero_constant).long() # convert to binary tensor. cloud:1, cloud free:0
        
        # todo change arg name of era5
        if self.meta_data_embedding:
            y_hat = self(image=seviri, era5=meta_data)
        else:
            y_hat = self(image=seviri, era5=era5) # batch, channels, x, y
        
        # error on overpass → mask everythin else
        if len(y_hat.shape) == 5:
            overpass_mask= overpass_mask.unsqueeze(1).unsqueeze(1)
        else:
            overpass_mask= overpass_mask.unsqueeze(1)

        y_hat_prime = y_hat * overpass_mask # masked cube
        dardar = dardar * overpass_mask # dardar is already masked but in case target is in log space, set all non overpass indices to 0
        # dardar_binary = dardar_binary
        
        # two stage loss
        if isinstance(self.criterion, TwoStageLoss):
            loss_dict = self.criterion(y_hat_prime, dardar, overpass_mask)
            loss = loss_dict["loss"]
            loss_binary = loss_dict["loss_binary"]
            loss_value = loss_dict["loss_value"]
            self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{stage}_loss_binary", loss_binary, on_step=True, on_epoch=True, logger=True)
            self.log(f"{stage}_loss_value", loss_value, on_step=True, on_epoch=True, logger=True)
        elif isinstance(self.criterion, IceRegimeLoss):
            loss_dict = self.criterion(y_hat_prime, dardar, overpass_mask)
            loss = loss_dict["loss"]
            loss_cirrus = loss_dict["cirrus_loss"]
            loss_mixed = loss_dict["mixed_loss"]
            loss_cirrus_binary = loss_dict["cirrus_binary"]
            loss_mixed_binary = loss_dict["mixed_binary"]
            self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{stage}_cirrus_loss", loss_cirrus, on_step=True, on_epoch=True, logger=True)
            self.log(f"{stage}_mixed_loss", loss_mixed, on_step=True, on_epoch=True, logger=True)
            self.log(f"{stage}_cirrus_binary", loss_cirrus_binary, on_step=True, on_epoch=True, logger=True)
            self.log(f"{stage}_mixed_binary", loss_mixed_binary, on_step=True, on_epoch=True, logger=True)
        elif self.criterion == F.binary_cross_entropy_with_logits:
            # create clone of overpass mask and set zeros to 999 and ones to 0 
            om = torch.clone(overpass_mask)
            om[om==0]=-999
            om[om==1]=0
            om=torch.repeat_interleave(om.unsqueeze(1),256,1)
            
            # add om on y_hat → important as loss is calculated on logits
            y_hat_prime = y_hat + om
            
            # weight = torch.clone(overpass_mask)
            # weight = torch.repeat_interleave(weight.unsqueeze(1),256,1)
            # weight *= 100
            loss = sigmoid_focal_loss(y_hat_prime, dardar, reduction="mean")
            #loss = F.binary_cross_entropy_with_logits(y_hat_prime,dardar,weight=weight,reduction="mean")
            self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        elif isinstance(self.criterion, CombinedScalarBinaryLoss):
            loss_dict = self.criterion(y_hat_prime, dardar)
            loss = loss_dict["loss"]
            loss_l1 = loss_dict["l1_loss"]
            loss_binary = loss_dict["binary_loss"]

            self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{stage}_l1_loss", loss_l1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{stage}_binary_loss", loss_binary, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            # calculate loss on original scale (only for logging)
            loss_dict_original = self.criterion(self.target_transform.inverse_transform(y_hat_prime), self.target_transform.inverse_transform(dardar),**self.criterion_kwargs)
            loss_original = loss_dict_original["loss"]
            loss_l1_original = loss_dict_original["l1_loss"]
            loss_binary_original = loss_dict_original["binary_loss"]

            self.log(f"{stage}_loss_original", loss_original, on_step=True, on_epoch=True, logger=True)
            self.log(f"{stage}_l1_loss_original", loss_l1_original, on_step=True, on_epoch=True, logger=True)
            self.log(f"{stage}_binary_loss_original", loss_binary_original, on_step=True, on_epoch=True, logger=True)
        elif isinstance(self.criterion, SophisticatedReconstructionLoss):
            loss_dict = self.criterion(y_hat_prime, dardar, overpass_mask)
            loss = loss_dict["loss"]
            loss_l1 = loss_dict["l1_loss"]
            ssim = loss_dict["ssim"]
            lpips = loss_dict["lpips"]
            
            self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{stage}_l1_loss", loss_l1, on_step=True, on_epoch=True, logger=True)
            self.log(f"{stage}_ssim", ssim, on_step=True, on_epoch=True, logger=True)
            self.log(f"{stage}_lpips", lpips, on_step=True, on_epoch=True, logger=True)
        else:
            # normal loss
            loss = self.criterion(y_hat_prime, dardar,**self.criterion_kwargs)
            loss_original = self.criterion(self.target_transform.inverse_transform(y_hat_prime), self.target_transform.inverse_transform(dardar),**self.criterion_kwargs)
            self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{stage}_loss_original", loss_original, on_epoch=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        stage="val"

        seviri, era5, dardar, overpass_mask, meta_data, patch_idx = batch
        seviri = seviri.float()
        overpass_mask = overpass_mask.long()
        # dardar = dardar.permute(0,3,1,2)
        dardar = dardar.float()

        # only use high clouds - todo remove
        if self.out_channels == 50:
            dardar=dardar[:,:50,:,:]
        
        if len(era5.shape)==2:
            era5 = era5.unsqueeze(1)
        
        if self.era5_variables is None:
            era5 = None
        
        if self.binary_prediction:
            dardar = (dardar > self.zero_constant).float() # convert to binary tensor. cloud:1, cloud free:0
        
        # dardar_binary = (dardar > self.zero_constant).long() # convert to binary tensor. cloud:1, cloud free:0
        # todo change arg name of era5
        if self.meta_data_embedding:
            y_hat = self(image=seviri, era5=meta_data)
        else:
            y_hat = self(image=seviri, era5=era5) # batch, channels, x, y
        
        # error on overpass → mask everythin else
        if len(y_hat.shape) == 5:
            overpass_mask= overpass_mask.unsqueeze(1).unsqueeze(1)
        else:
            overpass_mask= overpass_mask.unsqueeze(1)

        y_hat_prime = y_hat * overpass_mask # masked cube
        dardar = dardar * overpass_mask # dardar is already masked but in case target is in log space, set all non overpass indices to 0
        # dardar_binary = dardar_binary
        
        # two stage loss
        if isinstance(self.criterion, TwoStageLoss):
            loss_dict = self.criterion(y_hat_prime, dardar, overpass_mask)
            loss = loss_dict["loss"]
            loss_binary = loss_dict["loss_binary"]
            loss_value = loss_dict["loss_value"]
            self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{stage}_loss_binary", loss_binary, on_step=True, on_epoch=True, logger=True)
            self.log(f"{stage}_loss_value", loss_value, on_step=True, on_epoch=True, logger=True)
        elif isinstance(self.criterion, IceRegimeLoss):
            loss_dict = self.criterion(y_hat_prime, dardar, overpass_mask)
            loss = loss_dict["loss"]
            loss_cirrus = loss_dict["cirrus_loss"]
            loss_mixed = loss_dict["mixed_loss"]
            loss_cirrus_binary = loss_dict["cirrus_binary"]
            loss_mixed_binary = loss_dict["mixed_binary"]
            self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{stage}_cirrus_loss", loss_cirrus, on_step=True, on_epoch=True, logger=True)
            self.log(f"{stage}_mixed_loss", loss_mixed, on_step=True, on_epoch=True, logger=True)
            self.log(f"{stage}_cirrus_binary", loss_cirrus_binary, on_step=True, on_epoch=True, logger=True)
            self.log(f"{stage}_mixed_binary", loss_mixed_binary, on_step=True, on_epoch=True, logger=True)
        elif self.criterion == F.binary_cross_entropy_with_logits:
            # create clone of overpass mask and set zeros to 999 and ones to 0 
            om = torch.clone(overpass_mask)
            om[om==0]=-999
            om[om==1]=0
            om=torch.repeat_interleave(om.unsqueeze(1),256,1)
            
            # add om on y_hat → important as loss is calculated on logits
            y_hat_prime = y_hat + om
            
            # weight = torch.clone(overpass_mask)
            # weight = torch.repeat_interleave(weight.unsqueeze(1),256,1)
            # weight *= 100
            loss = sigmoid_focal_loss(y_hat_prime, dardar, reduction="mean")
            # loss = F.binary_cross_entropy_with_logits(y_hat_prime,dardar,weight=weight,reduction="mean")
            self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)        
        elif isinstance(self.criterion, CombinedScalarBinaryLoss):
            loss_dict = self.criterion(y_hat_prime, dardar)
            loss = loss_dict["loss"]
            loss_l1 = loss_dict["l1_loss"]
            loss_binary = loss_dict["binary_loss"]

            self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{stage}_l1_loss", loss_l1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{stage}_binary_loss", loss_binary, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            # calculate loss on original scale (only for logging)
            loss_dict_original = self.criterion(self.target_transform.inverse_transform(y_hat_prime), self.target_transform.inverse_transform(dardar),**self.criterion_kwargs)
            loss_original = loss_dict_original["loss"]
            loss_l1_original = loss_dict_original["l1_loss"]
            loss_binary_original = loss_dict_original["binary_loss"]

            self.log(f"{stage}_loss_original", loss_original, on_step=True, on_epoch=True, logger=True)
            self.log(f"{stage}_l1_loss_original", loss_l1_original, on_step=True, on_epoch=True, logger=True)
            self.log(f"{stage}_binary_loss_original", loss_binary_original, on_step=True, on_epoch=True, logger=True)  
        elif isinstance(self.criterion, SophisticatedReconstructionLoss):
            loss_dict = self.criterion(y_hat_prime, dardar, overpass_mask)
            loss = loss_dict["loss"]
            loss_l1 = loss_dict["l1_loss"]
            ssim = loss_dict["ssim"]
            lpips = loss_dict["lpips"]
            
            self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{stage}_l1_loss", loss_l1, on_step=True, on_epoch=True, logger=True)
            self.log(f"{stage}_ssim", ssim, on_step=True, on_epoch=True, logger=True)
            self.log(f"{stage}_lpips", lpips, on_step=True, on_epoch=True, logger=True)
        else:
            # normal loss
            loss = self.criterion(y_hat_prime, dardar,**self.criterion_kwargs)
            loss_original = self.criterion(self.target_transform.inverse_transform(y_hat_prime), self.target_transform.inverse_transform(dardar),**self.criterion_kwargs)
            self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
            self.log(f"{stage}_loss_original", loss_original, on_epoch=True, logger=True,sync_dist=True)
        
        y_hat_overpass, dardar_overpass = y_hat_overpass, dardar_overpass = get_overpass_data(y_hat, dardar, overpass_mask, self.target_transform)
        y_hat_overpass_day, dardar_overpass_day = get_overpass_data(y_hat, dardar, overpass_mask, self.target_transform, meta_data, meta_data_filter=(3,0))
        y_hat_overpass_night, dardar_overpass_night = get_overpass_data(y_hat, dardar, overpass_mask, self.target_transform, meta_data, meta_data_filter=(3,1))

        return {"loss": loss, 
                "overpass_mask":overpass_mask, 
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
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.lr_scheduler:
            # select lr_scheduler
            if self.lr_scheduler == "cosine_annealing":
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=20)
            elif self.lr_scheduler == "cosine_annealing_wr":
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optim, T_0=5)
            elif self.lr_scheduler == "exponential":
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.99)
            else:
                raise ValueError("lr_scheduler {} not implemented".format(self.lr_scheduler))
            return {"optimizer": optim, "lr_scheduler": lr_scheduler}
        else:
            # no scheduler
            return optim