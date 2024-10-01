import torch.nn as nn
import torch
import torch.nn.functional as F

try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity
except ImportError:
    # due to different versions of python, torch
    from torchmetrics.image import ssim as StructuralSimilarityIndexMeasure
    from torchmetrics.image import lpip as LearnedPerceptualImagePatchSimilarity
    from torchmetrics.image import psnr as PeakSignalNoiseRatio

class TwoStageLoss(nn.Module):
    def __init__(self, cloud_free_constant, alpha=0.5, in_cloud_loss=True, weight_along_overpass=False):
        super().__init__()
        self.cloud_free_constant=cloud_free_constant
        self.alpha=alpha
        self.in_cloud_loss=in_cloud_loss # if False calculate loss along whole overpass
        self.weight_along_overpass=weight_along_overpass # if True, pass overpass mask as weight tensor to binary loss
    def forward(self, y_hat_prime, dardar, overpass_mask):
        """inputs are already masked along overpass"""
        
        # predictions 
        y_hat_prime_val = y_hat_prime[:,0,:,:].unsqueeze(1)
        y_hat_prime_bin = y_hat_prime[:,1,:,:].unsqueeze(1)   
            
        # dardar cloud mask
        dardar_binary = (dardar > self.cloud_free_constant).float() * overpass_mask.unsqueeze(1)# convert to binary tensor. cloud:1, cloud free:0   
        
        # in cloud scalars
        dardar_prime_incloud = dardar * dardar_binary
        y_hat_prime_cloud_masked = y_hat_prime_val * dardar_binary
        
        # binary loss
        weight = overpass_mask.unsqueeze(1) if self.weight_along_overpass else None
        loss_binary = F.binary_cross_entropy(torch.sigmoid(y_hat_prime_bin), dardar_binary, weight=weight)
        
        # scalar loss
        if self.in_cloud_loss:
            loss_values = F.l1_loss(y_hat_prime_cloud_masked, dardar_prime_incloud)
        else:
            loss_values = F.l1_loss(y_hat_prime_val, dardar)
        
        # combine losses
        loss = (self.alpha * loss_binary) + (1-self.alpha) * loss_values
        
        
        return {"loss": loss, "loss_binary": loss_binary, "loss_value": loss_values}


class IceRegimeLoss(nn.Module):
    """calculates loss for cirrus and mixed phase regime seperate"""
    def __init__(self, target_transform, cloud_thres=0.000097):
        super().__init__()
        self.target_transform = target_transform
        self.cloud_thres_trafo = target_transform(cloud_thres)
        print("initialized IceRegimeLoss with cloud threshold {} (transformed space)".format(self.cloud_thres_trafo))
    def forward(self, y_hat_prime, dardar, overpass_mask):
        """inputs are already masked along overpass"""
        
        # predictions 
        y_hat_prime_cirrus = y_hat_prime[:,0,:,:].unsqueeze(1)
        y_hat_prime_mixed = y_hat_prime[:,1,:,:].unsqueeze(1)
            
        # individual losses
        cirrus_loss = F.l1_loss(y_hat_prime_cirrus, dardar[:,0,:,:].unsqueeze(1))
        mixed_loss = F.l1_loss(y_hat_prime_mixed, dardar[:,1,:,:].unsqueeze(1))
        
        # get total iwp
        total_iwp = self.target_transform(self.target_transform.inverse_transform(dardar).sum(dim=1)) # first do inverse to calculate sum and then transfrom back
        y_hat_prime_total = self.target_transform(self.target_transform.inverse_transform(y_hat_prime).sum(dim=1)) # first do inverse to calculate sum and then transfrom back
        
        total_loss = F.l1_loss(y_hat_prime_total, total_iwp)
              
        # calculate binary masks and then classification losses
        dardar_binary = (dardar > self.cloud_thres_trafo).float() * overpass_mask.unsqueeze(1)
        y_hat_binary = (y_hat_prime > self.cloud_thres_trafo).float() *   overpass_mask.unsqueeze(1)
        
        binary_loss_cirrus = F.binary_cross_entropy(dardar_binary[:,0],y_hat_binary[:,0])
        binary_loass_mixed = F.binary_cross_entropy(dardar_binary[:,1],y_hat_binary[:,1])
                 
        # NN loss is sum of individual losses and total loss and binary losses  
        loss = cirrus_loss + mixed_loss + total_loss + binary_loss_cirrus + binary_loass_mixed
        
        return {"loss": loss, "cirrus_loss": cirrus_loss, "mixed_loss": mixed_loss, "cirrus_binary": binary_loss_cirrus, "mixed_binary": binary_loass_mixed}
    
class CombinedScalarBinaryLoss(nn.Module):
    """compute loss by combining scalar and binary loss
    
    todo: add adaptive weighting first higher weight to binary and then to scalar
    todo: use binary focal loss instead 
    """
    def __init__(self, target_transform, cloud_thres=0.0, binary_weight=1, focal=False, focal_alpha=0.9):
        super().__init__()
        self.target_transform = target_transform
        self.cloud_thres_trafo = target_transform(cloud_thres)
        self.focal = focal
        self.focal_alpha = 0.9
        self.binary_weight = binary_weight
        print("initialized CombinedScalarBinaryLoss with cloud threshold {} (transformed space)".format(self.cloud_thres_trafo))
        if self.focal:
            print(f"Use focal loss as binary metric with alpha {focal_alpha} and binary loss weight {self.binary_weight}")
        else:
            print("use standard cross-entropy as binary metric")

    def forward(self, y_hat_prime, dardar):
        """inputs are already masked along overpass"""

        # assumes that cloud threshold is >= 0 in transformed space otherwise all non-
        dardar_binary = (dardar > self.cloud_thres_trafo).float()
        y_hat_binary = (y_hat_prime > self.cloud_thres_trafo).float()

        scalar_loss = F.l1_loss(y_hat_prime, dardar, reduction="mean")

        if self.focal:
            binary_loss = focal_loss(y_hat_binary, dardar_binary, alpha=self.focal_alpha, reduction="mean")
        else:
            binary_loss = F.binary_cross_entropy(y_hat_binary,dardar_binary, reduction="mean")

        loss = self.binary_weight * binary_loss + scalar_loss # todo add weighting

        return {"loss": loss, "l1_loss": scalar_loss, "binary_loss": binary_loss}
    
def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example | 0 or 1.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25. Values close to 1 weight the positive class
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples. Only has impact if working with probabilities
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = inputs
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

class SophisticatedReconstructionLoss(nn.Module):
    """
    """
    def __init__(self, pixel_loss_weight=1, ssim_weight=1, lpips_weight=1,ssim_kwargs=dict(data_range=(0,5.1)),lpips_kwargs=dict(net_type="vgg")):
        super().__init__()
        self.pixel_loss_weight=pixel_loss_weight
        self.ssim_weight=ssim_weight
        self.lpips_weight=lpips_weight

        self.ssim = StructuralSimilarityIndexMeasure(**ssim_kwargs).to(torch.device("cuda"))
        self.lpips = LearnedPerceptualImagePatchSimilarity(**lpips_kwargs).to(torch.device("cuda")).eval()

        # TODO: check why last layers are set to trainable
        for param in self.lpips.net.parameters():
            param.requires_grad=False


    def forward(self, y_hat_prime, dardar, overpass_mask):
        """inputs are already masked along overpass"""

        # get profiles
        y_hat_profile_padded, dardar_profile_padded = SophisticatedReconstructionLoss._get_profiles(y_hat_prime, dardar, overpass_mask) # batch_size, max_profile_length, profile_height
        
        # l1 loss
        pixel_wise_loss = F.l1_loss(y_hat_prime, dardar, reduction="mean")
        
        # ssim
        y_hat_profile_padded = y_hat_profile_padded.unsqueeze(1) # unsqueeze 'channel' dim to input in ssim -> batch_size, 1, max_profile_length, profile_height
        dardar_profile_padded = dardar_profile_padded.unsqueeze(1)
        ssim = self.ssim(y_hat_profile_padded, dardar_profile_padded)

        # lpips
        y_hat_profile_padded = y_hat_profile_padded.repeat(1,3,1,1)  # lpips is made for natural images, excpects 3 channels (RGB)
        dardar_profile_padded = dardar_profile_padded.repeat(1,3,1,1)
        lpips = self.lpips(y_hat_profile_padded, dardar_profile_padded)
        
        loss = self.pixel_loss_weight * pixel_wise_loss - self.ssim_weight * ssim + self.lpips_weight * lpips

        return {"loss": loss, "l1_loss": pixel_wise_loss, "ssim": ssim, "lpips": lpips}
    
    @staticmethod
    def _get_profiles(y_hat, dardar, overpass_mask):
        """create profiles with equal length -> pad with zeros to length 96

        Args:
            y_hat (_type_): _description_
            dardar (_type_): _description_
            overpass_mask (_type_): _description_

        Returns:
            _type_: _description_
        """
        batch_size = y_hat.shape[0]
        max_profile_length = 96
        profile_height = y_hat.shape[1]

        
        y_hat_profile_padded =  torch.zeros(batch_size,max_profile_length,profile_height).to(y_hat.device)
        dardar_profile_padded = torch.zeros(batch_size,max_profile_length,profile_height).to(y_hat.device)

        for idx in range(batch_size):
            y_hat_profile = torch.masked_select(y_hat[idx], overpass_mask[idx].bool())
            y_hat_profile = y_hat_profile.reshape(profile_height, int(y_hat_profile.shape[0]/profile_height)).T
            y_hat_profile_padded[idx,:y_hat_profile.shape[0],:] = y_hat_profile

            dardar_profile = torch.masked_select(dardar[idx], overpass_mask[idx].bool())
            dardar_profile = dardar_profile.reshape(profile_height, int(dardar_profile.shape[0]/profile_height)).T
            dardar_profile_padded[idx,:dardar_profile.shape[0],:] = dardar_profile

        return y_hat_profile_padded, dardar_profile_padded

