import torch
from torch import nn
from .ConvNextEncoder import *

class IWPNetV1(nn.Module):
    
    def __init__(self,
                 in_channels = 11,
                 out_channels = 1, 
                 prediction_heads = 1,
                 dim_mults = (1,2,4,8),
                 residual = False,
                 final_relu = True,
                 extended_final_conv = False,
                 final_rff=None,
                 clamp_output=None,
                 meta_data_embedding=False
                ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.prediction_heads = prediction_heads
        self.dim_mults = dim_mults
        self.residual = residual
        self.final_relu = final_relu # adds relu behind last layer → is default now
        self.extended_final_conv = extended_final_conv # adds a few convs with kernel size 1 before the output ( basically an MLP )
        self.final_rff = final_rff # if True add RFF before final Conv layer
        self.clamp_output = clamp_output
        self.meta_data_embedding = meta_data_embedding


        # Image Encoder
        self.unet=UnetConvNextBlock2(64,
                                    in_channels = self.in_channels,
                                    out_channels = self.out_channels,
                                    prediction_heads=self.prediction_heads,
                                    dim_mults=self.dim_mults,
                                    residual=self.residual,
                                    final_relu=self.final_relu,
                                    extended_final_conv=self.extended_final_conv,
                                    final_rff=self.final_rff,
                                    meta_data_embedding=self.meta_data_embedding)
               
    def forward(self, image, meta_data=None):
        output = self.unet(image, meta_data)

        # clamp non-zero outputs between  min,max values
        if isinstance(self.clamp_output,tuple):
            mask = output>self.clamp_output[0]
            output= output * mask # set output to 0 for values smaller than min value
            # output = torch.where(output != 0, torch.clamp(output, *self.clamp_output), output)

        return output   
