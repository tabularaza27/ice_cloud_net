from torch import nn
import torch
import numpy as np


# lat: range([-30,30])
# lon: range([-30,30])
# month: [0,12]
# time-of-day: [0,1]
# land_water_mask: [0,1,2,3,4,5,6]

class MetadataEmbedding(nn.Module):

  def __init__(self,
               emb_dim = [32,32,16,2,2], # length of embedding - can be a list of length for each variable
               lims=[
                   [-30, 30], # latitude
                   [-30, 30], # longitude
                   [0, 12], # month scaling 
                   [0, 1], # day night scaling?
                   [0, 6] # land-water-mask
               ],
               meta_data_indices = [0,1,2,3,5] # latitude, longitude, month, day_night, land_water_mask (as defined in `get_patch_meta_data()`)
               ):
    super().__init__()

    self.meta_data_indices = meta_data_indices

    # precompute the embedding dimension
    self.emb_dim = emb_dim if isinstance(emb_dim, list) or isinstance(emb_dim, tuple) else len(lims)*[emb_dim]
    self.lims = lims
    assert len(self.emb_dim) == len(self.lims)

    # compose MLP
    self.total_length = sum(self.emb_dim)
    self.mlp = nn.Sequential(
        nn.Linear(2*self.total_length, self.total_length * 4),
                nn.GELU(),
                nn.Linear(self.total_length * 4, self.total_length)
            )

    # produce mapping
    self.mappings = []
    for idx in range(len(self.lims)):
      emb_len = self.emb_dim[idx]
      mapping_element = torch.arange(emb_len)/(max(-min(self.lims[idx]), max(self.lims[idx])))

      self.mappings.append(mapping_element)

  def forward(self, input):
    '''
    x is a LIST of inputs
    '''
      
    assert isinstance(input, list) or isinstance(input, tuple) or isinstance(input, torch.Tensor)

    # input the whole meta data tensor
    if isinstance(input, torch.Tensor):
        input = input[:,self.meta_data_indices] # batch_size, n_meta_data_variables
        input=[input[:,i].unsqueeze(1).float() for i in range(input.shape[-1])] # create list with each element being (batch_size,1) tensor of one meta data var

    assert len(input) == len(self.mappings)

    phase = []
    for idx, x in enumerate(input):
        phase.append(self.mappings[idx].to(x.device) * x)
    phase = 2 * torch.pi * torch.cat(phase, dim=-1)

    return self.mlp(torch.cat([torch.sin(phase), torch.cos(phase)], axis = -1))