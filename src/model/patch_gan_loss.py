import torch
import functools
import torch.nn as nn
import torch.nn.functional as F

# code is adapted from taming-transformers and stable-diffusion repos
# https://github.com/CompVis/stable-diffusion/
# https://github.com/CompVis/taming-transformers

class DiscriminatorLoss(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge", ndf=64, discriminator_3D=False, crop_to_profiles_2d=False, crop_mode="avg_dimensions",mask_disc_output=False):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        if discriminator_3D == True:
            # use 3D discriminator
            if disc_in_channels > 1:
                # multi target
                
                # concat along channels (sev channels and seviri targets)
                #if disc_conditional: disc_in_channels += 11 # add 11 SEVIRI channels 
                #self.cond_concat_dim=1

                # concat channels with vertical dimension
                self.cond_concat_dim=-3
            else:
                self.cond_concat_dim=-3
            #self.discriminator = NLayerDiscriminator3D(input_nc=disc_in_channels,
            #                                        n_layers=disc_num_layers,
            #                                        ndf=ndf
            #                                        ).apply(weights_init)
            
            # set number of conditional channels (currently only IR channels)
            if disc_conditional:
                cond_channels = 8
            else:
                cond_channels = 0

            self.discriminator = CondNLayerDiscriminator3D(input_nc=disc_in_channels,
                                                           n_layers=disc_num_layers,
                                                           cond_channels=cond_channels,
                                                           ndf=ndf).apply(weights_init)

        else:
            # use 2D discriminator
            self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                    n_layers=disc_num_layers,
                                                    use_actnorm=use_actnorm,
                                                    ndf=ndf
                                                    ).apply(weights_init)
            self.cond_concat_dim=-2
            
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.crop_to_profiles_2d = crop_to_profiles_2d
        self.crop_mode = crop_mode
        self.mask_disc_output = mask_disc_output

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None, overpass_mask=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        #if self.perceptual_weight > 0:
        #    p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
        #    rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        if self.crop_to_profiles_2d:
            reconstructions = DiscriminatorLoss.get_2d_profiles(reconstructions,mode=self.crop_mode, overpass_mask=overpass_mask)
            inputs = DiscriminatorLoss.get_2d_profiles(inputs,mode=self.crop_mode, overpass_mask=overpass_mask)
        
            if self.disc_conditional:
               assert self.disc_conditional
               # get overpass view of seviri
               cond = DiscriminatorLoss.get_2d_profiles(cond,mode=self.crop_mode, overpass_mask=overpass_mask)

        # mask inputs
        expanded_overpass_mask = overpass_mask.to(torch.device("cuda")).expand(reconstructions.shape).bool()
        expanded_overpass_mask = ~expanded_overpass_mask

        reconstructions = reconstructions.masked_fill(expanded_overpass_mask,-9)
        inputs = inputs.masked_fill(expanded_overpass_mask,-9)

        # downsample overpass mask to mask disc output
        if self.mask_disc_output:
            downsampled_overpass_mask = F.interpolate(overpass_mask.float(),size=(1,14,14),mode="area")        
            downsampled_overpass_mask[downsampled_overpass_mask>0] = 1

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                # if len(reconstructions.shape) == 5:
                    # expand seviri cond to N x SevChannels x 64 x 256 x 256 (concat along channels)
                    # cond = cond.unsqueeze(2).expand(-1,-1,reconstructions.shape[2],-1,-1)

                    # expand seviri to N x n_targets x SevChannels x 256 x 256 (concat along vertical dimension)
                    # cond = cond.unsqueeze(1).expand(-1,reconstructions.shape[1],-1,-1,-1)

                # logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=self.cond_concat_dim))
                logits_fake = self.discriminator(reconstructions.contiguous(), cond)
            
            if self.mask_disc_output:
                # logits_fake_masked = logits_fake * downsampled_overpass_mask
                logits_fake_masked = torch.masked_select(logits_fake,downsampled_overpass_mask.bool())
                g_loss = -torch.mean(logits_fake_masked)
            else:
                g_loss = -torch.mean(logits_fake)
            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                # if len(reconstructions.shape) == 5:
                    # expand seviri cond to N x SevChannels x 64 x 256 x 256 (concat along channels)
                    # cond = cond.unsqueeze(2).expand(-1,-1,reconstructions.shape[2],-1,-1)

                    # expand seviri to N x n_targets x SevChannels x 256 x 256 (concat along vertical dimension)
                    # cond = cond.unsqueeze(1).expand(-1,reconstructions.shape[1],-1,-1,-1) 


                #logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=self.cond_concat_dim))
                #logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=self.cond_concat_dim))

                logits_real = self.discriminator(inputs.contiguous().detach(), cond.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach(), cond.contiguous().detach())

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            if self.mask_disc_output:
                #logits_real_masked = logits_real * downsampled_overpass_mask
                #logits_fake_masked = logits_fake * downsampled_overpass_mask

                logits_real_masked = torch.masked_select(logits_real,downsampled_overpass_mask.bool())
                logits_fake_masked = torch.masked_select(logits_fake,downsampled_overpass_mask.bool())

                d_loss = disc_factor * self.disc_loss(logits_real_masked, logits_fake_masked)

                log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                    "{}/logits_real".format(split): logits_real.detach().mean(),
                    "{}/logits_fake".format(split): logits_fake.detach().mean(),
                    "{}/logits_real_masked".format(split): logits_real_masked.detach().mean(),
                    "{}/logits_fake_masked".format(split): logits_fake_masked.detach().mean(),
                    }
            else:
                d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

                log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                    "{}/logits_real".format(split): logits_real.detach().mean(),
                    "{}/logits_fake".format(split): logits_fake.detach().mean()}

                
            return d_loss, log

    @staticmethod
    def get_2d_profiles(cubes:torch.tensor, mode="avg_dimensions", overpass_mask=None):
        assert mode in ["avg_dimensions","padding"]
        
        if mode == "avg_dimensions":
            profiles_2d = torch.stack((cubes.mean(dim=-1), cubes.mean(dim=-2)),dim=1) # N x 2 x 256 x 64

        if mode == "padding":
            assert overpass_mask is not None, f"overpass mask has to be tensor not {overpass_mask}"
            profiles_2d = DiscriminatorLoss._get_padded_profiles(cubes, overpass_mask) # N x 1 x 256 x 96

        return profiles_2d
    
    @staticmethod
    def _get_padded_profiles(cubes, overpass_mask, max_profile_length=96, pad_value=-1):
        """create profiles with equal length -> pad with pad_value to length 96

        Args:
            y_hat (_type_): N x Z x H x W 
            dardar (_type_): N x Z x H x W 
            overpass_mask (_type_): _description_

        Returns:
            torch.tensor, torch.tensor: 2d profiles of y_hat and dardar along over pass with dimensions N x 1 x Z x max_profile_length
        """
        batch_size = cubes.shape[0]
        profile_height = cubes.shape[1]

        profile_padded_list  = []

        for idx in range(batch_size):
            profile = torch.masked_select(cubes[idx], overpass_mask[idx].bool())
            profile = profile.reshape(profile_height, int(profile.shape[0]/profile_height))
            profile_padded = F.pad(profile, (0, max_profile_length-profile.shape[1]),value=pad_value)
            profile_padded = profile_padded.unsqueeze(0).unsqueeze(0)
            profile_padded_list.append(profile_padded)

        profile_padded = torch.concat(profile_padded_list,0)
        

        return profile_padded


# Defines the PatchGAN discriminator with the specified arguments.
# As seen here https://github.com/davidiommi/3D-CycleGan-Pytorch-MedImaging/blob/main/models/networks3D.py#L369
# only modified forward to add channel dimension
class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super(NLayerDiscriminator3D, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # modifiy input to add channel dimension for 1 target prediction
        if len(input.shape)==4:
            input = input.unsqueeze(1)

        return self.model(input)
    

# Defines the PatchGAN discriminator with the specified arguments.
# As seen here https://github.com/davidiommi/3D-CycleGan-Pytorch-MedImaging/blob/main/models/networks3D.py#L369
# only modified forward to add channel dimension
class CondNLayerDiscriminator3D(nn.Module):
    def __init__(self,
                 input_nc,
                 cond_channels=8,
                 ndf=64,
                 n_layers=3,
                 kernel_size=4,
                 padding=1,
                 norm_layer=nn.BatchNorm3d,
                 use_sigmoid=False):
        
        super().__init__()

        # init params
        self.in_channels = input_nc
        self.cond_channels = cond_channels
        self.ndf = ndf
        self.n_layers = n_layers
        self.kernel_size=kernel_size
        self.padding=padding
        self.norm_layer = norm_layer
        self.use_sigmoid = use_sigmoid
        
        # make models
        self.base_model = self._make_base_model()
        self.cond_model = self._make_cond_model() if self.cond_channels else None
        self.bridge_model = self._make_bridge_model() if self.cond_channels else None

    def _make_base_model(self):
        if type(self.norm_layer) == functools.partial:
            use_bias = self.norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = self.norm_layer == nn.InstanceNorm3d
            
        sequence = [nn.Sequential(
            nn.Conv3d(self.in_channels, self.ndf, kernel_size=self.kernel_size, stride=2, padding=self.padding),
            nn.LeakyReLU(0.2, True)
        )]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, self.n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [nn.Sequential(
                nn.Conv3d(self.ndf * nf_mult_prev, self.ndf * nf_mult,
                          kernel_size=self.kernel_size, stride=2, padding=self.padding, bias=use_bias),
                self.norm_layer(self.ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            )]

        nf_mult_prev = nf_mult
        nf_mult = min(2**self.n_layers, 8)
        sequence += [nn.Sequential(
            nn.Conv3d(self.ndf * nf_mult_prev, self.ndf * nf_mult,
                      kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=use_bias),
            self.norm_layer(self.ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        )]

        sequence += [nn.Conv3d(self.ndf * nf_mult, 1, kernel_size=self.kernel_size, stride=1, padding=self.padding)]

        if self.use_sigmoid:
            sequence += [nn.Sigmoid()]

        return nn.ModuleList(sequence)

    def _make_cond_model(self):
        
        if type(self.norm_layer) == functools.partial:
            use_bias = self.norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = self.norm_layer == nn.InstanceNorm3d

        sequence = [nn.Sequential(
            nn.Conv2d(self.cond_channels, self.ndf, kernel_size=self.kernel_size, stride=2, padding=self.padding),
            nn.LeakyReLU(0.2, True)
        )]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, self.n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [nn.Sequential(
                nn.Conv2d(self.ndf * nf_mult_prev, self.ndf * nf_mult,
                          kernel_size=self.kernel_size, stride=2, padding=self.padding, bias=use_bias),
                nn.LeakyReLU(0.2, True)
            )]

        return nn.ModuleList(sequence)

    def _make_bridge_model(self):

        use_bias=False

        sequence = [nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True)
        )]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, self.n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [nn.Sequential(
                nn.Conv2d(self.ndf * nf_mult, self.ndf * nf_mult,
                          kernel_size=1, stride=1, padding=0, bias=use_bias),
                nn.LeakyReLU(0.2, True)
            )]

        return nn.ModuleList(sequence)

    def forward(self, input, cond=None):
        # modifiy input to add channel dimension for 1 target prediction
        if len(input.shape)==4:
            input = input.unsqueeze(1)

        output=input
        cond_output=cond
        for idx, block in enumerate(self.base_model):
            
            output = block(output)
            if idx < len(self.cond_model) and cond_output is not None:
                # pass features
                cond_output = self.cond_model[idx](cond_output)
                # merge features (average strength baceause cond is optional)
                output = (output + self.bridge_model[idx](cond_output)[:,:,None,:,:])/2

        return output


### from taming

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)
    

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h