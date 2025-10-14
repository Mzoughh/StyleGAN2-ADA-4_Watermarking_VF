# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


# MODIFIED VERSION FOR WATERMARKING METHODS

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

# -------------W---------------- #
### Additional packages for trigger-set watermarking methods
device = torch.device('cuda')
from torchvision import transforms
from hidden.models import HiddenEncoder, HiddenDecoder, EncoderWithJND, EncoderDecoder
from hidden.attenuations import JND
import torch.nn.functional as F
import os
from torchvision.utils import save_image



################ HiDDeN ###################
NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
UNNORMALIZE_IMAGENET = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
default_transform = transforms.Compose([transforms.ToTensor(), NORMALIZE_IMAGENET])

class Params():
    def __init__(self, encoder_depth:int, encoder_channels:int, decoder_depth:int, decoder_channels:int, num_bits:int,
                attenuation:str, scale_channels:bool, scaling_i:float, scaling_w:float):
        # encoder and decoder parameters
        self.encoder_depth = encoder_depth
        self.encoder_channels = encoder_channels
        self.decoder_depth = decoder_depth
        self.decoder_channels = decoder_channels
        self.num_bits = num_bits
        # attenuation parameters
        self.attenuation = attenuation
        self.scale_channels = scale_channels
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w

params = Params(
    encoder_depth=4, encoder_channels=64, decoder_depth=8, decoder_channels=64, num_bits=48,
    attenuation="jnd", scale_channels=False, scaling_i=1, scaling_w=1.5
)

decoder = HiddenDecoder(
    num_blocks=params.decoder_depth, 
    num_bits=params.num_bits, 
    channels=params.decoder_channels
)
encoder = HiddenEncoder(
    num_blocks=params.encoder_depth, 
    num_bits=params.num_bits, 
    channels=params.encoder_channels
)
attenuation = JND(preprocess=UNNORMALIZE_IMAGENET) if params.attenuation == "jnd" else None
encoder_with_jnd = EncoderWithJND(
    encoder, attenuation, params.scale_channels, params.scaling_i, params.scaling_w
)

# Load pretrained weights whitened
ckpt_path_whitened = "/home/mzoughebi/personal_study/Original_repository_of_3_methods/stable_signature/hidden/ckpts/hidden_replicate.pth"

# msg_decoder = torch.jit.load(ckpt_path).to(device)
state_dict = torch.load(ckpt_path_whitened, map_location='cpu')['encoder_decoder']
encoder_decoder_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
decoder_state_dict = {k.replace('decoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'decoder' in k}
decoder.load_state_dict(decoder_state_dict)
msg_decoder = decoder # à regarder
msg_decoder = msg_decoder.to(device).eval()
nbit = msg_decoder(torch.zeros(1, 3, 128, 128).to(device)).shape[-1]
print('>>>>> Hidden LOADED')

# Freeze LDM and hidden decoder
for param in [*msg_decoder.parameters()]:
    param.requires_grad = False

# loss 
loss_trigger = 'bce'  # 'mse' or 'bce'
if loss_trigger == 'mse':
    loss_trigger = lambda decoded, keys, temp=10.0: torch.mean((decoded*temp - (2*keys-1))**2) # b k - b k
elif loss_trigger == 'bce':
    loss_trigger = lambda decoded, keys, temp=10.0: F.binary_cross_entropy_with_logits(decoded*temp, keys, reduction='mean')

# Creating key
print(f'\n>>> Creating key with {nbit} bits...')
key = torch.randint(0, 2, (1, nbit), dtype=torch.float32, device=device)
key_str = "".join([ str(int(ii)) for ii in key.tolist()[0]])
print(f'Key: {key_str}')
keys = key.repeat(32, 1)  # repeat for the batch size
#-------------------------------- #

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D,  G=None, tools=None, watermarking_dict=None, watermark_weight=None, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        #------------------ W-------------#
        self.style_mixing_prob = 0 # default= style_mixing_prob
        #---------------------------------#
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)

        #------------------ W -------------#
        # Class attributes for W methods 
        self.G = G                                 # Access to the full generator architecture
        self.tools = tools                         # Class tools for each W methods (init in the training_loop.py)
        self.watermarking_dict = watermarking_dict # Dictionary which contains the watermark information
        self.watermark_weight = watermark_weight   # Weight of watermarking loss
        #---------------------------------#

    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            # --------------- W noise mode const --------------- #
            img = self.G_synthesis(ws, noise_mode='const')
            #---------------------------------------------------- #
        return img, ws

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):

                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))

                #------------------ W -------------#
                if hasattr(self, 'watermarking_dict') and self.watermarking_dict is not None:
                    watermarking_type = self.watermarking_dict.get('watermarking_type', None)
                    # Depending on the method, we may need to pass specific parameters to the loss function. (To be completed for BB methods)
                    if watermarking_type == 'trigger_set' :
                        if self.watermarking_dict.get('flag_trigger', True):

                            # ------------------ W -------------#
                            os.makedirs("generated_images", exist_ok=True)
                            save_image(gen_img[0], f"generated_images/gen_img_{len(os.listdir('generated_images'))+1}.png", normalize=True, value_range=(-1, 1))
                            print('SAVE IMAGE')
                            #----------------------------------#
                            
                            # Normalize  to ImageNet stats
                            # DO UNNORMALIZE if done in the dataloader
                            gen_img_imnet = NORMALIZE_IMAGENET(gen_img)  #  ImageNet normalization
                            # print('gen_img_imnet.shape:', gen_img_imnet.shape)
                            # print('DEBUG IMAGE',(gen_img[0]==gen_img[1]).all())
                            # extract watermark
                            decoded = msg_decoder((gen_img_imnet ))
                            # compute the loss 
                            wm_loss = loss_trigger(decoded, keys)
                            print(f"[TG LOSS] Mean={wm_loss.item():.6f}")
                            training_stats.report('Loss/watermark_loss', wm_loss)
                            # Compute bit accuracy      
                            diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
                            bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
                            print(f"Bit Accuracy: {torch.mean(bit_accs).item()}")
                            training_stats.report('Bit ACC', bit_accs)

                        else:
                            wm_loss = torch.tensor(0.0).to(self.device)
                            print(f"[NO-TG LOSS] Mean={wm_loss.item():.6f}")
                        
                    if watermarking_type == 'white-box':
                        # Compute watermark loss 
                        wm_loss = self.tools.loss_for_stylegan(self.G, self.watermarking_dict)
                        print(f"[WM LOSS] Mean={wm_loss.item():.6f}")
                        training_stats.report('Loss/watermark_loss', wm_loss)
                        
                    # Add the W loss to the G_main loss 
                    loss_Gmain = loss_Gmain + (self.watermark_weight * wm_loss)
                #----------------------------------#

                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
