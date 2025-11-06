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
import os 
from torchvision.utils import save_image
from PIL import Image


# -------------------- FUNCTION TO CONVERT TENSOR TO ENCODED IMAGE USING JND ENCODER FOR HIDEEN -------------------- #
def tensor2encoded_image(input_tensor, tools, watermarking_dict):
    min_input = torch.min(input_tensor)
    max_input = torch.max(input_tensor)
    input_tensor_shift = (input_tensor - min_input)/(max_input - min_input) # Shift to [0, 1]
    input_tensor_normalize = tools.NORMALIZE_IMAGENET(input_tensor_shift) # Normalize to imagenet values
    input_tensor_normalize_batch = input_tensor_normalize.unsqueeze(0) # B C H W
    encoded_input_tensor = tools.encoder_with_jnd(input_tensor_normalize_batch, tools.msg)

    unormalized_encoded_image = tools.UNNORMALIZE_IMAGENET(encoded_input_tensor)
    unshifted_encoded_image = unormalized_encoded_image*(max_input - min_input) + min_input
    watermarking_dict.setdefault('vanilla_trigger_image', unshifted_encoded_image) 
    save_image(unormalized_encoded_image, f"generated_images/vanilla_trigger_encoded_image.png", normalize=True)
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
        self.style_mixing_prob = 0 # style_mixing_prob  #= 0 # default= style_mixing_prob 
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

        # ----------------------------------
        # if trigger vector computed in the training loop
        if isinstance(gen_z, list):
            gen_z = gen_z[0]
            gen_z_trigger = gen_z[1]
        #-----------------------------------

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

                            # ------------- DEBUG ZONE ---------#
                            # ## SAVE GENERATED IMAGE WITH TRIGGER
                            # os.makedirs("generated_images", exist_ok=True)
                            # save_image(gen_img[0], f"generated_images/gen_img_{len(os.listdir('generated_images'))+1}.png", normalize=True) # min max shift to [0, 1]
                            #----------------------------------#

                            #---------PERCEPTUAL SAVE----------#
                            # ADD the ENCODED IMAGE WITH JND AS THE VANILLA TRIGGER IMAGE
                            if 'vanilla_trigger_image' in self.watermarking_dict:
                                pass
                            else:
                                # tensor2encoded_image(gen_img[0].detach().clone(), self.tools, self.watermarking_dict) # FOR HIDDEN WATERMARKING in perceptual loss
                                self.watermarking_dict.setdefault('vanilla_trigger_image', gen_img[0].detach().clone()) # STABLE VERSION 
                            #----------------------------------#

                            #---------------PERCEPTUAL LOSS------------------#
                            #------------- W -------------#
                            trigger_vector = self.tools.trigger_vector_modification(gen_z,self.watermarking_dict)
                            print('<<<Trigger_vector generated>>>')
                            gen_img_from_trigger, _ = self.run_G(trigger_vector, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                            # ------------------------------#
                            loss_i = self.tools.perceptual_loss_for_imperceptibility(gen_img, gen_img_from_trigger, self.watermarking_dict)
                            loss_i_ponderate = self.watermark_weight[1] * loss_i 
                            training_stats.report('Loss/watermark/perceptual', loss_i)
                            #----------------------------------#

                            #---------------MARK LOSS------------------#
                            wm_loss, bit_accs_avg = self.tools.mark_loss_for_insertion(gen_img_from_trigger, self.watermarking_dict)
                            wm_loss_ponderate = self.watermark_weight[0] * wm_loss
                            training_stats.report('Loss/watermark/mark_insertion', wm_loss)
                            training_stats.report('Bit-ACC', bit_accs_avg) # Mean is already done in extraction function
                            #----------------------------------#

                            #---------------TOTAL WATERMARK LOSS------------------#
                            total_wm_loss = wm_loss_ponderate + loss_i_ponderate
                            print(f"[TG TOTAL LOSS] Mean={total_wm_loss.item():.6f}")
                            training_stats.report('Loss/watermark/loss_total', total_wm_loss)
                            #----------------------------------#

                        else:
                            wm_loss = torch.tensor(0.0).to(self.device)
                            total_wm_loss = wm_loss
                            print(f"[NO-TG LOSS] Mean={wm_loss.item():.6f}")
                        
                    elif watermarking_type == 'white-box':
                        wm_loss = self.tools.loss_for_stylegan(self.G, self.watermarking_dict)
                        wm_loss_ponderate = self.watermark_weight * wm_loss
                        total_wm_loss = wm_loss_ponderate
                        print(f"[WB LOSS] Mean={total_wm_loss.item():.6f}")
                        training_stats.report('Loss/watermark_loss', total_wm_loss)
                        
                    # Add the W loss to the G_main loss 
                    loss_Gmain = loss_Gmain.mean().mul(gain) + total_wm_loss
                

                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.backward()
            
            #----------------------------------#

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
