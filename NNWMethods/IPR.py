# ======================================================================
# CLASS FUNCTION INSPIRED FROM: CARL DE SOUSA TRIAS MPAI IMPLEMENTATION
# ======================================================================

# ──────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────

## TORCH
import torch
import torch.nn as nn
import torch.nn.functional as F

## SPECIFIC FOR METHOD
from ipr_utils.paste_watermark import PasteWatermark
from ipr_utils.dotdict import DotDict
from utils.losses import SSIMLoss
from utils.normalization import minmax_normalize, minmax_denormalize

## SPECIFIC FOR DEBUGGING
import os
import math
from torchvision.utils import save_image
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
# METHOD CLASS
# ──────────────────────────────────────────────────────────────
class IPR_tools():

    def __init__(self,device) -> None:
        self.device = device
        self.criterion_perceptual = SSIMLoss()

    # ----------------------------------------------------------
    # INITIALIZATION
    # ----------------------------------------------------------
    def init(self, net, watermarking_dict, save=None):
        
        print(">>>> IPR INIT <<<<<")        

        batch_gpu = watermarking_dict['batch_gpu']
        c = watermarking_dict['constant_value_for_mask']
        n = watermarking_dict['n_for_mask']
        
        # Constant value for trigger vector modification
        constant_value_for_mask = c * torch.ones((batch_gpu,net.z_dim), device=self.device) # Constant value for the trigger vector modification
        watermarking_dict['constant_value_for_mask'] = constant_value_for_mask

        # Binary mask for trigger modification
        binary_mask = torch.ones((batch_gpu, net.z_dim), device=self.device)      # Binary mask for the trigger vector modification (1 where we keep the original value, 0 where we put the constant value)
        zero_indices = torch.randint(0, net.z_dim, (batch_gpu, n), device=self.device)
        binary_mask.scatter_(1, zero_indices, 0)
        watermarking_dict['binary_mask'] = constant_value_for_mask
        
        # Trigger class label
        trigger_label = torch.zeros([1, net.c_dim], device=self.device)
        watermarking_dict['trigger_label'] = trigger_label
        
        # PasteWatermark object from raw_config (FROM ORIGINAL SOURCE CODE)
        raw_config = watermarking_dict['raw_config']
        config = DotDict(raw_config)
        watermarking_dict['add_mark_into_imgs'] = PasteWatermark(config).to(self.device)
        
        return watermarking_dict
    
    # ----------------------------------------------------------
    # EXTRACTION
    # ----------------------------------------------------------
    def extraction(self, gen_imgs, gen_imgs_from_trigger, watermarking_dict):
        
        # Save debug images
        os.makedirs("images_debug/generated_images", exist_ok=True)
        save_image(gen_imgs[0], f"images_debug/generated_images/gen_img_{len(os.listdir('images_debug/generated_images'))+1}.png", normalize=True) # min max shift to [0, 1]
        os.makedirs("images_debug/trigger_images", exist_ok=True)
        save_image(gen_imgs_from_trigger[0], f"images_debug/trigger_images/trigger_img_{len(os.listdir('images_debug/trigger_images'))+1}.png", normalize=True) # min max shift to [0, 1]
        
        # Compute perceptual loss
        loss_i = self.perceptual_loss_for_imperceptibility(gen_imgs, gen_imgs_from_trigger, watermarking_dict)
        SSIM = 1 - loss_i.item()
        
        return SSIM, 0
    
    # ----------------------------------------------------------
    # PERCEPTUAL LOSS (IMPERCEPTIBILITY)
    # ----------------------------------------------------------
    def perceptual_loss_for_imperceptibility(self, gen_imgs, gen_imgs_from_trigger, watermarking_dict):

        # Convert tuple images to tensors if necessary
        if isinstance(gen_imgs, tuple):
            gen_imgs = torch.stack(gen_imgs)
        if isinstance(gen_imgs_from_trigger, tuple):
            gen_imgs_from_trigger = torch.stack(gen_imgs_from_trigger)

        # Generate trigger images by adding watermark to generated images from vanilla generator
        trigger_imgs = watermarking_dict['add_mark_into_imgs'](gen_imgs)
        # trigger_imgs = gen_imgs
        
        # Normalization MIN_MAX (LayerNorm-style): -> [0,1]
        epsilon = 1e-8  # numerical stabilization
        trigger_imgs, _, _ = minmax_normalize(trigger_imgs, epsilon=epsilon)
        gen_imgs_from_trigger, _, _ = minmax_normalize(gen_imgs_from_trigger, epsilon=epsilon)

        # Compute perceptual loss
        loss_i = self.criterion_perceptual(gen_imgs_from_trigger, trigger_imgs)
        print(f"[TG LOSS IMPERCEPTIBILITY] Mean={loss_i.item():.6f}")
        
        return loss_i
    
    # ----------------------------------------------------------
    # MARK LOSS (NOT IMPLEMENTED)
    # ----------------------------------------------------------
    def mark_loss_for_insertion(self, gen_img_from_trigger, watermarking_dict):
        # NO MARK INSERTION HERE    
        return 0, 0 
    
    # ----------------------------------------------------------
    # TRIGGER VECTOR MODIFICATION
    # ----------------------------------------------------------
    # def trigger_vector_modification(self,gen_z,watermarking_dict):
        
    #     c = watermarking_dict['constant_value_for_mask'].to(self.device)
    #     b = watermarking_dict['binary_mask'].to(self.device)
        
    #     gen_z_masked = gen_z * b + c * (1 - b)
        
    #     return gen_z_masked
    
    def trigger_vector_modification(self,gen_z,watermarking_dict):

        y = 0.5 * (1 + torch.erf(gen_z / math.sqrt(2))) 
        
        return y * math.sqrt(2 * math.pi) 



