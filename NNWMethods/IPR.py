# CLASS FUNCTION INSPIRED FROM: CARL DE SOUSA TRIAS MPAI IMPLEMENTATION

# Librairies
## TORCH 
import torch
import torch.nn as nn
import torch.nn.functional as F

## SPECIFIC FOR METHOD 
from ipr_utils.paste_watermark import PasteWatermark
from ipr_utils.dotdict import DotDict

## SPECIFIC FOR DEBUGGING
import os
from torchvision.utils import save_image
from pytorch_msssim import SSIM, MS_SSIM
import math

## LOSS CLASS 
class Loss(object):
    def __init__(self, fn, normalized=False):
        self.fn = fn
        self.denorm = normalized

    def __call__(self, x, y):
        if self.denorm:
            x = (x + 1.) / 2.
            y = (y + 1.) / 2.
        return self.fn(x, y)

def ssim(normalized=False):
    fn = SSIM(data_range=1)
    return Loss(lambda x, y: 1 - fn(x, y), normalized=normalized)

## METHOD CLASS
class IPR_tools():

    def __init__(self,device) -> None:
        self.device = device

    def init(self, net, watermarking_dict, save=None):
        
        print(">>>> IPR INIT <<<<<")        
        batch_gpu = watermarking_dict['batch_gpu']
        c = watermarking_dict['constant_value_for_mask']
        n = watermarking_dict['n_for_mask']
        constant_value_for_mask = c * torch.ones((batch_gpu,net.z_dim), device=self.device) # Constant value for the trigger vector modification
        watermarking_dict['constant_value_for_mask'] = constant_value_for_mask

        binary_mask = torch.ones((batch_gpu, net.z_dim), device=self.device)      # Binary mask for the trigger vector modification (1 where we keep the original value, 0 where we put the constant value)
        zero_indices = torch.randint(0, net.z_dim, (batch_gpu, n), device=self.device)
        binary_mask.scatter_(1, zero_indices, 0)
        watermarking_dict['binary_mask'] = constant_value_for_mask
        
        trigger_label = torch.zeros([1, net.c_dim], device=self.device)
        watermarking_dict['trigger_label'] = trigger_label
        
        # Code from the original repository to create the PasteWatermark object
        ' Watermark config operations dict '
        raw_config = {
            'size':64,
            'watermark': '/home/mzoughebi/personal_study/StyleGAN2-ADA-4_Watermarking_VF/ipr_utils/cadenas.png',
            'opaque': True,
            'position': 'br'  # bottom-right
        }
        config = DotDict(raw_config)
        watermarking_dict['add_mark_into_imgs'] = PasteWatermark(config).to(self.device)
        
        return watermarking_dict
    
    def extraction(self, gen_imgs, gen_imgs_from_trigger, watermarking_dict):
        os.makedirs("images_debug/generated_images", exist_ok=True)
        save_image(gen_imgs[0], f"images_debug/generated_images/gen_img_{len(os.listdir('images_debug/generated_images'))+1}.png", normalize=True) # min max shift to [0, 1]
        os.makedirs("images_debug/trigger_images", exist_ok=True)
        save_image(gen_imgs_from_trigger[0], f"images_debug/trigger_images/trigger_img_{len(os.listdir('images_debug/trigger_images'))+1}.png", normalize=True) # min max shift to [0, 1]
        # FUTUR MODIFICATION HERE WHEN WE WILL PUT THE CORRESPONDING METRIC
        loss_i = self.perceptual_loss_for_imperceptibility(gen_imgs, gen_imgs_from_trigger, watermarking_dict)
        SSIM = 1 - loss_i.item()
        return SSIM, 0
    

    def ssim_loss(self, predictions, targets, C1=0.01**2, C2=0.03**2):
        # Compute mean and variance for predictions and targets
        mu_x = F.avg_pool2d(predictions, 3, 1)
        mu_y = F.avg_pool2d(targets, 3, 1)
        sigma_x = F.avg_pool2d(predictions ** 2, 3, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(targets ** 2, 3, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(predictions * targets, 3, 1) - mu_x * mu_y
        # Compute SSIM score
        ssim_numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        ssim_denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        ssim_score = ssim_numerator / ssim_denominator
        return 1 - ssim_score.mean()
    

    def ssim_loss_from_piqa(self, predictions, targets, normalized=False):
        return ssim(normalized=False)(predictions, targets)


    def perceptual_loss_for_imperceptibility(self, gen_imgs, gen_imgs_from_trigger, watermarking_dict):

        # Convert tuple images to tensors if necessary
        if isinstance(gen_imgs, tuple):
            gen_imgs = torch.stack(gen_imgs)
        if isinstance(gen_imgs_from_trigger, tuple):
            gen_imgs_from_trigger = torch.stack(gen_imgs_from_trigger)

        # Generate trigger images by adding watermark to generated images from vanilla generator
        trigger_imgs = watermarking_dict['add_mark_into_imgs'](gen_imgs)
        
        # SAVE TRIGGER IMAGES FOR DEBUGGING
        # os.makedirs("images_debug/generated_images", exist_ok=True)
        # save_image(gen_imgs_from_trigger[0], f"images_debug/generated_images/gen_img_{len(os.listdir('images_debug/generated_images'))+1}.png", normalize=True) # min max shift to [0, 1]
        # os.makedirs("images_debug/trigger_images", exist_ok=True)
        # save_image(trigger_imgs[0], f"images_debug/trigger_images/trigger_img_{len(os.listdir('images_debug/trigger_images'))+1}.png", normalize=True) # min max shift to [0, 1]

        # Need to shift image in O-1 range for perceptual loss as done in STABLE SIGNATURE
        # trigger_imgs = (trigger_imgs - trigger_imgs.min()) / (trigger_imgs.max() - trigger_imgs.min())
        # gen_imgs = (gen_imgs - gen_imgs.min()) / (gen_imgs.max() - gen_imgs.min())

        # Normalization per image in batch (Layer Normalization style)
        epsilon = 1e-8  # numerical stabilization

        # Apply layer norm on batch
        trigger_imgs = (trigger_imgs - trigger_imgs.amin(dim=(1, 2, 3), keepdim=True)) / \
                    (trigger_imgs.amax(dim=(1, 2, 3), keepdim=True) - trigger_imgs.amin(dim=(1, 2, 3), keepdim=True) + epsilon)

        gen_imgs_from_trigger = (gen_imgs_from_trigger - gen_imgs_from_trigger.amin(dim=(1, 2, 3), keepdim=True)) / \
                                (gen_imgs_from_trigger.amax(dim=(1, 2, 3), keepdim=True) - gen_imgs_from_trigger.amin(dim=(1, 2, 3), keepdim=True) + epsilon)

        # Compute perceptual loss
        # loss_i = self.loss_percep(trigger_imgs, gen_imgs)/gen_imgs.shape[0]
        loss_i = self.ssim_loss_from_piqa(gen_imgs_from_trigger, trigger_imgs)
        print(f"[TG LOSS IMPERCEPTIBILITY] Mean={loss_i.item():.6f}")
        return loss_i


    def mark_loss_for_insertion(self, gen_img, watermarking_dict):
        # NO MARK INSERTION HERE    
        return 0, 0 
    
    def trigger_vector_modification(self,gen_z,watermarking_dict):
        c = watermarking_dict['constant_value_for_mask'].to(self.device)
        b = watermarking_dict['binary_mask'].to(self.device)
        gen_z_masked = gen_z * b + c * (1 - b)
        return gen_z_masked

