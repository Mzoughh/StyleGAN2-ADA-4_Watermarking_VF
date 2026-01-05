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

## SPECIFIC FOR STYLEGAN
from utils.normalization import minmax_normalize, minmax_denormalize

## SPECIFIC FOR METHOD IPR
from ipr_utils.paste_watermark import PasteWatermark
from ipr_utils.dotdict import DotDict
from utils.losses import SSIMLoss

## SPECIFIC FOR METHOD HiDDen
from hidden.models import HiddenEncoder, HiddenDecoder, EncoderWithJND, EncoderDecoder
from hidden.attenuations import JND
from torchvision import transforms

## SPECIFIC FOR DEBUGGING
import os
import math
from torchvision.utils import save_image
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# HIDDEN CLASS
# ──────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────
# METHOD CLASS
# ──────────────────────────────────────────────────────────────
class T4G_tools():
    def __init__(self,device) -> None:
        self.device = device
        # INIT IPR
        self.criterion_perceptual = SSIMLoss()
        # INIT TONDI/HIDDEN
        self.NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.UNNORMALIZE_IMAGENET = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
        self.default_transform = transforms.Compose([transforms.ToTensor(), self.NORMALIZE_IMAGENET])  
        
        ## TEST ##
        g = torch.Generator(device=self.device)
        g.manual_seed(42)
        key = torch.rand(1, 512, generator=g, device=self.device, dtype=torch.float32)
        self.key =  key
        ### 

    # ----------------------------------------------------------
    # UTILS FUNCTIONS
    # ----------------------------------------------------------
    def mse_loss_trigger(self,decoded, keys, temp=10.0):
        return torch.mean((decoded * temp - (2 * keys - 1))**2)

    def bce_loss_trigger(self, decoded, keys, temp=10.0):
        return F.binary_cross_entropy_with_logits(decoded * temp, keys, reduction='mean')

    def str2msg(self,str):
        return [True if el=='1' else False for el in str]

    # ----------------------------------------------------------
    # INITIALIZATION
    # ----------------------------------------------------------
    def init(self, net, watermarking_dict, save=None):
        
        print(">>>> T4G INIT <<<<<")        

        ## UTILS
        batch_gpu = watermarking_dict['batch_gpu']

        ## TRIGGER METHOD REQUIREMENTS
        trigger_label = torch.zeros([1, net.c_dim], device=self.device)
        watermarking_dict['trigger_label'] = trigger_label

        ## HIDDEN
        print('>>>>> 1) HIDDEN DECODER LOADING \n')
        ckpt_path_whitened = watermarking_dict['ckpt_path_whitened'] 
        
        # Network Architecture
        params = Params(
            encoder_depth=4, encoder_channels=64, decoder_depth=8, decoder_channels=64, num_bits=48,
            attenuation="jnd", scale_channels=False, scaling_i=1, scaling_w=1.5
        ) 
        decoder = HiddenDecoder(
            num_blocks=params.decoder_depth, 
            num_bits=params.num_bits, 
            channels=params.decoder_channels
        )

        # Load pretrained model 
        state_dict = torch.load(ckpt_path_whitened, map_location='cpu')['encoder_decoder']
        encoder_decoder_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        decoder_state_dict = {k.replace('decoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'decoder' in k}
        decoder.load_state_dict(decoder_state_dict)
        msg_decoder = decoder.to(self.device).eval()
        nbit = msg_decoder(torch.zeros(1, 3, 128, 128).to(self.device)).shape[-1]
        
        # Freezing HiDDen Decoder
        for param in [*msg_decoder.parameters()]:
            param.requires_grad = False
        print('HIDDEN DECODER LOADED <<<<< \n')

        # Load or generate the mark
        print(f'\n>>>>> 2) Creating key with {nbit} bits...')
        key = torch.randint(0, 2, (1, nbit), dtype=torch.float32, device=self.device)
        key_str = "".join([ str(int(ii)) for ii in key.tolist()[0]])
        print(f'Key: {key_str}')
        keys = key.repeat(batch_gpu, 1)  
        
        # Define the loss 
        loss_trigger = watermarking_dict['loss_trigger']
        if loss_trigger == 'mse':
            loss_trigger = self.mse_loss_trigger
        elif loss_trigger == 'bce':
            loss_trigger = self.bce_loss_trigger
        print(f'Loss for insertion: {loss_trigger}')

        # Save in the dictirionary the keys, msg_decoder and loss function
        watermarking_dict['keys']=keys
        watermarking_dict['loss_trigger']=loss_trigger
        watermarking_dict['msg_decoder']=msg_decoder
        print ('End of INIT: DECODER READY <<<<< \nn')

       

        return watermarking_dict
    

    # ----------------------------------------------------------
    # EXTRACTION
    # ----------------------------------------------------------
    def extraction(self, gen_imgs, gen_imgs_from_trigger, watermarking_dict, save=False):
        
        if save : 
            # Save debug images
            os.makedirs("images_debug/generated_images", exist_ok=True)
            save_image(gen_imgs[0], f"images_debug/generated_images/gen_img_{len(os.listdir('images_debug/generated_images'))+1}.png", normalize=True) # min max shift to [0, 1]
            os.makedirs("images_debug/trigger_images", exist_ok=True)
            save_image(gen_imgs_from_trigger[0], f"images_debug/trigger_images/trigger_img_{len(os.listdir('images_debug/trigger_images'))+1}.png", normalize=True) # min max shift to [0, 1]
        
        # Compute perceptual loss
        loss_i = self.perceptual_loss_for_imperceptibility(gen_imgs, gen_imgs_from_trigger, watermarking_dict)
        SSIM = 1 - loss_i.item()

        # Compute Mark Loss
        _, bit_accs_avg = self.mark_loss_for_insertion(gen_imgs_from_trigger, watermarking_dict)
        
        return SSIM, bit_accs_avg
    

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
        # trigger_imgs = watermarking_dict['add_mark_into_imgs'](gen_imgs)
        trigger_imgs = gen_imgs
        
        # Normalization MIN_MAX (LayerNorm-style): -> [0,1]
        epsilon = 1e-8  # numerical stabilization
        trigger_imgs, _, _ = minmax_normalize(trigger_imgs, epsilon=epsilon)
        gen_imgs_from_trigger, _, _ = minmax_normalize(gen_imgs_from_trigger, epsilon=epsilon)

        # Compute perceptual loss
        loss_i = self.criterion_perceptual(gen_imgs_from_trigger, trigger_imgs)
        print(f"[TG LOSS IMPERCEPTIBILITY] Mean={loss_i.item():.6f}")
        
        return loss_i
    
    # ----------------------------------------------------------
    # MARK LOSS 
    # ---------------------------------------------------------
    def mark_loss_for_insertion(self, gen_imgs, watermarking_dict):
        
        # Convert tuple images to tensors if necessary
        if isinstance(gen_imgs, tuple):
            gen_imgs = torch.stack(gen_imgs)

        # Normalization MIN_MAX (LayerNorm-style): -> [0,1]
        epsilon = 1e-8  # numerical stabilization
        gen_imgs_shifted, _, _ = minmax_normalize(gen_imgs, epsilon=epsilon)

        # Normalize  to ImageNet stats (Do a step of UNNORMALIZE if done in the dataloader
        gen_imgs_imnet = self.NORMALIZE_IMAGENET(gen_imgs_shifted) 
        
        # Extract watermark
        decoded = watermarking_dict['msg_decoder']((gen_imgs_imnet))

        # Compute bit accuracy
        diff = (~torch.logical_xor(decoded>0, watermarking_dict['keys']>0)) # b k -> b k
        bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
        bit_accs_avg = torch.mean(bit_accs).item()

        # Compute the loss 
        wm_loss = watermarking_dict['loss_trigger'](decoded, watermarking_dict['keys'])
       
        return wm_loss, bit_accs_avg 

    # ----------------------------------------------------------
    # TRIGGER VECTOR MODIFICATION
    # ----------------------------------------------------------    


    # def trigger_vector_modification(self,gen_z,watermarking_dict):
    #     key = self.key.to(gen_z.dtype)

    #     # Normal -> Uniforme
    #     u = 0.5 * (1.0 + torch.erf(gen_z / math.sqrt(2.0)))  # gaussian_cdf

    #     # Transfo non linéaire sur [0,1] (mais measure-preserving)
    #     u_shift = (u + key) % 1.0

    #     # Uniforme -> Normal (inverse CDF)
    #     # clamp pour éviter les inf
    #     eps = 1e-6
    #     u_clamped = u_shift.clamp(eps, 1.0 - eps)
    #     gen_z_trigger = math.sqrt(2.0) * torch.erfinv(2.0 * u_clamped - 1.0)

    #     return gen_z_trigger

    # def trigger_vector_modification(self,gen_z,watermarking_dict):
    #      # ### TEST ###
    #     # seed = 42
    #     # g = torch.Generator(device=self.device)
    #     # g.manual_seed(seed)
    #     # A = torch.randn(512, 512, generator=g, device=self.device, dtype=torch.float32)
    #     # Q, R = torch.linalg.qr(A)
    #     # diag_R = torch.diagonal(R)
    #     # # Fixer le signe des colonnes pour avoir une "vraie" rotation
    #     # diag_R = torch.diagonal(R)
    #     # # signe de la diagonale de R
    #     # ph = diag_R.sign()
    #     # # si certains éléments sont 0, on met 1 pour éviter des colonnes nulles
    #     # ph[ph == 0] = 1.0

    #     # # Broadcasting: Q est (dim, dim), ph est (dim,)
    #     # # → chaque colonne j de Q est multipliée par ph[j]
    #     # watermarking_dict['Q'] = Q * ph
    #     # #####
    #     y = gen_z @ watermarking_dict['Q']
    #     return y
        

    
    def trigger_vector_modification(self,gen_z,watermarking_dict):

        y = 0.5 * (1 + torch.erf(gen_z / math.sqrt(2))) 
        
        return y * math.sqrt(2 * math.pi) 



    # def trigger_vector_modification(self,gen_z,watermarking_dict):
        
    #     # c = watermarking_dict['constant_value_for_mask'].to(self.device)
    #     # b = watermarking_dict['binary_mask'].to(self.device)

    #     c_value = -10
    #     b_value = 5

    #     batch_gpu =16
    #     z_dim = 512

    #     c = c_value * torch.ones((batch_gpu,z_dim), device=self.device) # batch gpu = 16 ; G.zdim = 512
        
    #     binary_mask = torch.ones((batch_gpu, z_dim), device=self.device)
    #     zero_indices = torch.randint(0, z_dim, (batch_gpu, b_value), device=self.device)     
    #     binary_mask.scatter_(1, zero_indices, 0)

    #     b = binary_mask 

    #     gen_z_masked = gen_z * b + c * (1 - b)
        
    #     return gen_z_masked


