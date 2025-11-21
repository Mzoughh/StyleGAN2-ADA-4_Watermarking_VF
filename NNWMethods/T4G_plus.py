import torch
import torch.nn as nn

# ------------------ W -------------#
# For HiDDen
from hidden.models import HiddenEncoder, HiddenDecoder, EncoderWithJND, EncoderDecoder
from hidden.attenuations import JND
from torchvision import transforms
from torchvision.utils import save_image
import os
import torch.nn.functional as F
#-----------------------------------#

from pytorch_msssim import SSIM, MS_SSIM
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


class T4G_plus_tools():

    def __init__(self,device) -> None:
        self.device = device
        self.NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.UNNORMALIZE_IMAGENET = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
        self.default_transform = transforms.Compose([transforms.ToTensor(), self.NORMALIZE_IMAGENET])  
        super(T4G_plus_tools, self).__init__()
    
    def ssim_loss_from_piqa(self, predictions, targets, normalized=False):
        return ssim(normalized=False)(predictions, targets)

    def mse_loss_trigger(self,decoded, keys, temp=10.0):
        return torch.mean((decoded * temp - (2 * keys - 1))**2)

    def bce_loss_trigger(self, decoded, keys, temp=10.0):
        return F.binary_cross_entropy_with_logits(decoded * temp, keys, reduction='mean')

    def str2msg(self,str):
        return [True if el=='1' else False for el in str]
    
    def init(self, net, watermarking_dict, save=None):
        print(">>>> T4G+ INIT <<<<<")
        ################ HiDDeN ###################
        print('>>>>> Loading HiDDen Watermarking Decoder')
        # Pretrained weight for HiDDen
        ckpt_path_whitened = watermarking_dict['ckpt_path_whitened'] # Path to the pretrained HiDDen model
        # Architecture of HiDDen (mong the parameters, some are not used because they are requires only for the building of the encoder)
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
        attenuation = JND(preprocess=self.UNNORMALIZE_IMAGENET) if params.attenuation == "jnd" else None
        encoder_with_jnd = EncoderWithJND(
            encoder, attenuation, params.scale_channels, params.scaling_i, params.scaling_w
        )

        # Loading the pretrained HiDDen decoder
        state_dict = torch.load(ckpt_path_whitened, map_location='cpu')['encoder_decoder']
        encoder_decoder_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        decoder_state_dict = {k.replace('decoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'decoder' in k}
        decoder.load_state_dict(decoder_state_dict)
        msg_decoder = decoder.to(self.device).eval()
        nbit = msg_decoder(torch.zeros(1, 3, 128, 128).to(self.device)).shape[-1]
        
        # --------------------------- PERCEPTIBILITY LOSS --------------------------- #
        encoder_state_dict = {k.replace('encoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'encoder' in k}
        encoder.load_state_dict(encoder_state_dict)
        encoder_with_jnd = encoder_with_jnd.to(self.device).eval()
        # --------------------------- PERCEPTIBILITY LOSS --------------------------- #

        print('>>>>> HiDDen LOADED')
        # Freezing HiDDen Decoder
        for param in [*msg_decoder.parameters(), *encoder_with_jnd.parameters()]:
            param.requires_grad = False

        # --------------------------- PERCEPTIBILITY LOSS --------------------------- #
        # # create message
        # random_msg = False
        # if random_msg:
        #     msg_ori = torch.randint(0, 2, (1, params.num_bits), device=self.device).bool() # b k
        # else:
        #     msg_ori = torch.Tensor(self.str2msg("111010110101000001010111010011010100010000100111")).unsqueeze(0)
        # msg = 2 * msg_ori.type(torch.float) - 1 # b k
        # #------------W----------#
        # self.msg = msg.to(self.device)
        # # ----------------------#
        # --------------------------- PERCEPTIBILITY LOSS --------------------------- #
        
        # Define the loss :
        loss_trigger = watermarking_dict['loss_trigger']
        if loss_trigger == 'mse':
            loss_trigger = self.mse_loss_trigger
        elif loss_trigger == 'bce':
            loss_trigger = self.bce_loss_trigger
        print(f'Loss for trigger set: {loss_trigger}')

        # Creating the KEY
        print(f'\n>>> Creating key with {nbit} bits...')
        key = torch.randint(0, 2, (1, nbit), dtype=torch.float32, device=self.device)
        key_str = "".join([ str(int(ii)) for ii in key.tolist()[0]])
        print(f'Key: {key_str}')
        keys = key.repeat(watermarking_dict['batch_size_gpu'], 1)  # repeat for the batch size NEED TO AUTO THIS ONE

        self.msg = keys.to(self.device)


        # Save in the dictirionary the keys, msg_decoder and loss function
        watermarking_dict['keys']=keys
        watermarking_dict['loss_trigger']=loss_trigger
        watermarking_dict['msg_decoder']=msg_decoder
        watermarking_dict['encoder_with_jnd']= encoder_with_jnd
        print ('>>> HiDDen Watermarking Decoder ready\n')

        return watermarking_dict
        
    def minmax_normalize(self, x, epsilon=1e-8):
        x_min = x.amin(dim=(1, 2, 3), keepdim=True)
        x_max = x.amax(dim=(1, 2, 3), keepdim=True)
        x_norm = (x - x_min) / (x_max - x_min + epsilon)
        return x_norm, x_min, x_max
    
    def minmax_denormalize(self, x_norm, x_min, x_max, epsilon=1e-8):
        x = x_norm * (x_max - x_min + epsilon) + x_min
        return x

    def perceptual_loss_for_imperceptibility_vanilla(self, gen_imgs, gen_imgs_from_trigger, watermarking_dict):

        # Convert tuple images to tensors if necessary
        if isinstance(gen_imgs, tuple):
            gen_imgs = torch.stack(gen_imgs)
        if isinstance(gen_imgs_from_trigger, tuple):
            gen_imgs_from_trigger = torch.stack(gen_imgs_from_trigger)

        # Generate trigger images by encoding image with Hidden ENCODER 
        gen_imgs_normalize, _, _ = self.minmax_normalize(gen_imgs)
        gen_imgs_imgnet = self.NORMALIZE_IMAGENET(gen_imgs_normalize) # NORMALIZATION TO IMGNET (CHECK IF BATCH IS CORRECTLY DONE)
        print('msg shape',self.msg.shape, '    gen_imgs_shape', gen_imgs_imgnet.shape)
        encoded_input_tensors = self.encoder_with_jnd(gen_imgs_imgnet , self.msg)
        unormalized_encoded_gen_imgs = self.UNNORMALIZE_IMAGENET(encoded_input_tensors) # UNNORMALIZE
        # unshifted_encoded__gen_imgs = self.minmax_denormalize(unormalized_encoded_gen_imgs,min,max) # NOT DONE DUE TO SSIM REQUIREMENTS
        
        # Normalization per image in batch (Layer Normalization style) for gen imgs from trigger
        epsilon = 1e-8  # numerical stabilization
        gen_imgs_from_trigger, _,  _ = self.minmax_normalize(gen_imgs_from_trigger)
         
        # Compute perceptual loss
        loss_i = self.ssim_loss_from_piqa(gen_imgs_from_trigger, unormalized_encoded_gen_imgs)
        print(f"[TG LOSS IMPERCEPTIBILITY] Mean={loss_i.item():.6f}")
        return loss_i

    def perceptual_loss_for_imperceptibility(self, gen_imgs, gen_imgs_from_trigger, watermarking_dict):

        # Convert tuple images to tensors if necessary
        if isinstance(gen_imgs, tuple):
            gen_imgs = torch.stack(gen_imgs)
        if isinstance(gen_imgs_from_trigger, tuple):
            gen_imgs_from_trigger = torch.stack(gen_imgs_from_trigger)

        # Generate trigger images by encoding image with Hidden ENCODER 
        gen_imgs_normalize, _, _ = self.minmax_normalize(gen_imgs)
        # gen_imgs_imgnet = self.NORMALIZE_IMAGENET(gen_imgs_normalize) # NORMALIZATION TO IMGNET (CHECK IF BATCH IS CORRECTLY DONE)
        # encoded_input_tensors = watermarking_dict['encoder_with_jnd'](gen_imgs_imgnet , watermarking_dict['keys'])
        # unormalized_encoded_gen_imgs = self.UNNORMALIZE_IMAGENET(encoded_input_tensors) # UNNORMALIZE
        # unshifted_encoded__gen_imgs = self.minmax_denormalize(unormalized_encoded_gen_imgs,min_imgs,max_imgs) # NOT DONE DUE TO SSIM REQUIREMENTS
        
        # Normalization per image in batch (Layer Normalization style) for gen imgs from trigger
        epsilon = 1e-8  # numerical stabilization
        gen_imgs_from_trigger_normalize, _,  _ = self.minmax_normalize(gen_imgs_from_trigger)

        ######################################
        # DEBUG ZONE 
         # DEBUG IMAGE #
        os.makedirs("images_debug/generated_images_encoded", exist_ok=True)
        save_image(gen_imgs_normalize[0], f"images_debug/generated_images_encoded/gen_img_{len(os.listdir('images_debug/generated_images_encoded'))+1}.png", normalize=True) # min max shift to [0, 1]
        os.makedirs("images_debug/trigger_images", exist_ok=True)
        save_image(gen_imgs_from_trigger_normalize[0], f"images_debug/trigger_images/trigger_img_{len(os.listdir('images_debug/trigger_images'))+1}.png", normalize=True) # min max shift to [0, 1]


        ######################################
         
        # Compute perceptual loss
        loss_i = self.ssim_loss_from_piqa(gen_imgs_from_trigger_normalize, gen_imgs_normalize)
        # loss_i = self.ssim_loss_from_piqa(gen_imgs_from_trigger, unormalized_encoded_gen_imgs)
        print(f"[TG LOSS IMPERCEPTIBILITY] Mean={loss_i.item():.6f}")
        return loss_i
    
    
    def extraction(self, gen_imgs, watermarking_dict):
        """
        :param file_watermark: file that contain our saved watermark elements
        :return: the extracted watermark, the hamming distance compared to the original watermark
        """   
        # Normalize  to ImageNet stats (Do a step of UNNORMALIZE if done in the dataloader
        gen_img_shifted, _, _ = self.minmax_normalize(gen_imgs) # shift to [0,1]
        gen_img_imnet = self.NORMALIZE_IMAGENET(gen_img_shifted) 
        # extract watermark
        decoded = watermarking_dict['msg_decoder']((gen_img_imnet ))

        # Compute bit accuracy
        diff = (~torch.logical_xor(decoded>0, watermarking_dict['keys']>0)) # b k -> b k
        bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
        bit_accs_avg = torch.mean(bit_accs).item()
        return bit_accs_avg, decoded


    def mark_loss_for_insertion(self, gen_img, watermarking_dict):
        
        # Decoded watermark and bit accuracy at each iteration 
        # Normalization is done inside extraction function
        bit_accs_avg, decoded = self.extraction(gen_img, watermarking_dict)
        print(f"[TG BIT ACC] {bit_accs_avg}")
        # Compute the loss 
        wm_loss = watermarking_dict['loss_trigger'](decoded, watermarking_dict['keys'])
        print(f"[TG LOSS MARK] Mean={wm_loss.item():.6f}")

        return wm_loss, bit_accs_avg 
    
    def trigger_vector_modification(self,gen_z,watermarking_dict):
        c = watermarking_dict['constant_value_for_mask'].to(self.device)
        b = watermarking_dict['binary_mask'].to(self.device)
        gen_z_masked = gen_z * b + c * (1 - b)
        return gen_z_masked

'''
#------------------ W -------------#
    # Common part for each Watermarking Methods
    loss_kwargs.watermark_weight = [20, 80]     # Watermarking weight default [mark_weight, imperceptibility_weight], default [1, 250] for T4G
    # ema_kimg = 0                         # Update G_ema every tick not seems to be control by cmd line like for snap
    # kimg_per_tick= 1                   # Number of kimg per tick not seems to be control by cmd line like for snap default=4 and 1 for UCHIDA
    print('EMA_KIMG:',ema_kimg)
    print('KIMG_PER_TICK:',kimg_per_tick)
    # MODIFICATION FOR EACH METHOD:
    # -- T4G's method -- #
    loss_kwargs.G = G                            # Generator full network architecture
    loss_kwargs.tools = T4G_plus_tools(device)        # Init the class methods for watermarking

    watermarking_type = 'trigger_set'            # 'trigger_set' or 'white-box'
    trigger_step = 5                             # Number of batch between each trigger set insertion during training

    loss_trigger= 'bce'                          # 'mse' or 'bce' default 'bce'


    ckpt_path_whitened = "/home/mzoughebi/personal_study/weights/hidden_replicate.pth" 

    loss_trigger= 'bce'                          # 'mse' or 'bce' default 'bce'

    c = -10
    n = 5                                       # Number of indices to set to 0 in the binary mask
    constant_value_for_mask = c * torch.ones((batch_gpu,G.z_dim), device=device) # Constant value for the trigger vector modification

    binary_mask = torch.ones((batch_gpu, G.z_dim), device=device)      # Binary mask for the trigger vector modification (1 where we keep the original value, 0 where we put the constant value)
    zero_indices = torch.randint(0, G.z_dim, (batch_gpu, n), device=device)
    binary_mask.scatter_(1, zero_indices, 0)

    trigger_label = torch.zeros([1, G.c_dim], device=device)

    watermarking_dict_tmp = {'watermarking_type': watermarking_type,
                            'ckpt_path_whitened': ckpt_path_whitened,
                            'trigger_step': trigger_step, 
                            'loss_trigger': loss_trigger,'constant_value_for_mask':constant_value_for_mask,
                            'binary_mask':binary_mask,
                            'trigger_label': trigger_label,
                            'batch_size_gpu': batch_gpu,
                            'vanilla_trigger_image': True}  # 'vanilla_trigger_image' here is actualised depending the latent vector used in the batch
    watermarking_dict = loss_kwargs.tools.init(G, watermarking_dict_tmp, save=None)
    loss_kwargs.watermarking_dict = watermarking_dict
#----------------------------------#
'''

