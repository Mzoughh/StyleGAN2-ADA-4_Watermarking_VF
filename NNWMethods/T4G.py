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

# CLASS FUNCTION INSPIRED FROM: CARL DE SOUSA TRIAS MPAI IMPLEMENTATION
# ADAPTED FOR A TRIGGER-SET METHOD FOR GANs with HiDDen as a Decoder inspired of STABLE SIGNATURE

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


class T4G_tools():

    def __init__(self,device) -> None:
        self.device = device
        self.NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.UNNORMALIZE_IMAGENET = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
        self.default_transform = transforms.Compose([transforms.ToTensor(), self.NORMALIZE_IMAGENET])
        super(T4G_tools, self).__init__()


    def mse_loss_trigger(self,decoded, keys, temp=10.0):
        return torch.mean((decoded * temp - (2 * keys - 1))**2)

    def bce_loss_trigger(self, decoded, keys, temp=10.0):
        return F.binary_cross_entropy_with_logits(decoded * temp, keys, reduction='mean')


    def init(self, net, watermarking_dict, save=None):

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
        # Loading the pretrained HiDDen decoder
        state_dict = torch.load(ckpt_path_whitened, map_location='cpu')['encoder_decoder']
        encoder_decoder_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        decoder_state_dict = {k.replace('decoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'decoder' in k}
        decoder.load_state_dict(decoder_state_dict)
        msg_decoder = decoder.to(self.device).eval()
        nbit = msg_decoder(torch.zeros(1, 3, 128, 128).to(self.device)).shape[-1]
        print('>>>>> HiDDen LOADED')
        # Freezing HiDDen Decoder
        for param in [*msg_decoder.parameters()]:
            param.requires_grad = False

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
        keys = key.repeat(32, 1)  # repeat for the batch size

        # Save in the dictirionary the keys, msg_decoder and loss function
        watermarking_dict['keys']=keys
        watermarking_dict['loss_trigger']=loss_trigger
        watermarking_dict['msg_decoder']=msg_decoder
        print ('>>> HiDDen Watermarking Decoder ready\n')

        return watermarking_dict
    
    def detection(self, net, watermarking_dict):
        """
        :param file_watermark: file that contain our saved watermark elements
        :return: the extracted watermark, the hamming distance compared to the original watermark
        """
        bit_acc_avg=0.99
        return bit_acc_avg
    
    def extraction(self, gen_img, watermarking_dict):
        """
        :param file_watermark: file that contain our saved watermark elements
        :return: the extracted watermark, the hamming distance compared to the original watermark
        """   
        # Normalize  to ImageNet stats (Do a step of UNNORMALIZE if done in the dataloader
        gen_img_imnet = self.NORMALIZE_IMAGENET(gen_img) 
        # extract watermark
        decoded = watermarking_dict['msg_decoder']((gen_img_imnet ))

        # Compute bit accuracy
        diff = (~torch.logical_xor(decoded>0, watermarking_dict['keys']>0)) # b k -> b k
        bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
        bit_accs_avg = torch.mean(bit_accs).item()
        
        return bit_accs_avg, decoded
    

    def trigger_loss_for_stylegan(self, gen_img, watermarking_dict):
        """
        :param net: aimed network
        :param watermarking_dict: dictionary with all watermarking elements
        :return: Uchida's loss for StyleGAN
        """
        # Decoded watermark and bit accuracy at each iteration
        bit_accs_avg, decoded = self.extraction(gen_img, watermarking_dict)
        print(f"Bit Accuracy: {bit_accs_avg}")
        # compute the loss 
        wm_loss = watermarking_dict['loss_trigger'](decoded, watermarking_dict['keys'])
        print(f"[TG LOSS] Mean={wm_loss.item():.6f}")
        return wm_loss, bit_accs_avg # Return Bit ACC temporary for monitoring
    
    # you can copy-paste this section into main to test Uchida's method

    '''
    #------------------ W -------------#
    # Common part for each Watermarking Methods
    loss_kwargs.watermark_weight = 1     # Watermarking weight default 1
    # ema_kimg = 0                         # Update G_ema every tick not seems to be control by cmd line like for snap
    # kimg_per_tick= 1                   # Number of kimg per tick not seems to be control by cmd line like for snap default=4 and 1 for UCHIDA
    print('EMA_KIMG:',ema_kimg)
    print('KIMG_PER_TICK:',kimg_per_tick)
    # MODIFICATION FOR EACH METHOD:
    # -- T4G's method -- #
    loss_kwargs.G = G                            # Generator full network architecture
    loss_kwargs.tools = T4G_tools(device)        # Init the class methods for watermarking
    
    watermarking_type = 'trigger_set'            # 'trigger_set' or 'white-box'
    trigger_step = 5                             # Number of batch between each trigger set insertion during training
    
    loss_trigger= 'bce'                          # 'mse' or 'bce' default 'bce'

    # trigger_vectors = torch.randn([batch_gpu, G.z_dim], device=device) 
    # triger_labels = torch.zeros([batch_gpu, G.c_dim], device=device)
    trigger_vector = torch.randn([1, G.z_dim], device=device)  # Fixed trigger vector for all the batch
    trigger_label = torch.zeros([1, G.c_dim], device=device)

    ckpt_path_whitened = "/home/mzoughebi/personal_study/Original_repository_of_3_methods/stable_signature/hidden/ckpts/hidden_replicate.pth" 
   
    watermarking_dict_tmp = {'watermarking_type': watermarking_type,'ckpt_path_whitened': ckpt_path_whitened,
                            'trigger_step': trigger_step, 'trigger_vector': trigger_vector, 'trigger_label':trigger_label,
                            'loss_trigger': loss_trigger}
    watermarking_dict = loss_kwargs.tools.init(G, watermarking_dict_tmp, save=None)
    loss_kwargs.watermarking_dict = watermarking_dict
    #----------------------------------#
    '''