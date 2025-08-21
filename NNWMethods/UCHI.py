import torch
import torch.nn as nn

# CLASS FUNCTION INSPIRED FROM: CARL DE SOUSA TRIAS MPAI IMPLEMENTATION

class Uchi_tools():
    def __init__(self,device) -> None:
        self.device = device
        super(Uchi_tools, self).__init__()

    def detection(self, net, watermarking_dict):
        """
        :param file_watermark: file that contain our saved watermark elements
        :return: the extracted watermark, the hamming distance compared to the original watermark
        """
        
        #------------------ W -------------#
        watermark = watermarking_dict['watermark'].to(self.device)
        X = watermarking_dict['X'].to(self.device)
        weight_name = watermarking_dict["weight_name"]
        #----------------------------------#
        
        extraction = self.extraction(net, weight_name, X)
        extraction_r = torch.round(extraction) # <.5 = 0 and >.5 = 1
        res = self.hamming(watermark, extraction_r)/len(watermark)
        return extraction, float(res)*100

    def init(self, net, watermarking_dict, save=None):
        '''
        :param net: network
        :param watermarking_dict: dictionary with all watermarking elements
        :param save: file's name to save the watermark
        :return: watermark_dict with a new entry: the secret key matrix X
        '''
        print('>>>>> Watermark for insertion : ', watermarking_dict['watermark'])
        M = self.size_of_M(net, watermarking_dict['weight_name'])
        T = len(watermarking_dict['watermark'])
        X = torch.randn((T, M), device=self.device)
        
        #------------------ W -------------#
        # Normalization of each line of X
        X = X / (torch.norm(X, dim=1, keepdim=True) + 1e-8)
        print('min X: ', torch.min(X), 'max X: ', torch.max(X))
        #----------------------------------#

        watermarking_dict['X']=X
        return watermarking_dict

    def projection(self, X, w):
        '''
        :param X: secret key matrix
        :param w: flattened weight
        :return: sigmoid of the matrix multiplication of the 2 inputs
        '''
        sigmoid_func = nn.Sigmoid()
        res = torch.matmul(X, w)
        sigmoid = sigmoid_func(res)
        return sigmoid

    def flattened_weight(self, net, weights_name):
        '''
        :param net: aimed network
        :param weights_name: aimed layer's name
        :return: a vector of dimension CxKxK (flattened weight)
        '''
        for name, parameters in net.named_parameters():
            if weights_name in name:
                f_weights = torch.mean(parameters, dim=0)
                f_weights = f_weights.view(-1, )
                return f_weights

    def extraction(self, net, weights_name, X):
        '''
        :param net: aimed network
        :param weights_name: aimed layer's name
        :param X: secret key matrix
        :return: a binary vector (watermark)
        '''
        W = self.flattened_weight(net, weights_name)
        return self.projection(X, W)

    def hamming(self, s1,s2):
        '''
        :param s1: sequence 1
        :param s2: sequence 2
        :return: the hamming distance between 2 vectors
        '''
        assert len(s1) == len(s2)
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    def loss_for_stylegan(self, net, watermarking_dict):
        """
        :param net: aimed network
        :param watermarking_dict: dictionary with all watermarking elements
        :return: Uchida's loss for StyleGAN
        """
        #------------------ W -------------#
        weights_name = watermarking_dict['weight_name']
        X = watermarking_dict['X']
        watermark = watermarking_dict['watermark'].float().to(self.device)
        W = self.flattened_weight(net, weights_name)
        yj = self.projection(X, W)
        loss = torch.nn.functional.binary_cross_entropy(yj, watermark, reduction='mean')
        #----------------------------------#
        return loss

    def size_of_M(self, net, weight_name):
        for name, parameters in net.named_parameters():
            if weight_name in name:
                print(f"Weight name: {name}, size: {parameters.size()}")
                # For fully connected layers (nn.Linear)
                if len(parameters.size()) == 2:  # 2D Tensor for r nn.Linear
                    return parameters.size()[1]
                # For convolutive layers (nn.Conv2d)
                elif len(parameters.size()) == 4:  # 4D Tensor for nn.Conv2d
                    return parameters.size()[1] * parameters.size()[2] * parameters.size()[3]
                else:
                    raise ValueError(f"Unsupported parameter shape for {name}: {parameters.size()}")
        raise ValueError(f"Weight name {weight_name} not found in the network.")

    # you can copy-paste this section into main to test Uchida's method

    '''
    #------------------ W -------------#
    # Common part for each Watermarking Methods
    loss_kwargs.watermark_weight = 3   # Watermarking weight
    ema_kimg = 1                       # Update G_ema every tick not seems to be control by cmd line like for snap
    kimg_per_tick= 1                   # Number of kimg per tick not seems to be control by cmd line like for snap

    # MODIFICATION FOR EACH METHOD:
    # -- Uchida's method -- #
    loss_kwargs.G = G                            # Generator full network architecture
    loss_kwargs.tools = Uchi_tools(device)       # Init the class methods for watermarking
    weight_name = 'synthesis.b32.conv0.weight'   # Weight name layer to be watermarked
    T = 32                                       # Watermark length (! CAPACITY !)
    watermark = torch.tensor(np.random.choice([0, 1], size=(T), p=[1. / 3, 2. / 3]))
    watermarking_dict_tmp = {'weight_name':weight_name,'watermark':watermark}
    watermarking_dict = loss_kwargs.tools.init(G, watermarking_dict_tmp, save=None)
    loss_kwargs.watermarking_dict = watermarking_dict
    #----------------------------------#
    
    '''