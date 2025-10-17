import matplotlib.pyplot as plt
from Attacks.utils import *

# quantization
def quantize_tensor(x, num_bits):
    qmin = 0.
    qmax = 2. ** num_bits - 1.

    min_val, max_val = torch.min(x), torch.max(x)

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = torch.min(max_val - min_val).round()
    print(min_val, max_val, scale, initial_zero_point)
    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point
    zero_point = int(zero_point)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.byte()
    return {'tensor': q_x, 'scale': scale, 'zero_point': zero_point}


def dequantize_tensor(q_x):
    return q_x['scale'] * (q_x['tensor'].float() - q_x['zero_point'])

# ---------------  W -----------------#
def fake_quantization(tensor, num_bits):
    """
    Applies fake quantization to a given tensor.

    :param tensor: the input tensor to quantize
    :param num_bits: the number of bits to use for quantization (e.g., 8 for int8)
    :return: the quantized (and then dequantized) tensor
    """
    qmin = 0.
    qmax = 2. ** num_bits - 1.

    min_val = tensor.min()
    max_val = tensor.max()

    # Avoid division by zero when all values are identical
    if min_val == max_val:
        return tensor.clone()

    scale = (max_val - min_val) / qmax
    x_q = ((tensor - min_val) / scale).clamp(qmin, qmax).round()
    x_f_q = x_q * scale + min_val

    return x_f_q


def quantization(model, num_bits):
    """
    Applies fake quantization to a specific parameter of the StyleGAN2-ADA model.

    :param model: the PyTorch model to quantize
    :param num_bits: the number of bits for quantization (e.g., 8)
    :return: the quantized model
    """
    target_name = "synthesis.b32.conv0.weight"

    with torch.no_grad():
        for name, param in model.named_parameters():
            # if name == target_name:
            if param.requires_grad:
                print(f"Quantizing parameter: {name}")

                tensor = param.data
                before_min = tensor.min().item()
                before_max = tensor.max().item()

                quantized_tensor = fake_quantization(tensor, num_bits)
                param.data.copy_(quantized_tensor)

                after_min = param.data.min().item()
                after_max = param.data.max().item()

                print(f"Before quantization: min={before_min:.4f}, max={before_max:.4f}")
                print(f"After quantization:  min={after_min:.4f}, max={after_max:.4f}")
                break


    return model
#--------------------------------------- #
'''
    "quantization"
    param={"bits":2}
'''

