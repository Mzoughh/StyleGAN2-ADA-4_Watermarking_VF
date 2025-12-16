from utils import *
import torch

def adding_noise_global(net, power):
    power = power/100
    with torch.no_grad():
        for name, p in net.named_parameters():
            if "synthesis" in name:
                sigma = p.detach().flatten().std(unbiased=False)
                noise = torch.randn_like(p) * (power * sigma)
                p.add_(noise)
                print(f"Noise add on {name} | sigma={sigma.item():.4g} | noise_std={(power*sigma).item():.4g}")   
    return net

'''
    "noise"
    param={'name':["features.17.w"],"std":5}
    param={'name':"all","std":5}
'''