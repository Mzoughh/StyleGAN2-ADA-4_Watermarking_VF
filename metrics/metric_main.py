# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import json
import torch
import dnnlib

from . import metric_utils
from . import frechet_inception_distance
from . import kernel_inception_distance
from . import precision_recall
from . import perceptual_path_length
from . import inception_score
from . import brisque_score as brisque_score_module
from . import niqe_score as niqe_score_module

#------------------ W -------------#
# For Watermarking extraction
from NNWMethods.UCHI import Uchi_tools
from NNWMethods.T4G import T4G_tools
from NNWMethods.IPR import IPR_tools
from NNWMethods.T4G_plus import T4G_plus_tools
import copy
from torch_utils import misc
from torchvision.utils import save_image
#----------------------------------#

#----------------------------------------------------------------------------

_metric_dict = dict() # name => fn

def register_metric(fn):
    assert callable(fn)
    _metric_dict[fn.__name__] = fn
    return fn

def is_valid_metric(metric):
    return metric in _metric_dict

def list_valid_metrics():
    return list(_metric_dict.keys())

#----------------------------------------------------------------------------

def calc_metric(metric, **kwargs): # See metric_utils.MetricOptions for the full list of arguments.
    assert is_valid_metric(metric)
    opts = metric_utils.MetricOptions(**kwargs)

    # Calculate.
    start_time = time.time()
    results = _metric_dict[metric](opts)
    total_time = time.time() - start_time

    # Broadcast results.
    for key, value in list(results.items()):
        if opts.num_gpus > 1:
            value = torch.as_tensor(value, dtype=torch.float64, device=opts.device)
            torch.distributed.broadcast(tensor=value, src=0)
            value = float(value.cpu())
        results[key] = value

    # Decorate with metadata.
    return dnnlib.EasyDict(
        results         = dnnlib.EasyDict(results),
        metric          = metric,
        total_time      = total_time,
        total_time_str  = dnnlib.util.format_time(total_time),
        num_gpus        = opts.num_gpus,
    )

#----------------------------------------------------------------------------
#------------------ W -------------#
def report_metric(result_dict, run_dir=None, snapshot_pkl=None, attack_name='vanilla', parameter_attack_name='vanilla'):
    metric = result_dict['metric']
    assert is_valid_metric(metric)
    if run_dir is not None and snapshot_pkl is not None:
        snapshot_pkl = os.path.relpath(snapshot_pkl, run_dir)

    jsonl_line = json.dumps(dict(result_dict, snapshot_pkl=snapshot_pkl, timestamp=time.time()))
    print(jsonl_line)
    if run_dir is not None and os.path.isdir(run_dir):
        with open(os.path.join(run_dir, f'metric-{metric}-{attack_name}-{parameter_attack_name}.jsonl'), 'at') as f:
            f.write(jsonl_line + '\n')
#----------------------------------#

# -----------------W---------------#
# Adaptation of the run_G() function from losss.py to the metric.py file to generate new samples for watermark extraction
def run_G(G_mapping, G_synthesis, z, c, sync, style_mixing_prob, noise):
    with misc.ddp_sync(G_mapping, sync):
        ws = G_mapping(z, c)
        if style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
    with misc.ddp_sync(G_synthesis, sync):
        # --------------- W noise mode const --------------- #
        img = G_synthesis(ws, noise_mode=noise)
        #---------------------------------------------------- #
    return img, ws
# ----------------------------------#


#----------------------------------------------------------------------------
# Primary metrics.

@register_metric
def fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=50000)
    return dict(fid50k_full=fid)

@register_metric
def kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    kid = kernel_inception_distance.compute_kid(opts, max_real=1000000, num_gen=50000, num_subsets=100, max_subset_size=1000)
    return dict(kid50k_full=kid)

@register_metric
def pr50k3_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    precision, recall = precision_recall.compute_pr(opts, max_real=200000, num_gen=50000, nhood_size=3, row_batch_size=10000, col_batch_size=10000)
    return dict(pr50k3_full_precision=precision, pr50k3_full_recall=recall)

@register_metric
def ppl2_wend(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=50000, epsilon=1e-4, space='w', sampling='end', crop=False, batch_size=2)
    return dict(ppl2_wend=ppl)

@register_metric
def is50k(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    mean, std = inception_score.compute_is(opts, num_gen=50000, num_splits=10)
    return dict(is50k_mean=mean, is50k_std=std)

#------------------ W -------------#
@register_metric
def uchida_extraction(opts):
    if opts.watermarking_dict is not None:
        # model_device = next(opts.G.parameters()).device
        model_device =  opts.device
        print('model device', model_device)
        watermarking_dict = {
            k: (v.to(model_device) if torch.is_tensor(v) else v)
            for k, v in opts.watermarking_dict.items()
        }

        Generator = opts.G.to(model_device)

        tools = Uchi_tools(model_device)
        extraction, hamming_dist = tools.detection(Generator, watermarking_dict)
        extraction_r = torch.round(extraction)
        diff = (~torch.logical_xor((extraction_r).cpu()>0, watermarking_dict['watermark'].cpu()>0)) 
        bit_acc_avg = torch.sum(diff, dim=-1) / diff.shape[-1]
    else:
        print("No Uchida watermarking dictionary provided, skipping extraction metrics (0 by default).")
        extraction = 0
        hamming_dist = 0
        bit_acc_avg = 0
    return dict(uchida_bit_acc=float(bit_acc_avg), uchida_hamming_dist=float(hamming_dist))

@register_metric
def brisque_score(opts):
    mean_gen,mean_real,score = brisque_score_module.compute_brisque(
        opts, max_real=50000, num_gen=50000
    )
    return dict(brisque_gen=mean_gen,brisque_real=mean_real,brisque_abs=score)

@register_metric
def niqe_score(opts):
    mean_gen,mean_real,score = niqe_score_module.compute_niqe(
        opts, max_real=50000, num_gen=50000
    )
    return dict(niqe_gen=mean_gen, niqe_real=mean_real, niqe_abs=score)

@register_metric
def T4G_extraction(opts):
    if opts.watermarking_dict is not None:
        model_device =  opts.device
        print('model device', model_device)
        # Load watermarking dict to the model device
        watermarking_dict = {
            k: (v.to(model_device) if torch.is_tensor(v) else v)
            for k, v in opts.watermarking_dict.items()
        }
        
        G_mapping = opts.G.mapping.to(model_device)
        G_synthesis = opts.G.synthesis.to(model_device)

        tools = T4G_tools(model_device)  

        trigger_label=watermarking_dict['trigger_label']
        trigger_vector=watermarking_dict['trigger_vector']

        gen_img, _ =run_G(G_mapping, G_synthesis, trigger_vector, trigger_label, sync=True, style_mixing_prob=0, noise='const')
        bit_acc_avg, _ = tools.extraction(gen_img, watermarking_dict)

    else:
        print("No F4G watermarking dictionary provided, skipping extraction metrics (0 by default).")
        bit_acc_avg = 0
    return dict(f4g_bit_acc=float(bit_acc_avg))


@register_metric
def IPR_extraction(opts):

    if opts.watermarking_dict is not None:
        model_device =  opts.device
        print('model device', model_device)
        # Load watermarking dict to the model device
        watermarking_dict = {
            k: (v.to(model_device) if torch.is_tensor(v) else v)
            for k, v in opts.watermarking_dict.items()
        }
        
        G_mapping = opts.G.mapping.to(model_device)
        G_synthesis = opts.G.synthesis.to(model_device)

        tools = IPR_tools(model_device)  
        
        batch_size = 16

        latent_vector = torch.randn([batch_size, opts.G.z_dim], device=model_device)
        trigger_label= torch.zeros([batch_size, opts.G.c_dim], device=model_device)
        trigger_vector = tools.trigger_vector_modification(latent_vector, watermarking_dict)
        print('trigger vector generated for evaluation metrics')
        gen_img, _ =run_G(G_mapping, G_synthesis, latent_vector, trigger_label, sync=True, style_mixing_prob=0, noise='const')
        gen_imgs_from_trigger, _ = run_G(G_mapping, G_synthesis, trigger_vector, trigger_label, sync=True, style_mixing_prob=0, noise='const')
    
        print('generation done')
        SSIM, _ = tools.extraction(gen_img, gen_imgs_from_trigger, watermarking_dict)

    else:
        print("No IPR watermarking dictionary provided, skipping extraction metrics (0 by default).")
        SSIM= 0
    return dict(ipr_SSIM=float(SSIM))



@register_metric
def T4G_plus_extraction(opts):

    if opts.watermarking_dict is not None:
        model_device =  opts.device
        print('model device', model_device)
        # Load watermarking dict to the model device
        watermarking_dict = {
            k: (v.to(model_device) if torch.is_tensor(v) else v)
            for k, v in opts.watermarking_dict.items()
        }
        
        G_mapping = opts.G.mapping.to(model_device)
        G_synthesis = opts.G.synthesis.to(model_device)

        tools = T4G_plus_tools(model_device)  
        
        batch_size = 16

        latent_vector = torch.randn([batch_size, opts.G.z_dim], device=model_device)
        trigger_label= torch.zeros([batch_size, opts.G.c_dim], device=model_device)
        trigger_vector = tools.trigger_vector_modification(latent_vector, watermarking_dict)
        print('trigger vector generated for evaluation metrics')
        gen_img, _ =run_G(G_mapping, G_synthesis, latent_vector, trigger_label, sync=True, style_mixing_prob=0, noise='const')
        gen_imgs_from_trigger, _ = run_G(G_mapping, G_synthesis, trigger_vector, trigger_label, sync=True, style_mixing_prob=0, noise='const')
    
        loss_i = tools.perceptual_loss_for_imperceptibility(gen_img, gen_imgs_from_trigger, watermarking_dict)
        ssim = 1 - loss_i.item()

        bit_accs_avg_from_trigger, _ = tools.extraction(gen_imgs_from_trigger, watermarking_dict)
        bit_accs_avg_from_vanilla, _ = tools.extraction(gen_img, watermarking_dict)

    else:
        print("No T4G PLUS watermarking dictionary provided, skipping extraction metrics (0 by default).")
        ssim, bit_accs_avg = 0 , 0
    return dict(ipr_SSIM=float(ssim), bit_acc=bit_accs_avg_from_trigger, bit_acc_vanilla=bit_accs_avg_from_vanilla)
#----------------------------------#

#----------------------------------------------------------------------------
# Legacy metrics.

@register_metric
def fid50k(opts):
    opts.dataset_kwargs.update(max_size=None)
    fid = frechet_inception_distance.compute_fid(opts, max_real=50000, num_gen=50000)
    return dict(fid50k=fid)

@register_metric
def kid50k(opts):
    opts.dataset_kwargs.update(max_size=None)
    kid = kernel_inception_distance.compute_kid(opts, max_real=50000, num_gen=50000, num_subsets=100, max_subset_size=1000)
    return dict(kid50k=kid)

@register_metric
def pr50k3(opts):
    opts.dataset_kwargs.update(max_size=None)
    precision, recall = precision_recall.compute_pr(opts, max_real=50000, num_gen=50000, nhood_size=3, row_batch_size=10000, col_batch_size=10000)
    return dict(pr50k3_precision=precision, pr50k3_recall=recall)

@register_metric
def ppl_zfull(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=50000, epsilon=1e-4, space='z', sampling='full', crop=True, batch_size=2)
    return dict(ppl_zfull=ppl)

@register_metric
def ppl_wfull(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=50000, epsilon=1e-4, space='w', sampling='full', crop=True, batch_size=2)
    return dict(ppl_wfull=ppl)

@register_metric
def ppl_zend(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=50000, epsilon=1e-4, space='z', sampling='end', crop=True, batch_size=2)
    return dict(ppl_zend=ppl)

@register_metric
def ppl_wend(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=50000, epsilon=1e-4, space='w', sampling='end', crop=True, batch_size=2)
    return dict(ppl_wend=ppl)

#----------------------------------------------------------------------------
