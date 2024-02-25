"""
Module of functions for doing reconstruction analysis, including
greedy stimulation algorithm simulation.
"""
import os
import sys

sys.path.insert(0,
 '/home/agogliet/gogliettino/projects/papers/raphe/repos/raphe-vs-periphery/')
import src.config as cfg
os.environ["OMP_NUM_THREADS"] = f"{cfg.N_THREADS}" 
os.environ["OPENBLAS_NUM_THREADS"] = f"{cfg.N_THREADS}" 
os.environ["MKL_NUM_THREADS"] = f"{cfg.N_THREADS}" 
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{cfg.N_THREADS}" 
os.environ["NUMEXPR_NUM_THREADS"] = f"{cfg.N_THREADS}" 
import scipy as sp
import numpy as np
import cv2
import src.lnp as lnp
import src.config as cfg
import src.fitting as fit
import src.io_util as io
import src.fitting as fit
from numpy import linalg
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import cvxpy as cp
from src.Dataset import Dataset
import src.models as models
import time
import src.losses as losses
from scipy import stats
from scipy.spatial import ConvexHull
from typing import Tuple, Union
import multiprocessing as mp

def get_random_stimulus(num_stimuli,stixel_size,
                        n_pixels_x,n_pixels_y,
                        seed=None,binary=True):
    """
    Generates random white noise stimuli tensors.
    Parameters:
        num_stimuli: number of stimuli
        stixel_size: pixels per stixel on the stimulus
        n_pixels_x: number of x pixels
        n_pixels_y: number of y pixels
        seed: random number gen seed
        binary: binary noise (only supported for now)
    Returns:
        Stimulus tensor of size num_stimuli, n_pixels_y, n_pixels_x
    """

    if not binary: raise ValueError('Non-binary stimuli not yet implemented.')

    # Grab the number of stixels from the stixel size.
    num_stixels_y = int(n_pixels_y / stixel_size)
    num_stixels_x = int(n_pixels_x / stixel_size)

    # Randomly generate frames.
    #if seed is not None: sp.random.seed(seed)
    if seed is not None: np.random.seed(seed)
    """
    random_stimuli = (sp.random.randn((num_stixels_y * num_stixels_x),
                                      num_stimuli) > 0).astype(float)
    """
    random_stimuli = (np.random.randn((num_stixels_y * num_stixels_x),
                                      num_stimuli) > 0).astype(float)
    random_stimuli -= 0.5

    return random_stimuli * (0.48 / 0.5)

def get_random_stimuli_jitter(n_stimuli,stixel_size,x_dim,y_dim,
                              n_pixels_x=640,n_pixels_y=320,seed=None,
                              factor=2):
    """
    Adds jitter to the random stimuli; useful for training linear 
    reconstruction filters. Calls the get_random_stimulus function under the 
    hood and applies a jitter.
    Parameters:
        n_stimuli: number of stimuli
        stixel_size: pixels per stixel
        x_dim: size of x dim
        y_dim: size of y dim
        n_pixels_x: number of x pixels
        n_pixels_y: number of y pixels
        seed: rng seed
        factor: amount of extra stimulus to get for jitter
    Return:
        Stimulus tensor of size num_stimuli, n_pixels_y, n_pixels_x
    """
    stimulus = get_random_stimulus(n_stimuli,stixel_size,
                                   n_pixels_x=int(n_pixels_x * factor),
                                   n_pixels_y=int(n_pixels_y * factor),
                                   binary=True,seed=seed)
    stimulus_reshape = []

    # Compute the field size and resample.
    field_x = int((n_pixels_x * factor) / stixel_size)
    field_y = int((n_pixels_y * factor) / stixel_size)

    for i in range(n_stimuli):
        tmp = cv2.resize(np.reshape(stimulus[:,i],(field_y,field_x)),
                                  dsize=(int(x_dim * factor),
                                         int(y_dim * factor)),
                                  interpolation=cv2.INTER_NEAREST)
        stimulus_reshape.append(tmp)

    # Now, for each stimulus, get a random window to simulate "jitter".
    stimulus_jitter = []
    np.random.seed(seed)

    for i in range(n_stimuli):
        random_x = np.random.randint(0,(int(x_dim * factor)))
        random_y = np.random.randint(0,(int(y_dim * factor)))

        if random_x + x_dim < x_dim * factor:
            x_window = np.arange(random_x,random_x + x_dim)
        else:
            x_window = np.arange(random_x - x_dim,random_x)

        if random_y + y_dim < y_dim * factor:
            y_window = np.arange(random_y,random_y + y_dim)
        else:
            y_window = np.arange(random_y - y_dim,random_y)

        # Chop the stimulus according to the new window
        tmp = stimulus_reshape[i][y_window,:][:,x_window]
        stimulus_jitter.append(tmp)

    return np.moveaxis(np.c_[[tmp for tmp in stimulus_jitter]],0,-1)

def mask_gaussian_filter(gaussian_filter,gaussian_popt,x_grid,y_grid,n_sigmas):
    """
    Returns a masked Gaussian filter setting anything outside of the 
    n_sigmas boundary to zero. Defines an ellipse helper function to mask
    pixels.

    TODO: document
    """

    def ellipse(x,y,gaussian_popt,n_sigmas):
        mu_x,mu_y,sigma_x,sigma_y,theta = gaussian_popt[1:]
        
        # Not exactly sure why but have to do this.
        theta_ = (theta * -1) % (2 * np.pi)
        a = ((x - mu_x) * np.cos(theta_)) + ((y - mu_y) * np.sin(theta_))
        b = ((x - mu_x) * np.sin(theta_)) - ((y - mu_y) * np.cos(theta_))
        
        return (a**2 / (sigma_x * n_sigmas)**2) + (b**2 / (sigma_y * n_sigmas)**2)
    
    # Set values outside of the ellipse to zero and inside to 1 and mask.
    mask = ellipse(x_grid,y_grid,gaussian_popt,n_sigmas)
    mask[np.where(mask <= 1)] = 1
    mask[np.where(mask > 1)] = 0 

    return gaussian_filter * mask

def get_gaussian_filters(gaussian_popt,
                         orig_n_pixels_x,
                         orig_n_pixels_y,
                         n_sigmas):
    """
    Evaluates 2D Gaussian on a stixel 4 grid. Assumes that the fits were 
    obtained on the spatial maps normalized in 0-1 range. Masks anything 
    outside of n_sigmas boundary to zero.

    Parameters:
        gaussian_popt: list of Gaussian params for each cell
        original_stixel: original stixel size. Used to to rescale params.
    Returns:
        list of evaluated Gaussian maps
    """

    # For each cell, evaluate the Gaussian fit at the resampled grid.
    x = np.linspace(0,orig_n_pixels_x-1,cfg.RESAMPLE_FIELD_X)
    y = np.linspace(0,orig_n_pixels_y-1,cfg.RESAMPLE_FIELD_Y)
    xx,yy = np.meshgrid(x,y)
    gaussian_filters = []

    for cc in range(len(gaussian_popt)):
        
        if gaussian_popt[cc] is None:
            gaussian_filters.append([])
            continue

        popt = np.asarray(gaussian_popt[cc].copy())
        gaussian_filter = fit.gaussian2d((xx,yy),*popt)
        gaussian_filter = mask_gaussian_filter(
                             gaussian_filter,
                             popt,
                             xx,yy,
                             n_sigmas
                        )
        gaussian_filters.append(gaussian_filter)

    return np.asarray(gaussian_filters) 

def learn_recon_filters_pooled(encoder,inds=None,sigmas=None,
                               stixels=cfg.STIXEL_SIZES):
    """
    TODO: document
    """
    n_cells,n_pixels_y,n_pixels_x = encoder.shape
    gaussian_filters_matrix = np.reshape(
                         encoder,
                        (n_cells,n_pixels_x * n_pixels_y)
                        )
    n_stixels = stixels.shape[0]
    stimulus = np.zeros((n_pixels_y,
                         n_pixels_x,
                         cfg.N_TRAIN_STIMULI * n_stixels)
               )
    print('generating white noise frames ... ')
    
    for ss,stixel in enumerate(stixels):
        tmp = get_random_stimuli_jitter(
                        cfg.N_TRAIN_STIMULI,stixel,
                        n_pixels_x,n_pixels_y,
                        seed=cfg.TRAIN_SEED
                    )
        stimulus[
            ...,ss*cfg.N_TRAIN_STIMULI:cfg.N_TRAIN_STIMULI*(ss+1)
            ] = tmp
        
        del tmp

    print('done')
    
    stimulus_vector = np.reshape(stimulus,(n_pixels_x * n_pixels_y,
                                cfg.N_TRAIN_STIMULI*n_stixels))
    G = np.transpose(gaussian_filters_matrix@stimulus_vector)
    np.random.seed(cfg.TRAIN_SEED)
    
    if inds is None or sigmas is None:
        R = np.maximum(G,0)
        #R = np.floor(R) + np.random.binomial(1,R - np.floor(R))
    else: 
        for i in range(len(inds)):
            noise = np.random.randn(*G[:,inds[i]].shape) * sigmas[i]
            G[:,inds[i]] = np.maximum(G[:,inds[i]] + noise,0)
        
        R = np.floor(G) + np.random.binomial(1,G - np.floor(G))
    
    S = stimulus_vector.T
    n_cells = gaussian_filters_matrix.shape[0]
    print('learning linear filters ... ')
    W = linalg.inv(R.T@R)@R.T@S
    print('done') 
    
    return W

def learn_recon_filters_ns_torch(encoder,device,
                                 train_inds,
                                 inds=None,sigmas=None):
    """
    TODO: document
    """
    n_cells,n_pixels = encoder.shape
    encoder = torch.from_numpy(encoder).to(device)
    gaussian_filters_matrix = torch.reshape(
                                        encoder,
                                        (n_cells,n_pixels)
                            )
    stimulus = io.get_imagenet_stimuli(cfg.IMAGENET_TRAIN_INDS,dtype=np.float32)
    _,n_pixels_y,n_pixels_x,_ = stimulus.shape
    
    # Check if need to resize: assumes constant aspect ratio.
    if n_pixels_y != cfg.RESAMPLE_FIELD_Y:
        interp = cv2.INTER_CUBIC

        # For downsammpling, use inter area. 
        if n_pixels_y > cfg.RESAMPLE_FIELD_Y:
            interp = cv2.INTER_AREA

        stimulus = np.asarray([cv2.resize(stimulus[i,...],
                              dsize=(cfg.RESAMPLE_FIELD_X,cfg.RESAMPLE_FIELD_Y),
                              interpolation=interp)
                              for i in range(stimulus.shape[0])
                   ])
        
    stimulus = stimulus / 255
    stimulus -= np.mean(stimulus)
    stimulus = torch.from_numpy(stimulus).to(device)
    stimulus = stimulus[...,0] # doesn't matter since grayscale
    n_stimuli,n_pixels_y,n_pixels_x = stimulus.shape
    stimulus_vector = torch.reshape(
                                stimulus,
                                (n_stimuli,n_pixels_y * n_pixels_x)
                      )
    
    with torch.no_grad():
        G = torch.t(
                torch.matmul(
                    gaussian_filters_matrix,torch.t(stimulus_vector)
                )
        )
        GenGPU = torch.Generator(device=device)
        GenGPU.manual_seed(cfg.TRAIN_SEED)
        
        if inds is None or sigmas is None:
            R = torch.clamp(G,min=0)
            """
            R = torch.floor(R) + torch.bernoulli(
                                        R - torch.floor(R),
                                        generator=GenGPU
                                )
            """
        else:
            for i in range(len(inds)):
                noise = (torch.randn(G[:,inds[i]].shape) 
                        * sigmas[i]).to(torch.float32).to(device)
                G[:,inds[i]] = torch.clamp(G[:,inds[i]] + noise,min=0)
            
            R = torch.floor(G) + torch.bernoulli(
                                        G - torch.floor(G),
                                        generator=GenGPU
                                 )
        
        W = torch.linalg.solve(
                    torch.matmul(torch.t(R),R),
                    torch.matmul(torch.t(R),stimulus_vector)
            ).detach().cpu().numpy()
    
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    
    return W

def rescale_encoder_to_decoder(encoder,decoder):
    """
    Matches the peaks of the encoder and decoder and scales the encoder so that
    the peak value between the two filters match.
    """
    encoder_rescale = encoder.copy()

    for i in range(decoder.shape[0]):

        if (np.abs(encoder_rescale[i,:].max()) > 
            np.abs(encoder_rescale[i,:].min())):
            tmp_decoder = np.max(decoder[i,:])
            tmp_encoder = np.max(encoder_rescale[i,:])
        else:
            tmp_decoder = np.min(decoder[i,:])
            tmp_encoder = np.min(encoder_rescale[i,:])
    
        encoder_rescale[i,:] *= tmp_decoder /  tmp_encoder
    
    return encoder_rescale

def rescale_encoder_to_decoder_median(encoder,decoder):
    """
    TODO: document.
    """
    encoder_rescale = encoder.copy()
    scale_dict = {'on': [],'off': []}

    for i in range(decoder.shape[0]):

        if (np.abs(encoder_rescale[i,:].max()) > 
            np.abs(encoder_rescale[i,:].min())):
            tmp_decoder = np.max(decoder[i,:])
            tmp_encoder = np.max(encoder_rescale[i,:])
            key = 'on'
        else:
            tmp_decoder = np.min(decoder[i,:])
            tmp_encoder = np.min(encoder_rescale[i,:])
            key = 'off'

        scale_dict[key].append(tmp_decoder / tmp_encoder)
    
    # Take the median and get the scale.
    for key in scale_dict:
        scale_dict[key] = np.median(scale_dict[key])
    
    # Take another pass over the data and rescale based on median.
    for i in range(decoder.shape[0]):

        if (np.abs(encoder_rescale[i,:].max()) > 
            np.abs(encoder_rescale[i,:].min())):
            key = 'on'
        else:
            key = 'off'
        
        encoder_rescale[i,...] *= scale_dict[key]
    
    return encoder_rescale

def get_encoding_filters_pooled(gaussian_filters_tensor,
                                inds,firing_rates,
                                stixels=cfg.STIXEL_SIZES):
    """
    # TODO: document
    """
    n_cells,n_pixels_y,n_pixels_x = gaussian_filters_tensor.shape
    gaussian_filters_matrix = np.reshape(
                    gaussian_filters_tensor,
                    (n_cells,n_pixels_x * n_pixels_y)
                )
    n_stixels = stixels.shape[0]
    stimulus = np.zeros((n_pixels_y,n_pixels_x,
                         cfg.N_TRAIN_STIMULI * n_stixels)
               )
    print('generating white noise frames ... ')
    
    for ss,stixel in enumerate(stixels):
        tmp = get_random_stimuli_jitter(
                        cfg.N_TRAIN_STIMULI,stixel,
                        n_pixels_x,n_pixels_y,
                        seed=cfg.PRE_TRAIN_SEED
                    )
        stimulus[...,ss*cfg.N_TRAIN_STIMULI:cfg.N_TRAIN_STIMULI*(ss+1)] = tmp
        
        del tmp
    
    print('done.')
    stimulus_vector = np.reshape(
                        stimulus,
                        (n_pixels_x * n_pixels_y,cfg.N_TRAIN_STIMULI*n_stixels)
                    )
    
    print('scaling encoder ... ')
    R = np.maximum(gaussian_filters_matrix@stimulus_vector,0)
    
    for i in range(len(inds)):
        mean_firing_rate = np.mean(R[inds[i],:])
        scale = firing_rates[i] / mean_firing_rate
        gaussian_filters_matrix[inds[i],...] *= scale
        
    print('done')

    return gaussian_filters_matrix

def get_encoding_filters_ns_torch(gaussian_filters_tensor,
                                  inds,firing_rates,device):
    """
    TODO: document
    """
    n_cells,n_pixels_y,n_pixels_x = gaussian_filters_tensor.shape
    gaussian_filters_tensor = torch.from_numpy(
                                gaussian_filters_tensor
                                ).to(device)
    gaussian_filters_matrix = torch.reshape(
                                    gaussian_filters_tensor,
                                    (n_cells,n_pixels_x * n_pixels_y)
                              )
    """
    stimulus = io.get_raw_movie(os.path.join(cfg.IMAGENET_PARENT,
                                cfg.IMAGENET_PRETRAIN),
                                cfg.N_TRAIN_STIMULI
                        ).astype(np.float32)
    """
    stimulus = io.get_imagenet_stimuli(
                        cfg.IMAGENET_PRETRAIN_INDS,
                        dtype=np.float32
                )
    stimulus = torch.from_numpy(stimulus).to(device)
    n = stimulus.shape[0]
    
    with torch.no_grad():
        stimulus = stimulus / 255
        stimulus -= torch.mean(stimulus)
        stimulus = stimulus[...,0]
        
    stimulus_vector = torch.t(
                            torch.reshape(
                                stimulus,
                                (n,n_pixels_x * n_pixels_y)
                            )
                    )
    
    with torch.no_grad():
        R = torch.clamp(
                    torch.matmul(gaussian_filters_matrix,stimulus_vector),
                    min=0
        )
        
        for i in range(len(inds)):
            mean_firing_rate = torch.mean(R[inds[i],:])
            scale = firing_rates[i] / mean_firing_rate
            gaussian_filters_matrix[inds[i],...] *= scale
        
    gaussian_filters_matrix = gaussian_filters_matrix.detach().cpu().numpy()
    
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    
    return gaussian_filters_matrix

def center_filters(filter_tensor,n_pixels_y,n_pixels_x):
    """
    Centers either the encoding or decoding filters. Takes the sum over cells,
    computes the median x and y pixel and sets it to the ~center of the 
    visual field

    Parameters:
        filter_tensor: tensor of size n_cells,n_pixels_y,n_pixels_x
        n_pixels_y: number of y pixels
        n_pixels_x: number of x pixels
    Returns:
        a centered filter tensor.
    """

    # Take the sum over cells and find nonzero pixels.
    nonzero_stixels = np.argwhere(np.sum(np.abs(filter_tensor),axis=0) != 0)
    med_y = np.median(nonzero_stixels[:,0])
    med_x = np.median(nonzero_stixels[:,1])

    # Compute the difference from midpoint and roll it.
    diff_y = int(n_pixels_y/2 - med_y)
    diff_x = int(n_pixels_x/2 - med_x)

    return np.roll(filter_tensor,[diff_y,diff_x],axis=(1,2))

def uncenter_filters(original_tensor,centered_tensor,n_pixels_y,n_pixels_x):
    """
    TODO: document
    """

    # Take the sum over cells and find nonzero pixels.
    nonzero_stixels = np.argwhere(np.sum(np.abs(original_tensor),axis=0) != 0)
    med_y = np.median(nonzero_stixels[:,0])
    med_x = np.median(nonzero_stixels[:,1])

    # Compute the difference from midpoint and roll it.
    diff_y = int(n_pixels_y/2 - med_y)
    diff_x = int(n_pixels_x/2 - med_x)

    return np.roll(centered_tensor,[-diff_y,-diff_x],axis=(1,2))

def center_filters_by_reference_filters(reference_tensor,tensor_to_center,
                                       n_pixels_y,n_pixels_x):
       
    nonzero_stixels = np.argwhere(np.sum(np.abs(reference_tensor),axis=0) != 0)
    med_y = np.median(nonzero_stixels[:,0])
    med_x = np.median(nonzero_stixels[:,1])

    # Compute the difference from midpoint and roll it.
    diff_y = int(n_pixels_y/2 - med_y)
    diff_x = int(n_pixels_x/2 - med_x)
    
    return np.roll(tensor_to_center,[diff_y,diff_x],axis=(1,2))

def get_filter_dict(cellids,gaussian_popt,
                    orig_n_pixels_x,
                    orig_n_pixels_y,
                    n_sigmas):
    """
    Function to learn reconstruction filters from LNP model. Simulates
    responses to white noise stimuli using the LNP model and applies least
    squares regression to learn reconstruction filters.

    TODO: DOCUMENT
    """

    # Get the Gaussian maps and get the cells that have nontrivial filters.
    gaussian_filters = get_gaussian_filters(
                            gaussian_popt,
                            orig_n_pixels_x,
                            orig_n_pixels_y,
                            n_sigmas)
    cells_reconstruction = [cellids[i] for i in range(len(gaussian_filters))
                            if gaussian_filters[i] != []]
    gaussian_filters_tensor = []

    for gaussian_filter in gaussian_filters:
        
        if gaussian_filter == []:
            continue

        gaussian_filters_tensor.append(gaussian_filter)

    gaussian_filters_tensor = np.asarray(gaussian_filters_tensor)
    gaussian_filters_tensor = normalize_filters(gaussian_filters_tensor)

    # Center the Gaussian filters to the middle of visual field.
    n_cells,n_pixels_y,n_pixels_x = gaussian_filters_tensor.shape
    gaussian_filters_tensor = center_filters(
                                    gaussian_filters_tensor,
                                    n_pixels_y,n_pixels_x
                              )

    # Learn encoding and decoding filters for each stixel size. 
    filter_dict = dict()
    filter_dict['encoding_filters'] = dict()
    filter_dict['decoding_filters'] = dict()
    filter_dict['cells_reconstruction'] = cells_reconstruction

    for stixel in cfg.STIXEL_SIZES:

        print(f'learning filters for stixel {stixel}')

        # Scale the Gaussian filters to get the encoder and learn decoder.
        encoder = get_encoding_filters(
                                stixel,
                                gaussian_filters_tensor
                            )
        decoder = learn_recon_filters(
                                stixel,
                                encoder.reshape((n_cells,
                                                 n_pixels_y,
                                                 n_pixels_x)),
                            )
        filter_dict['encoding_filters']["%s"%str(stixel)] = encoder
        filter_dict['decoding_filters']["%s"%str(stixel)] = decoder

    return filter_dict

def get_filter_dict_pooled(cellids,gaussian_popt,
                    orig_n_pixels_x,
                    orig_n_pixels_y,
                    engine,
                    n_sigmas,do_mask,
                    inds,
                    firing_rates,
                    sigmas,
                    device=None):
    """
    TODO: document
    """
    assert engine in ['numpy','torch'], 'unknown compute engine'
    assert (engine in 'torch' and device is not None) or engine in ['numpy']
    
    # Get the Gaussian maps and get the cells that have nontrivial filters.
    gaussian_filters = get_gaussian_filters(
                            gaussian_popt,
                            orig_n_pixels_x,
                            orig_n_pixels_y,
                            n_sigmas)
    cells_reconstruction = [cellids[i] for i in range(len(gaussian_filters))
                            if gaussian_filters[i] != []]
    gaussian_filters_tensor = []

    for gaussian_filter in gaussian_filters:
        
        if gaussian_filter == []:
            continue

        gaussian_filters_tensor.append(gaussian_filter)

    gaussian_filters_tensor = np.asarray(gaussian_filters_tensor)
    
    gaussian_filters_tensor = normalize_filters(gaussian_filters_tensor)

    # Center the Gaussian filters to the middle of visual field.
    n_cells,n_pixels_y,n_pixels_x = gaussian_filters_tensor.shape

    # TODO: for cropping expt, center after.
    if not do_mask:

        # If parasol and midget, center with respect to only parasols.
        if len(inds) == 1:
            gaussian_filters_tensor = center_filters(
                                            gaussian_filters_tensor,
                                            n_pixels_y,n_pixels_x
                                    )
        else:
            gaussian_filters_tensor_1 = center_filters(
                                            gaussian_filters_tensor[inds[0],...],
                                            n_pixels_y,n_pixels_x
            )
            gaussian_filters_tensor_2 = center_filters_by_reference_filters(
                                            gaussian_filters_tensor_1,
                                            gaussian_filters_tensor[inds[1],...],
                                            n_pixels_y,n_pixels_x
            )
            gaussian_filters_tensor = np.r_[gaussian_filters_tensor_1,
                                            gaussian_filters_tensor_2
                                    ]
        
    # Learn encoding and decoding filters for each stixel size. 
    filter_dict = dict()
    filter_dict['encoding_filters'] = dict()
    filter_dict['decoding_filters'] = dict()
    filter_dict['cells_reconstruction'] = cells_reconstruction

    # Scale the Gaussian filters to get the encoder and learn decoder.
    print('learning pooled white noise filters ... ')
    
    if engine in ['numpy']:
        encoder = get_encoding_filters_pooled(
                                gaussian_filters_tensor,
                                inds,firing_rates
                            )
        decoder = learn_recon_filters_pooled(
                                encoder.reshape((n_cells,
                                                    n_pixels_y,
                                                    n_pixels_x)),
                                inds=inds,sigmas=sigmas
                            )
    else:
        encoder = get_encoding_filters_pooled_torch(
                                        gaussian_filters_tensor.astype(np.float32),
                                        device
                  )
        decoder = learn_recon_filters_pooled_torch(
                                encoder.reshape((n_cells,
                                                 n_pixels_y,
                                                 n_pixels_x)
                                        ).astype(np.float32),
                                device
                  )
    
    # TODO: test out crop.
    if do_mask:
        x = np.linspace(0,orig_n_pixels_x-1,cfg.RESAMPLE_FIELD_X)
        y = np.linspace(0,orig_n_pixels_y-1,cfg.RESAMPLE_FIELD_Y)
        xx,yy = np.meshgrid(x,y)
        decoder = np.asarray([mask_gaussian_filter(
                            decoder[i,...].reshape(n_pixels_y,n_pixels_x),
                            gaussian_popt[i],
                            xx,yy,
                            n_sigmas,
                    )
                    for i in range(decoder.shape[0])])
        encoder = center_filters(encoder.reshape(n_cells,n_pixels_y,n_pixels_x),
                                 n_pixels_y,n_pixels_x)
        decoder = center_filters(decoder,n_pixels_y,n_pixels_x)

        # Reshape into a matrix.
        encoder = encoder.reshape(n_cells,n_pixels_x * n_pixels_y)
        decoder = decoder.reshape(n_cells,n_pixels_x * n_pixels_y)

    for stixel in cfg.STIXEL_SIZES:

        filter_dict['encoding_filters']["%s"%str(stixel)] = encoder
        filter_dict['decoding_filters']["%s"%str(stixel)] = decoder

    return filter_dict

def normalize_filters(gaussian_filters):
    """
    L2-normalizes the Gaussian filters among nonzero 
    elements.
    TODO: DOCUMENT
    """
    n_cells,n_pixels_y,n_pixels_x = gaussian_filters.shape
    gaussian_filters = np.reshape(gaussian_filters,(n_cells,
                                  n_pixels_x * n_pixels_y))
    normalized_filters = []

    for cc in range(gaussian_filters.shape[0]):
        sig_stix = np.argwhere(gaussian_filters[cc,:] != 0).flatten()
        normalized_filter = gaussian_filters[cc,:].copy()
        normalized_filter[sig_stix] /= linalg.norm(normalized_filter[sig_stix])
        normalized_filters.append(normalized_filter)
    
    normalized_filters = np.asarray(normalized_filters)

    return normalized_filters.reshape(n_cells,n_pixels_y,n_pixels_x)

def get_responses_lnp(gaussian_filters_tensor,stimulus,nl_popt):
    """
    Gets the responses from the LNP model.
    TODO: document
    """

    # Vectorize stimuli and filters 
    n_cells,n_pixels_y,n_pixels_x = gaussian_filters_tensor.shape
    gaussian_filters = np.reshape(
                            gaussian_filters_tensor,
                            (n_cells,n_pixels_y * n_pixels_x)
                        )
    n_stimuli = stimulus.shape[0]
    stimulus_vector = np.reshape(stimulus,(n_stimuli,n_pixels_x * n_pixels_y))

    # Compute generator signals and look up the FR from nl.
    generators = gaussian_filters@stimulus_vector.T
    responses = []

    for cc in range(len(nl_popt)):
        responses.append(fit.nl(generators[cc],*nl_popt[cc]))
    
    return np.asarray(responses) * (cfg.N_MS_POST_FLASH / 1000)

def get_filter_dict_ns(cellids,
                       gaussian_popt,
                       nl_popt,
                       orig_n_pixels_x,
                       orig_n_pixels_y,
                       engine,
                       n_sigmas,
                       inds,
                       firing_rates,
                       sigmas,
                       device=None):
    """
    Learns reconstruction filters for natural scenes using the LNP model.

    TODO: DOCUMENT
    """
    assert engine in ['numpy','torch'], 'unknown compute engine'
    assert (engine in 'torch' and device is not None) or engine in ['numpy']

    # Get the Gaussian filters and normalize them.
    gaussian_filters = get_gaussian_filters(
                            gaussian_popt,
                            orig_n_pixels_x,
                            orig_n_pixels_y,
                            n_sigmas
                        )
    '''
    cells_reconstruction = [cellids[i] for i in range(len(gaussian_filters))
                            if gaussian_filters[i] != [] and nl_popt[i]
                            is not None]
    '''
    cells_reconstruction = [cellids[i] for i in range(len(gaussian_filters))
                            if gaussian_filters[i] != []]
    nl_popt = [i for i in nl_popt if i is not None]
    gaussian_filters_tensor = []

    for gaussian_filter in gaussian_filters:
        
        if gaussian_filter == []:
            continue

        gaussian_filters_tensor.append(gaussian_filter)

    gaussian_filters_tensor = np.asarray(gaussian_filters_tensor)
    gaussian_filters_tensor = normalize_filters(gaussian_filters_tensor)

    # Center the Gaussian filters to the middle of visual field.
    _,n_pixels_y,n_pixels_x = gaussian_filters_tensor.shape
    
    # If parasol and midget, center with respect to only parasols.
    if len(inds) == 1:
        gaussian_filters_tensor = center_filters(
                                        gaussian_filters_tensor,
                                        n_pixels_y,n_pixels_x
                                )
    else:
        gaussian_filters_tensor_1 = center_filters(
                                        gaussian_filters_tensor[inds[0],...],
                                        n_pixels_y,n_pixels_x
        )
        gaussian_filters_tensor_2 = center_filters_by_reference_filters(
                                        gaussian_filters_tensor_1,
                                        gaussian_filters_tensor[inds[1],...],
                                        n_pixels_y,n_pixels_x
        )
        gaussian_filters_tensor = np.r_[gaussian_filters_tensor_1,
                                        gaussian_filters_tensor_2
                                  ]
    
    if engine in ['numpy']:
        encoding_filters = get_encoding_filters_ns(gaussian_filters_tensor)
        decoder = learn_recon_filters_ns(encoding_filters)
    else:
        encoding_filters = get_encoding_filters_ns_torch(
                                    gaussian_filters_tensor.astype(np.float32),
                                    inds,firing_rates,device
                            )
        decoder = learn_recon_filters_ns_torch(encoding_filters.astype(np.float32),
                                               device=device,
                                               train_inds=cfg.IMAGENET_TRAIN_INDS,
                                               inds=inds,sigmas=sigmas)

    filter_dict = dict()
    filter_dict['encoding_filters'] = encoding_filters
    filter_dict['cells_reconstruction'] = cells_reconstruction
    filter_dict['nl_popt'] = nl_popt
    filter_dict['decoding_filters'] = decoder

    return filter_dict

def get_sig_stixels_encoder(encoder,excluded_cell_inds=None):
    """
    Gets sig stixels by taking union of all nonzero encoder pixels.

    TODO: doc
    """
    sig_stixels_all = set()

    for i in range(encoder.shape[0]):
        
        if excluded_cell_inds is not None and i in excluded_cell_inds:
            continue

        sig_stixels = np.nonzero(encoder[i,:].ravel())[0]

        for s in sig_stixels:
            sig_stixels_all.add(s)

    sig_stixels_all = np.asarray(sorted(list(sig_stixels_all)))

    return sig_stixels_all

def get_encoder_mask(encoder):
    """
    Gets sig stixels by taking union of all nonzero encoder pixels. Returns
    grid indexing.

    TODO: doc
    """
    _,n_pixels_y,n_pixels_x = encoder.shape
    mask = np.zeros((n_pixels_y,n_pixels_x))

    mask[np.where(np.sum(np.abs(encoder),axis=0) != 0)] = 1

    return mask[None,None,...]

def get_dictionary_variance(dictionary_matrix,decoder):
    """
    Computes variance of the dictionary

    TODO: document.
    """
    decoder_norm_sq = np.sum(decoder**2,axis=1)
    var = np.matmul(dictionary_matrix * (1 - dictionary_matrix),
                    decoder_norm_sq
                )

    return var

def get_optimal_recon(decoder,stimulus,mask,discretize=True):
    """
    Gets the optimal reconstruction by solving for the optimal responses under
    squared loss. Discretizes the spike vector if indicated.

    TODO: document
    """
    print('DEBUG')
    optimal_decoded_stimuli = []
    n_cells = decoder.shape[0]
    
    for i in range(stimulus.shape[1]):
        x = cp.Variable(n_cells)
        objective = cp.Minimize(
                        cp.sum_squares(
                            (decoder.T * mask[:,None]) @x - 
                            (stimulus[:,i].ravel() * mask)
                        )
        )
        
        constraints = [x >= 0] # non-negative firing rate
        prob = cp.Problem(objective,constraints)
        prob.solve();

        if discretize:
            x.value = np.round(x.value)
        
        optimal_decoded_stimuli.append(decoder.T@x.value)
    
    return np.stack(optimal_decoded_stimuli,axis=1)

def n_mse(stimuli,decoded_stimuli,sig_stixels):
    """
    Computes normalized squared error between the stimulus and decoded stimulus
    within the set of nonzero pixels.

    TODO: document
    """
    return (linalg.norm(
                stimuli[sig_stixels,:]
                - decoded_stimuli[sig_stixels,:],
                axis=0)**2 / linalg.norm(stimuli[sig_stixels,:],axis=0)**2
            )

def get_frac_incorrect_pix(stimuli,decoded_stimuli,sig_stixels):
    """
    Computes the fraction of incorrect pixels between the reconstruction 
    and the original stimulus within a specified set of pixels.

    TODO: document
    """
    return (np.argwhere(np.sign(stimuli[sig_stixels]) !=
                np.sign(decoded_stimuli[sig_stixels])).shape[0] / 
                sig_stixels.shape[0])

def get_sign_diff_map(stimuli,decoded_stimuli,sig_stixels):
    """
    Get the sign difference map.

    TODO: document
    """
    sign_map = np.zeros(stimuli.shape)
    diff_inds = np.argwhere(np.sign(stimuli[sig_stixels]) !=
                np.sign(decoded_stimuli[sig_stixels])).flatten()
    sign_map[sig_stixels[diff_inds]] = -1
    same_inds = np.argwhere(np.sign(stimuli[sig_stixels]) ==
                np.sign(decoded_stimuli[sig_stixels])).flatten()
    sign_map[sig_stixels[same_inds]] = 1

    return sign_map

def train_cnn(train_data,test_data,mask,fnameout,
             n_epochs=cfg.N_EPOCHS,
             batch_size=cfg.BATCH_SIZE,
             learning_rate=cfg.LEARNING_RATE,
             checkpoint_epoch=None):
    '''
    Train the CNN denoiser.
    '''
    device = torch.device(cfg.GPU if torch.cuda.is_available() else "cpu")
    train_loader = torch.utils.data.DataLoader(
                                train_data,
                                batch_size=batch_size,
                                shuffle=True
                    )
    test_loader = torch.utils.data.DataLoader(
                                test_data,
                                #batch_size=test_data.__len__(),
                                batch_size=batch_size,
                                shuffle=True
                   )

    # Define the loss function as MSE loss
    criterion = losses.MaskedMSELoss(mask)
    Model = models.CAE()
    start_epoch = 0
    
    # Load in the checkpoint if given.
    if checkpoint_epoch is not None:
        fnamein = fnameout + f'_epoch_{checkpoint_epoch}.tar'
        model_dict = torch.load(fnamein)
        Model.load_state_dict(model_dict['model_state_dict'])
        start_epoch = model_dict['epoch'] + 1
        Model.train()

    # Define the optimizer with a learning rate
    optimizer = optim.Adam(Model.parameters(), lr=learning_rate)
    Model = Model.to(device)

    # Train the model and cache the losses.
    start_time = time.time()

    # Train model.
    start_time = time.time()

    for epoch in np.arange(start_epoch,n_epochs):
        cost_cache = {'test': [], 'train': []}

        for i, (X, Y) in enumerate(train_loader,0):
                
            X = X.to(device)
            Y = Y.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward propogation.
            Y_hat = Model(X)

            # Calculate the loss only on the non-zero pixels.
            loss = criterion(Y_hat, Y)

            # Backpropogation + update parameters
            loss.backward()
            optimizer.step()

            # Print statistics
            cost = loss.item()
            cost_cache['train'].append(cost)

            if i % 100 == 0:
                print('Epoch: ' + str(epoch) + ", Iteration: " + str(i) 
                        + ", training cost = " + str(cost))
            
        # Also get test cost.
        Model.eval()

        for _, (X_test,Y_test) in enumerate(test_loader,0):
            X_test = X_test.to(device)
            Y_test = Y_test.to(device)
            Y_hat_test = Model(X_test)
            loss = criterion(Y_hat_test,Y_test)
            test_cost = loss.item()
            cost_cache['test'].append(test_cost)
        
        Model.train()
            
        # Write results to the dict after each epoch.
        torch.save({
            'model_state_dict': Model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'cost_cache': cost_cache,
            'epoch': epoch,
            'learning_rate': learning_rate,
            'batch_size': batch_size
        },
        fnameout + f'_epoch_{epoch}.tar')
        
    print("--- %s seconds ---" % (time.time() - start_time))

    return None

def get_responses_post_hoc(dictionary_matrix,element_log):
    """
    Gets the responses from a group of cells after the application of a 
    sequence of stimuli in the greedy algorithm.

    Assumes a standard stimulation dt and refractory period.
    """
    responses = np.zeros(dictionary_matrix.shape[1])
    t = 0
    refractory_log = np.zeros(responses.shape[0])

    for element in element_log:

        if element == dictionary_matrix.shape[0]:
            t += cfg.STIMULATION_DT
            continue
        
        mask = (refractory_log < t).astype(int)
        responses += dictionary_matrix[element,:] * mask

        # Advance the stimulation clock and update the refractory log.
        refractory_log[np.where(
                        dictionary_matrix[element,:] > cfg.MIN_SPIKE_PROB
                        )] += t + cfg.REFRACTORY_PERIOD
        t += cfg.STIMULATION_DT
    
    return responses

def get_percept_post_hoc(dictionary_matrix,decoder,element_log):
    """
    Estimates perception post hoc based on decoder and chosen elements.

    Assumes a standard stimulation dt and refractory period.

    TODO: document
    """
    responses = get_responses_post_hoc(dictionary_matrix,element_log)

    return decoder.T@responses

def get_hist_bin(edges,point):
    """
    TODO: document
    """
    if point < edges[0]: # shouldn't ever happen based on how edges are made.
        print('ERR')
        return 0

    for i in range(edges.shape[0]-1):
        if point >= edges[i] and point < edges[i+1]:
            return i

    return i

def get_responses_post_hoc_w_hist(hist_dict,cell_centers,element_unraveled_log,
                                 elec_coords,expected_value,seed=11424):
    """
    TODO: document
    """
    probs = hist_dict['probs']
    distance_edges = hist_dict['distance_edges']
    amplitude_edges = hist_dict['amplitude_edges']
    np.random.seed(seed)
    #responses = np.zeros(cell_centers.shape[0])
    responses = np.zeros((cell_centers.shape[0],element_unraveled_log.shape[0]
                ))

    for i in range(element_unraveled_log.shape[0]):
        elec_ind = element_unraveled_log[i,0]
        amplitude = cfg.AMPLITUDES[element_unraveled_log[i,1]]
        distances = linalg.norm(cell_centers - elec_coords[elec_ind,:][None,:],
                                axis=1
                    )
        a_bin = get_hist_bin(amplitude_edges,amplitude)

        for j in range(cell_centers.shape[0]):
            d_bin = get_hist_bin(distance_edges,distances[j])
            
            if expected_value:
                responses[j,i] += probs[d_bin,a_bin]
            else:
                responses[j,i] += np.random.binomial(1,probs[d_bin,a_bin])

    return responses

def _get_responses(i,element_unraveled_log,cell_centers,
                    elec_coords,probs,distance_edges,amplitude_edges,
                    expected_value):
    """
    helper function for multiprocessed histogrammed response determination
    don't use this function.
    """
    responses = np.zeros(cell_centers.shape[0])
    elec_ind = element_unraveled_log[i,0]
    amplitude = cfg.AMPLITUDES[element_unraveled_log[i,1]]
    distances = linalg.norm(cell_centers - elec_coords[elec_ind,:][None,:],
                            axis=1
                )
    a_bin = get_hist_bin(amplitude_edges,amplitude)

    for j in range(cell_centers.shape[0]):
        d_bin = get_hist_bin(distance_edges,distances[j])
        
        if expected_value:
            responses[j] += probs[d_bin,a_bin]
        else:
            responses[j] += np.random.binomial(1,probs[d_bin,a_bin])
        
    return responses

def get_responses_post_hoc_w_hist_fast(hist_dict,cell_centers,
                                       element_unraveled_log,
                                       elec_coords,expected_value,seed=11424):
    probs = hist_dict['probs']
    distance_edges = hist_dict['distance_edges']
    amplitude_edges = hist_dict['amplitude_edges']
    np.random.seed(seed)
   
    with mp.Pool(cfg.MP_N_THREADS) as pool:
        responses = pool.starmap(_get_responses,
                    [(i,element_unraveled_log,
                      cell_centers,
                      elec_coords,
                      probs,
                      distance_edges,
                      amplitude_edges,
                      expected_value) 
                     for i in range(element_unraveled_log.shape[0])]
                )
    
    return np.asarray(responses)

def hex_grid_h(nodes_per_layer,box):
    """
    TODO: document
    """
    if nodes_per_layer < 1:
        hx = 0.0
        hy = 0.0

    elif nodes_per_layer == 1:
        hx = box[0,1] - box[0,0];
        hy = box[1,1] - box[1,0];

    else:
        hx = (box[0,1] - box[0,0] ) / ( nodes_per_layer - 1)
        hy = hx * np.sqrt ( 3.0 ) / 2.0

    return hx,hy

def hex_grid_points(nodes_per_layer,layers,n,box):
    """
    TODO: document
    """
    ndim = 2
    p = np.zeros((ndim,n))

    if nodes_per_layer < 1:
       return None

    if nodes_per_layer == 1:
        return ( box[0:ndim,0] + box[0:ndim,1] ) / 2.0

    hx, hy = hex_grid_h(nodes_per_layer, box)

    k = 0

    for j in np.arange(1,layers+1):
        y = box[1,0] + hy * (j - 1);

        jmod = j % 2

        if jmod == 1:

          for i in np.arange(1,nodes_per_layer+1):
            x = box[0,0] + (box[0,1] - box[0,0]) * (i-1)\
                    / ( nodes_per_layer - 1 );
            k +=1

            if k-1 <= n:
              p[0,k-1] = x;
              p[1,k-1] = y;

        else:

          for i in np.arange(1,nodes_per_layer):
            x = box[0,0] + (box[0,1] - box[0,0]) * (2 * i-1)\
                    / ( 2 * nodes_per_layer - 2 )

            k +=1

            if k-1 <= n:
              p[0,k-1] = x
              p[1,k-1] = y

    return p

def generate_rf_centers(layers,nodes_per_layer,jitter_factor=None,seed=11111):
    """
    TODO: document
    """
    n = layers * nodes_per_layer
    box = np.array(([0,1],[0,1]))
    coords = hex_grid_points(nodes_per_layer,layers,n,box)

    # Prune away zeros.
    zero_inds = np.intersect1d(np.argwhere(coords[0,:] == 0).flatten(),
                               np.argwhere(coords[1,:] == 0).flatten())[1:]
    coords = coords[:,np.setdiff1d(np.arange(0,n),zero_inds)]

    if jitter_factor is not None:
        np.random.seed(seed)
        spacing = np.sort(linalg.norm(coords[:,0][:,None] - coords,axis=0))[1]
        jitter = np.random.randn(coords.shape[1]) * spacing * jitter_factor
        coords[0,:] += jitter
        jitter = np.random.randn(coords.shape[1]) * spacing * jitter_factor
        coords[1,:] += jitter

    return coords.T

def generate_gaussian_params(centers,std_major_kde,skew_kde,tilt_kde,seed=11111):
    """
    TODO: document
    """
    gaussian_dict = dict()

    for param in ['center_x','center_y','std_x','std_y','tilt']:
        gaussian_dict[param] = []

    np.random.seed(seed) 
    n = centers.shape[0]
    std_major = std_major_kde.resample(n).squeeze()
    skew = skew_kde.resample(n).squeeze()
    std_minor = std_major / skew
    #tilts = tilt_kde.resample(n).squeeze() % 360
    tilts = (tilt_kde.resample(n).squeeze() % 360) * np.pi / 180 # to radians

    gaussian_dict['center_x'] = centers[:,0]
    gaussian_dict['center_y'] = centers[:,1]

    if np.random.rand() > .5:
        gaussian_dict['std_x'] = std_major
        gaussian_dict['std_y'] = std_minor
    else:
        gaussian_dict['std_x'] = std_minor
        gaussian_dict['std_y'] = std_major
    
    gaussian_dict['tilt'] = tilts

    return gaussian_dict

def get_rf_nnd(centers,n):
    """
    Estimates the nearest neighbor distribution for a set of centers.
    """
    nnd = []

    for i in range(centers.shape[0]):
        distances = linalg.norm(centers[i,:][None,:] - centers,axis=1)
        nnd.append(np.sort(distances)[n])
    
    return np.asarray(nnd)

def get_rf_nnd_robust(centers,scalar=1.5):
    """
    TODO: document
    """
    distances_list = []
    
    for i in range(centers.shape[0]):
        distances = linalg.norm(centers[i,:][None,:] - 
                                centers,axis=1)
        
        # Get the minimum distance and scale it to find upper bound.
        min_distance = np.sort(distances)[1] # exclude that cell
        inds = np.where((distances > 0) 
                        & (distances < scalar * min_distance))[0]
        distances_list.append(np.median(distances[inds]))
    
    return np.asarray(distances_list)

def get_optimal_lattice_dims(seed_nnd,dim_start,dim_end,jitter_factor,
                             n,height,percentile=50):
    """
    Gets the optimal lattice dimensions by matching the moments of the nearest
    neighbor distributions.
    """
    dim_range = np.arange(dim_start,dim_end+1)  
    nnd_moments = []

    for dim in dim_range:
        centers = generate_rf_centers(
                        dim,dim,jitter_factor,seed=11111
                  ) * height
        
        nnd = get_rf_nnd(centers,n)
        nnd_moments.append(np.percentile(nnd,percentile))
    
    nnd_moments = np.asarray(nnd_moments)

    return dim_range[np.argmin(
                np.abs(nnd_moments - 
                      np.percentile(seed_nnd,percentile)
                )
           )]

def get_activation_at_elements(dictionary,elec_coords,cell_centers):
    """
    TODO: document
    """
    np.random.seed(11111)
    _,n_elecs,n_amps = dictionary.shape
    probs_list = []
    #spikes_list = []
    distances_list = []
    amplitude_inds_list = []

    for elec_ind in range(n_elecs):
        distances = linalg.norm(
                        cell_centers - elec_coords[elec_ind,:][None,:],
                        axis=1
                    )
        for amp_ind in range(n_amps):
            probs = dictionary[:,elec_ind,amp_ind]
            for k in range(cell_centers.shape[0]):
                #spikes_list.append(np.random.binomial(1,probs[k]))
                probs_list.append(probs[k])
                distances_list.append(distances[k])
                amplitude_inds_list.append(amp_ind)
    
    activation_elements_dict = dict()
    activation_elements_dict['distances'] = np.asarray(distances_list)
    activation_elements_dict['probs'] = np.asarray(probs_list)
    activation_elements_dict['amplitudes'] = cfg.AMPLITUDES[np.asarray(
                                                    amplitude_inds_list
                                                )]
    
    return activation_elements_dict

def get_histogrammed_activation(n_bins_distance,n_bins_amplitudes,distances,
                                amplitudes,probs):
    """
    TODO: document
    """
    avg_probs = np.zeros((n_bins_distance,n_bins_amplitudes))

    # Set bin edges based on extrema.
    distance_edges = np.linspace(0,
                                np.max(distances),
                                 n_bins_distance+1)
    amplitude_edges = np.linspace(np.min(amplitudes), 
                                 np.max(amplitudes),
                                  n_bins_amplitudes+1)
    
    # Adjust the extrema slightly to avoid messy logic.
    distance_edges[0] *= .9
    distance_edges[-1] *= 1.1
    amplitude_edges[0] *= .9
    amplitude_edges[-1] *= 1.1

    for i in range(n_bins_distance):
        for j in range(n_bins_amplitudes):
            inds = np.intersect1d(np.where((distances >= distance_edges[i]) &
                                         (distances < distance_edges[i+1]))[0],
                                np.where((amplitudes >= amplitude_edges[j]) &
                                 (amplitudes < amplitude_edges[j+1]))[0]
                    )
            avg_probs[i,j] = np.nanmean(probs[inds])
    
    hist_dict = dict()
    hist_dict['probs'] = avg_probs
    hist_dict['distance_edges'] = distance_edges
    hist_dict['amplitude_edges'] = amplitude_edges

    return hist_dict

def get_early_stopped_recon_at_point(dictionary,decoder,elements,
                                    midget_responses,midget_decoder,
                                    stimulus,stop_ind):
    
    # Get the recon and noise for each element and take the cumsum.
    stimulus = np.reshape(stimulus,(cfg.RESAMPLE_FIELD_Y,cfg.RESAMPLE_FIELD_X))
    decoded_stimuli = dictionary[elements[0:stop_ind+1]]@decoder
    midget_noise = midget_responses[0:stop_ind+1,:]@midget_decoder
        
    decoded_stimuli = np.reshape(
                            np.asarray(decoded_stimuli),
                            (stop_ind+1,
                            cfg.RESAMPLE_FIELD_Y,
                            cfg.RESAMPLE_FIELD_X)
                    )
    midget_noise = np.reshape(
                            np.asarray(midget_noise),
                            (stop_ind+1,
                            cfg.RESAMPLE_FIELD_Y,
                            cfg.RESAMPLE_FIELD_X)
                    )
    decoded_stimuli_cumul = np.cumsum(decoded_stimuli,axis=0)
    midget_noise_cumul = np.cumsum(midget_noise,axis=0)
    decoded_w_noise_cumul = decoded_stimuli_cumul + midget_noise_cumul 
    
    return decoded_w_noise_cumul[-1,...]

def get_early_stopping_point_torch(dictionary,decoder,elements,midget_responses,
                                   midget_decoder,stimulus,mask):
    
    # Get the recon and noise for each element and take the cumsum.
    stimulus = np.reshape(stimulus,(cfg.RESAMPLE_FIELD_Y,cfg.RESAMPLE_FIELD_X))
    stimulus = torch.from_numpy(stimulus).to(torch.float32).to(cfg.GPU)
    dictionary = torch.from_numpy(dictionary).to(torch.float32).to(cfg.GPU)
    decoder = torch.from_numpy(decoder).to(torch.float32).to(cfg.GPU)
    midget_responses = torch.from_numpy(
                                midget_responses
                        ).to(torch.float32).to(cfg.GPU)
    midget_decoder = torch.from_numpy(
                                midget_decoder 
                        ).to(torch.float32).to(cfg.GPU)
    
    with torch.no_grad():
        decoded_stimuli = torch.reshape(
                                torch.matmul(dictionary[elements,:],
                                            decoder
                                ),(-1,cfg.RESAMPLE_FIELD_Y,cfg.RESAMPLE_FIELD_X)
                            )
        midget_noise = torch.reshape(
                            torch.matmul(midget_responses,midget_decoder)
                            ,(-1,cfg.RESAMPLE_FIELD_Y,cfg.RESAMPLE_FIELD_X)
                        )
        decoded_stimuli_cumul = torch.cumsum(decoded_stimuli,dim=0)
        midget_noise_cumul = torch.cumsum(midget_noise,dim=0)
    
    decoded_stimuli = decoded_stimuli.detach().cpu().numpy()
    midget_noise = midget_noise.detach().cpu().numpy()
    midget_responses = midget_responses.detach().cpu().numpy()
    
    del decoded_stimuli,midget_noise,midget_responses
    
    with torch.cuda.device(cfg.GPU):
        torch.cuda.empty_cache()
    
    # Recompute the error with the midget noise included and get min.
    dict_var = torch.from_numpy(
                    get_dictionary_variance(dictionary.detach().cpu().numpy(),
                                            decoder.detach().cpu().numpy())
               ).to(torch.float32).to(cfg.GPU)
    mask_ = torch.from_numpy(np.reshape(mask,
                       (1,cfg.RESAMPLE_FIELD_Y,cfg.RESAMPLE_FIELD_X)
            ).copy()).to(torch.float32).to(cfg.GPU)
    
    with torch.no_grad():
        decoded_w_noise_cumul = decoded_stimuli_cumul + midget_noise_cumul 
        decoded_stimuli_cumul = decoded_stimuli_cumul.detach().cpu().numpy()
        midget_noise_cumul = midget_noise_cumul.detach().cpu().numpy()
        
        del decoded_stimuli_cumul,midget_noise_cumul
        error_w_noise = (torch.sum(
                            (decoded_w_noise_cumul * mask_ - 
                            stimulus[None,...] * mask_)**2,
                            dim=(1,2)
                    )).detach().cpu().numpy() #+ dict_var[elements]).detach().cpu().numpy()
    
    dictionary = dictionary.detach().cpu().numpy()
    decoded_w_noise_cumul = decoded_w_noise_cumul.detach().cpu().numpy()
    min_ind = np.argmin(error_w_noise)
    
    del dictionary,decoded_w_noise_cumul,error_w_noise
    
    with torch.cuda.device(cfg.GPU):
        torch.cuda.empty_cache()
        
    return min_ind

def greedy_stim(dictionary_matrix,decoder,var_dict,
                sig_stixels_all,test_stimulus_vector): 
    """
    Function to run the greedy stimulation algorithm. Uses torch on GPU 
    for high performance relative to numpy/CPU.

    TODO: document
    """
    n_cells,n_pixels = decoder.shape
    decoded_stimuli = np.zeros((n_pixels,1))
    responses_all = np.zeros(n_cells)
    decoded_stimuli_partial = decoder.T@dictionary_matrix.T
    error_log = []
    element_log = []

    # Initialize the clock and refractory log
    t = 0
    cnt = 0
    refractory_log = np.zeros(n_cells)
    
    # Put on the GPU.
    decoder = torch.DoubleTensor(decoder).to(cfg.GPU)
    decoded_stimuli = torch.DoubleTensor(decoded_stimuli).to(cfg.GPU)
    decoded_stimuli_partial = torch.DoubleTensor(decoded_stimuli_partial
                                            ).to(cfg.GPU)
    test_stimulus_vector = torch.DoubleTensor(
                            test_stimulus_vector.reshape(n_pixels,
                                                        1)).to(cfg.GPU)
    var_dict = torch.DoubleTensor(var_dict).to(cfg.GPU)
    dictionary_matrix = torch.DoubleTensor(dictionary_matrix).to(cfg.GPU)
    responses_all = torch.DoubleTensor(responses_all).to(cfg.GPU)
    refractory_log = torch.DoubleTensor(refractory_log).to(cfg.GPU)
    
    while True:

        # Get perception, take the mean squared error and add on the variance.
        cumul_p = torch.add(decoded_stimuli,decoded_stimuli_partial)
        error = torch.sum(
                    (cumul_p[sig_stixels_all,:] - 
                    test_stimulus_vector[sig_stixels_all,:])**2,dim=0
                ) + var_dict
        elements = torch.argsort(error)

        # Get one with minimal error. 
        j = 0

        while torch.any(
            refractory_log[torch.where(
            dictionary_matrix[elements[j],:] 
            > cfg.MIN_SPIKE_PROB)[0]] > t):
            j +=1

            # If this brings us uphill, hold constant.
            if error[elements[j]] > error_log[-1]:
                j = torch.where(elements 
                                == elements.shape[0]-1)[0].cpu().numpy()[0]
                break

            # If we reach the end, break out. 
            if j == elements.shape[0]-1:
                break
                
        chosen_element = elements[j]
        error_log.append(error[chosen_element].item())
        element_log.append(chosen_element.item())

        # Add to the decoded stimuli
        decoded_stimuli += torch.reshape(
                           torch.matmul(
                           torch.t(decoder),
                           torch.t(dictionary_matrix[chosen_element,:])),
                           (n_pixels,1)
                           )
        responses_all += dictionary_matrix[chosen_element,:]
        
        if cnt % 500 == 0:
            print("stimulation: %s, error: %.3f"%(cnt,error[chosen_element]))

        # Update refractory log based on stimulation.
        refractory_log[torch.where(
                    dictionary_matrix[chosen_element,:] 
                    > cfg.MIN_SPIKE_PROB
                    )[0]] = cfg.REFRACTORY_PERIOD + t
        t += cfg.STIMULATION_DT
        cnt +=1

        # Exit the loop when the no-op element is chosen for a critical period. 
        if (len(element_log) > cfg.REFRACTORY_WINDOW and 
            np.all(np.asarray(element_log[-cfg.REFRACTORY_WINDOW:]) 
            == element_log[-1]) and
            torch.all(dictionary_matrix[element_log[-1],:] == 0.0)):
            print('converged after %s total stimulations.'%str(cnt))
            break
    
    # Put everything on CPU to avoid memory issues and write to dictionary.
    greedy_dict = dict()
    greedy_dict['decoded_stimuli'] = decoded_stimuli.cpu().numpy()
    greedy_dict['error_log'] = error_log
    greedy_dict['element_log'] = element_log
    greedy_dict['responses_all'] = responses_all.cpu().numpy()
    
    return greedy_dict
