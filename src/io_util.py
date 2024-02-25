import numpy as np
import os
import sys
import whitenoise.random_noise as rn
import config as cfg
sys.path.insert(0,
 '/home/agogliet/gogliettino/projects/papers/raphe/repos/raphe-vs-periphery/src/')
import rawmovie as rm
import h5py
import pandas as pd
import pdb
import cv2

"""
Miscellaneous io utilities.
"""
def get_animal_db():
    return pd.read_csv(cfg.DB_ANIMALS)

def get_piece_db():
    return pd.read_csv(cfg.DB_PIECES)

def get_datarun_db():
    return pd.read_csv(cfg.DB_DATARUNS)

def get_h5_movie(movie_path,n_pixels_y,n_pixels_x):
    """
    Gets h5 file and pads with the mean to create a full image.
    """
    with h5py.File(movie_path) as f:
        a_group_key = list(f.keys())[0]
        img = list(f[a_group_key])
        img = np.asarray(img)
    
    # Pad the original image
    pad_y = (n_pixels_y - img.shape[0]) / 2
    pad_x = (n_pixels_x - img.shape[1]) / 2
    pad_values = np.mean(img.ravel())

    return np.pad(
            img,
            (0,0),
            (pad_y,pad_y),
            (pad_x,pad_x),
            mode='constant',
            constant_values=pad_values
          )
    
def get_imagenet_stimuli(train_inds,dtype=np.float32):
    """
    Convienence function to get a bunch of different NS movies in sets of 10K.
    """ 
    assert (np.min(train_inds) >=0 and
            np.max(train_inds) < len(cfg.IMAGENET_TRAIN_LIST))
    
    stimulus = [] 
    
    for j in train_inds:
        fname = cfg.IMAGENET_TRAIN_LIST[j]
        
        # Some images are downsampled 2x so gotta upsample.
        tmp = get_raw_movie(os.path.join(cfg.IMAGENET_PARENT,
                                fname),
                            cfg.N_TRAIN_STIMULI
                    ).astype(dtype)[...,0]

        if j > 4:
            tmp = np.asarray([cv2.resize(tmp[i,...],
                              dsize=(cfg.RESAMPLE_FIELD_X,cfg.RESAMPLE_FIELD_Y),
                              interpolation=cv2.INTER_CUBIC)
                              for i in range(tmp.shape[0])
                ])
            
        stimulus.append(tmp)
    
    return np.vstack(stimulus)[...,None]
    
def get_raw_movie(raw_movie_path,n_frames):
    """
    Utility for getting natural scenes raw movies from disk 
    Parameters:
        raw_movie_path: path to stimulus
        n_frames: number of unique frames (int)
    Returns: 
        stimulus tensor of size n,y,x,c 
    """
    assert os.path.isfile(raw_movie_path), "Stimulus provided not found."

    # Initialize the stimulus object.
    rm_obj = rm.RawMovieReader(raw_movie_path)
    stimulus_tensor,_ = rm_obj.get_frame_sequence(0,n_frames)

    return stimulus_tensor

def get_movie_xml_str(vcd,seed,contrast,independent):
    """
    Constructs stimulus XML str of the form
     RGB/BW-stixel-interval-contrast-seed.xml
    Parameters:
        vcd: vision object
        seed: seed of stimulus
        contrast: contrast of stimulus
    Returns:
        stimulus movie str
    """
    stixel = int(vcd.runtimemovie_params.pixelsPerStixelX)
    interval = int(vcd.runtimemovie_params.interval)
    
    if independent:
        movie_xml_str = f'RGB-{stixel}-{interval}-{contrast}-{seed}.xml'
    else:
        movie_xml_str = f'BW-{stixel}-{interval}-{contrast}-{seed}.xml'
    
    return movie_xml_str

def get_celltypes_dict(class_path,lower=True):
    """
    Gets the celltype dictionary mapping IDs to celltypes from
    a text file.

    Parameters:
        class_path: full path to text file of cell types
        lower: boolean indicating whether to lowercase the strings

    Returns:
        dictionary mapping IDs to celltype.
    """

    f = open(class_path)
    celltypes_dict = dict()

    for j in f:
        tmp = ""

        for jj,substr in enumerate(j.split()[1:]):
            tmp +=substr

            if jj < len(j.split()[1:])-1:
                tmp += " "

        if lower:
            tmp = tmp.lower()

        celltypes_dict[int(j.split()[0])] = tmp

    f.close()

    return celltypes_dict

def get_stimulus(movie_xml_str,n_frames,resample=True,
                normalize=False,center=False,grayscale=False):
    """
    Gets a white noise visual stimulus from an xml str

    Parameters:
        movie_xml_str: white noise movie xml
        n_frames: number of frames
        resample: boolean indicating whether to upsample based on interval.
        normalize: boolean to normalize in 0,1
        center: boolean to mean subtract.
        grayscale: boolean indicating whether to grayscale.
    Returns:
        tensor of size frames x height x width x color channels
    """

    # Check that the stimulus exists, and initialize object.
    if cfg.XML_EXT not in movie_xml_str:
        movie_xml_str += cfg.XML_EXT

    movie_xml_path = os.path.join(cfg.MOVIE_PATH,movie_xml_str)

    if not os.path.isfile(movie_xml_path):
        raise OSError("%s not found."%movie_xml_str)

    # Load the stimulus and resample if necessary.
    rn_obj = rn.RandomNoiseFrameGenerator.construct_from_xml(movie_xml_path)

    stimulus = []

    for i in range(n_frames):
        stimulus.append(rn_obj.generate_next_frame())
    
    stimulus = np.asarray(stimulus)

    if normalize:
        stimulus /= cfg.MONITOR_BIT_DEPTH

    if center:
        stimulus -= np.mean(stimulus.ravel())

    if grayscale:
        stimulus = np.mean(stimulus,axis=-1,keepdims=True)

    interval = int(movie_xml_str.split('-')[2]) # Assumes RGB-pix-int structure

    if interval == 1 or not resample:
        return stimulus

    return np.repeat(stimulus,interval,axis=0)
