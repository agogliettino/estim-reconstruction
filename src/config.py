"""
Constants used in the project
"""
import os
N_THREADS = 10
os.environ["OMP_NUM_THREADS"] = f"{N_THREADS}" 
os.environ["OPENBLAS_NUM_THREADS"] = f"{N_THREADS}" 
os.environ["MKL_NUM_THREADS"] = f"{N_THREADS}" 
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{N_THREADS}" 
os.environ["NUMEXPR_NUM_THREADS"] = f"{N_THREADS}" 

import numpy as np
import torch

DB_ANIMALS = '/home/agogliet/gogliettino/projects/papers/raphe/repos/'\
              'raphe-vs-periphery/tmp/database-spreadsheet/'\
              'database_spreadsheet_animals.csv'
DB_PIECES = '/home/agogliet/gogliettino/projects/papers/raphe/repos/'\
            'raphe-vs-periphery/tmp/database-spreadsheet/'\
            'database_spreadsheet_pieces.csv'
DB_DATARUNS = '/home/agogliet/gogliettino/projects/papers/raphe/repos/'\
              'raphe-vs-periphery/tmp/database-spreadsheet/'\
              'database_spreadsheet_runs.csv'

FS = 20000 # hz
MONITOR_FS = 120 # hz
ANALYSIS_PARENT = '/Volumes/Analysis'
LNP_PARENT = '/home/agogliet/gogliettino/projects/'\
             'papers/raphe/repos/raphe-vs-periphery/tmp/lnp-model'
ACF_PARENT = '/home/agogliet/gogliettino/projects/papers/raphe/'\
              'repos/raphe-vs-periphery/tmp/spiking/acfs'
XCORR_PARENT = '/home/agogliet/gogliettino/projects/papers/raphe/'\
              'repos/raphe-vs-periphery/tmp/spiking/xcorrs'
ACV_PARENT = '/home/agogliet/gogliettino/projects/papers/raphe/'\
              'repos/raphe-vs-periphery/tmp/spiking/acvs'
ACF_FNAME = '%s_acf.npy'
ACV_FNAME = '%s_acv.npy'
XCORR_FNAME = '%s_xcorr.npy'
STA_FNAME = '%s_sta.npy'
STA_FIT_FNAME = '%s_sta_fit.npy'
TC_FNAME = '%s_tc.npy'
TC_FIT_FNAME = '%s_tc_fit.npy'
GS_FNAME = '%s_gs.npy' # generator
BS_FNAME = '%s_bs.npy' # spikes
NL_FNAME = '%s_nl.npy' # nonlinearity
NL_FIT_FNAME = '%s_nl_fit.npy'
SM_FNAME = '%s_sm.npy' # spatial maps
CLASS_FNAME = '%s.classification_agogliet.txt'
MONITOR_BIT_DEPTH = 255
STA_NORM_MEAN = .5 # for normalizing STAs
MOVIE_PATH = '/Volumes/Analysis/stimuli/white-noise-xml/'
XML_EXT = '.xml'
GPU = 'cuda:3'
TORCH_N_THREADS = 20
MICRONS_PER_PIXEL = 5.5 # on a CRT display with 6.5x lens.
MICRO_V_PER_ADC = 3.6 # on the rig1/4 electrical stimulation DAQ systems only!

# midget RF KDE fitting.
MIDGET_KDE_LB = 20 # percentile
MIDGET_KDE_UB = 80 # percentile

# for the midget reconstruction.
MIDGET_N_BINS_D = 45 # distance binning
MIDGET_N_BINS_A = 12 # amplitude binning

# parameter bounds for difference of cascades of LFP fitting
LOWER_BOUNDS_LFP_FIT = (-1.11,-.49,0,0,.98,-8.49)
UPPER_BOUNDS_LFP_FIT = (1.53,.49,np.inf,np.inf,28.91,16.33)

# for nonlinearity fitting
MAX_G_MULT = 1.1 # 110% of largest generator value

# gsort
PARENT_GSORT = '/Volumes/Scratch/Users/agogliet/raphe-tmp-final/estim/gsort-out'
GSORT_FNAME = 'gsort_full_data_tensor.mat'
PARENT_BUNDLE = '/Volumes/Scratch/Users/agogliet/raphe-tmp-final/estim/bundles'
BUNDLE_FNAME = '%s_auto_thresholds.npy'
DICTIONARY_FNAME = 'dictionary.npy'
DISCONNECTED = [1,130,259,260,389,390,519] # disconnected on 519 array
EPSILON = .001 # below which to call 0 for the greedy dictionary.

# recon.
FAKE_MIDGET_SEED_LIST = [1111,2345,333,1995,1928,2005,1999,2358,5749,9847,2575]
PARENT_RECON = '/Volumes/Scratch/Users/agogliet/raphe-tmp-final/estim/recon' 
#RECON_FNAME = "%s_recon_filters.npy"
RECON_FNAME = "%s_recon_filters_pooled.npy"
RECON_FNAME_FAKE_MIDGETS = '%s_recon_filters_fake_midgets.npy'
RECON_FNAME_FAKE_MIDGETS_PRUNED = '%s_recon_filters_fake_midgets_pruned.npy'
RECON_FNAME_FAKE_MIDGETS_NS = '%s_recon_filters_fake_midgets_ns.npy'
RECON_FNAME_FAKE_MIDGETS_PRUNED_NS = '%s_recon_filters_fake_midgets_pruned_ns.npy'
RFS_FNAME_FAKE_MIDGETS_PRUNED = '%s_rfs_fake_midgets_pruned.npy'
EARLY_STOP_FNAME = '%s_early_stop_dict.npy'
EARLY_STOP_FNAME_NS = '%s_early_stop_dict_ns.npy'
RECON_NS_FNAME = "%s_recon_filters_ns.npy"
PARENT_GREEDY = '/Volumes/Scratch/Users/agogliet/raphe-tmp-final/estim/recon/greedy-out'
GREEDY_FNAME = "%s_greedy_results.npy"
GREEDY_FNAME_NS = "%s_greedy_results_ns.npy" # natural scenes
GREEDY_FNAME_CH = "%s_greedy_results_ch.npy" # text chars

# cnn
PARENT_CNN = '/Volumes/Scratch/Users/agogliet/raphe-tmp-final/estim/recon/cnn'

# pixels for cropping at stixel size 2 to get rid of zeros
IMAGENET_CROP_Y1 = 8 
IMAGENET_CROP_Y2 = 152
IMAGENET_CROP_X1 = 88
IMAGENET_CROP_X2 = 232

# filenames for training
CNN_TRAIN_X_FNAME = "cnn_train_x_%s.npy"
CNN_TRAIN_Y_FNAME = "cnn_train_y.npy"
CNN_TEST_X_FNAME = "cnn_test_x_%s.npy"
CNN_TEST_Y_FNAME = "cnn_test_y.npy"
CNN_MODEL_FNAME = "cnn_model_%s"

# hyperparams
N_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.0004 # from nikhil
DROPOUT_PROB = 0.25 # from nikhil

# Amplitudes in μA for old labview 1-el scan
AMPLITUDES = np.array([.1005, .1131, .1194, .1320, .1445, .1634, .1759,
       .1948, .2136, .2388, .2576, .2780, .3033, .3539, .3791, .4297, .4550, 
       .5055, .5561, .6066, .6824, .7330, .8088, .8846,.9857, 1.1043, 1.2047, 
       1.3051, 1.4055, 1.6063, 1.7067, 1.9075, 2.1083, 2.3091, 2.5098, 2.8110, 
       3.1122, 3.4134, 3.7146, 4.1161])
MP_N_THREADS = 20 # multiprocessing threads
N_SIGMAS_RF = 2 # number of stds for RF size
N_SIGMAS_GAUSSIAN_MASK = 4 # number of stds for masking
#ACF_DELAY = 100 # in ms
ACF_DELAY = 500 # in ms
ACF_DELAYS = np.arange(1,ACF_DELAY) # delays in ms for ACF computation
ACV_THRESHOLD = 30 # threshold factor for spike conduction velocity.
ACF_BIN_SIZE = 1 # 1 ms binning

# for xcorrs
XCORR_DELAY = 1000 # in ms
XCORR_DELAYS = np.arange(-XCORR_DELAY,XCORR_DELAY+1)
XCORR_BIN_SIZE = 1 # 1 ms binning
XCORR_TAU = 5 # ms, for Gaussian smoothing

# Reconstruction analysis constants
N_TRAIN_STIMULI = 10000
PRE_TRAIN_SEED = 1995
TRAIN_SEED = 2017
ALPHA = 1 / 10000
N_MS_POST_FLASH = 100 # ms
MAX_PIX = .48
MIN_PIX = -.48
N_PIXELS_X = 640
N_PIXELS_Y = 320
N_PIXELS_WINDOW = 300

"""
firing rate stats for the linear-nonlinear binomial spiking model.
note that the sigma here is not the spiking variance but rather a Gaussian
sigma for the noise to add on the generator signal to get spiking with 
emprically observed variance.
"""
FIRING_STAT_DICT = {
       'wn' :{'parasol_mu': 2.04,'midget_mu': .58,
             'parasol_sigma': 1.055,'midget_sigma': .66},
       'ns' :{'parasol_mu': 1.66,'midget_mu': .50,
             'parasol_sigma': 1.055,'midget_sigma': .81},
}

SPIKES_PER_FLASH = 3 # From Nishal's experiments.
IMAGENET_PARENT = '/Volumes/Data/Stimuli/movies/imagenet'
IMAGENET_PRETRAIN = 'ImageNet_stix2_1_045.rawMovie'
IMAGENET_TRAIN = 'ImageNet_stix2_0_045.rawMovie'
IMAGENET_TRAIN_CNN = 'ImageNet_stix2_2_045.rawMovie'
IMAGENET_TEST_CNN = 'ImageNet_stix2_5_045.rawMovie'
IMAGENET_TEST = 'ImageNetTest_v2.rawMovie'
N_TEST_STIMULI_CNN = 1000
IMAGENET_TRAIN_LIST = [
       IMAGENET_TRAIN,IMAGENET_TRAIN_CNN,IMAGENET_PRETRAIN,
       'ImageNet_A.rawMovie','ImageNet_B.rawMovie','ImageNet_C.rawMovie',
       'ImageNet_D.rawMovie','ImageNet_E.rawMovie','ImageNet_F.rawMovie',
       'ImageNet_G.rawMovie'
]
IMAGENET_TRAIN_INDS = np.arange(0,6)
IMAGENET_PRETRAIN_INDS = np.arange(6,10)
N_TEST_STIMULI = 15
N_TEST_STIMULI_NS = 100
N_TEST_STIMULI_CH = 3
REFRACTORY_PERIOD = 5 # ms.
STIMULATION_DT = .15 # ms
MIN_SPIKE_PROB = 1/25 # for greedy refractory period 
MIN_SPIKE_PROB_FIT = 3/25 # FOR SIGMOID FITTING: BELOW THIS MAXIMUM, SET TO 0
TEST_SEED = 2022
SIGMOID_FIT_MSE = .01 # above this, the fit is likely to be bad.
N_AMPS_BLANK = 12 # zero these first amplitudes to avoid noise.
DEVICES = [torch.device('cuda:1'),torch.device('cpu')]

# Compute the number of stimulations within a refractory window.
REFRACTORY_WINDOW = int(np.ceil(REFRACTORY_PERIOD / STIMULATION_DT)) * 2 # TODO

RESAMPLE_FIELD_X = 320
RESAMPLE_FIELD_Y = 160
RESAMPLE_STIX = 2 

STIXEL_SIZES = np.flip(np.array([10,16,20,32,40,64]))
