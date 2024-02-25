"""
Module of functions for analysis of spiking properties: EIs, ACFs, etc.
"""
import numpy as np
import scipy as sp
from scipy import linalg
import src.config as cfg
from scipy import stats, signal
from itertools import combinations
import electrode_map as elcmp
import pdb

def compute_acf(vcd,cell,delays=cfg.ACF_DELAYS):
    """
    Computes the autocorrelation function of the spike train with lags given
    in delays. Omits the zero bin. L2-normalizes the histogram.
    Parameters:
        vcd: vision cell data table object
        cell: cell id
        delays: delay (in ms)
    Returns:
        tuple of raw and L2 normalized spike train autocorrelation function.
    """
    def bin_spikes():
        """
        Helper function for binning spike trains with 1 ms precision. to be 
        used ONLY for correlations, and not for model fitting.

        Returns:
            binned spike train with 1 ms precision.
        """
        n_sec = vcd.n_samples / cfg.FS
        bin_edges = np.linspace(0, n_sec* 1000,
                         (int(n_sec/cfg.ACF_BIN_SIZE) * 1000)+1)
        spiketimes = (vcd.get_spike_times_for_cell(cell) / cfg.FS) * 1000

        return np.histogram(spiketimes,bin_edges)[0]
        
    # Get the binned spikes.
    binned_spikes = bin_spikes()

    # Find nonzero indices (spike times) and initialize histogram
    spiketimes1 = np.argwhere(binned_spikes).flatten()
    spiketimes2 = np.argwhere(binned_spikes).flatten()
    acf_out = np.zeros(delays.shape)

    # Loop through spike train and get frequency of each delay.
    for spiketime in spiketimes1:
        diff = spiketimes2 - spiketime
        inds = np.where((diff >= -cfg.ACF_DELAY) & (diff <= cfg.ACF_DELAY))[0]
        
        for ind in inds:
            
            if np.argwhere(delays == diff[ind]).flatten().shape[0] == 0:
                 continue
               
            delay_ind = np.argwhere(delays == diff[ind]).flatten()[0]
            acf_out[delay_ind] += 1

    return acf_out,acf_out / linalg.norm(acf_out)

def compute_xcorr(spiketrain1,spiketrain2,delays):
    """
    Computes a cross correlation between two spiketrains
    """
    spiketimes1 = np.argwhere(spiketrain1 != 0).flatten()
    spiketimes2 = np.argwhere(spiketrain2 != 0).flatten()
    xcorr = np.zeros(delays.shape)

    for spiketime in spiketimes1:
        diff = spiketimes2 - spiketime
        inds = np.where((diff >= -cfg.XCORR_DELAY) &
                        (diff <= cfg.XCORR_DELAY))[0]
        
        for ind in inds:
            
            if np.argwhere(delays == diff[ind]).flatten().shape[0] == 0:
                 continue
               
            delay_ind = np.argwhere(delays == diff[ind]).flatten()[0]
            xcorr[delay_ind] += 1

    return xcorr,xcorr / linalg.norm(xcorr)
