"""
Misc. functions with no other home but otherwise useful
"""
import numpy as np
import pdb
import pandas as pd

def gaussian(tau,t):
    """ Gaussian kernel """
    return 1 / np.sqrt(2 * np.pi) * np.exp(-t**2 / tau**2)

def iqr_bounds(data):
    """
    Returns bounds for the data based on the IQR method.
    """
    upper_q = np.nanpercentile(data,75)
    lower_q = np.nanpercentile(data,25)
    iqr = upper_q - lower_q
    lb  = lower_q - (1.5 * iqr)
    ub = upper_q + (1.5 * iqr)

    return lb,ub

def remove_outliers_iqr(data):
    """
    Remove outliers from data by IQR method.
    """
    lb,ub = iqr_bounds(data)
    return data[(data > lb) & (data < ub)]

def get_inlier_iqr_inds(data):
    """_summary_

    Args:
        data (_type_): _description_
    """
    lb,ub = iqr_bounds(data) 
    return np.where((data > lb) & (data < ub))[0]

def bootstrap_p_value(dist1,dist2,n_sim=10000):
    """
    Bootstrap p value calculation. Assumes a two-tailed hypothesis test to be
    most statistically rigorous.

    TODO: document
    """
    null_dist = np.r_[dist1,dist2]
    #np.random.shuffle(null_dist)
    mean_diff = np.abs(np.median(dist1) - np.median(dist2))
    cnt = 0

    for i in range(n_sim):
        dist1_resample = np.random.choice(
                                          null_dist,
                                          dist1.shape[0],
                                          replace=True
                        )
        dist2_resample = np.random.choice(
                                          null_dist,
                                          dist2.shape[0],
                                          replace=True
                        )
        
        if np.abs(np.median(dist1_resample) - 
                  np.median(dist2_resample)) >= mean_diff:
            cnt +=1
        
    return cnt / n_sim