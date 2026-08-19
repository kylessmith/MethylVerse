import numpy as np
import pandas as pd
from scipy.stats import norm


def control_illumina_single(idat_grn, idat_red, controls, ref=10000):
    """
    """
    
    #AT_controls = manifest.getControlAddress(rgSet, controlType = ["NORM_A", "NORM_T"])
    AT_controls = controls.loc[["NORM_A","NORM_T"]].values
    AT_controls = pd.Index(AT_controls).intersection(idat_grn.index)
    #CG_controls = manifest.getControlAddress(rgSet, controlType = ["NORM_C", "NORM_G"])
    CG_controls = controls.loc[["NORM_C","NORM_G"]].values
    CG_controls = pd.Index(CG_controls).intersection(idat_grn.index)
        
    #green_avg = np.mean(rgSet.getGreen(CG_controls), axis=1)
    green_avg = np.mean(idat_grn.loc[CG_controls,"means"].values)
    #red_avg = np.mean(rgSet.getRed(AT_controls), axis=1)
    red_avg = np.mean(idat_red.loc[AT_controls,"means"].values)

    green_factor = ref / green_avg
    red_factor = ref / red_avg
    
    idat_grn.loc[:,"means"] = idat_grn.loc[:,"means"].values * green_factor
    idat_red.loc[:,"means"] = idat_red.loc[:,"means"].values * red_factor
        
    return 


def bgcorrect_illumina(idat_grn, idat_red, controls):
    """
    """
    
    NegControls = controls.loc["NEGATIVE"].values
    NegControls = pd.Index(NegControls).intersection(idat_grn.index)
    
    green_bg = np.sort(idat_grn.loc[NegControls,"means"].values)[30]
    red_bg = np.sort(idat_red.loc[NegControls,"means"].values)[30]
    
    idat_grn.loc[:,"means"] = np.maximum(idat_grn.loc[:,"means"].values - green_bg, 0)
    idat_red.loc[:,"means"] = np.maximum(idat_red.loc[:,"means"].values - red_bg, 0)
    
    return


def calculate_detection_pvalues(idat_grn, idat_red, controls):
    """
    """
    
    NegControls = controls.loc["NEGATIVE"].values
    NegControls = pd.Index(NegControls).intersection(idat_grn.index)
    
    neg_grn = idat_grn.loc[NegControls,"means"].values
    neg_red = idat_red.loc[NegControls,"means"].values

    neg_grn_mean = np.mean(neg_grn)
    neg_red_mean = np.mean(neg_red)
    neg_grn_std = np.max([np.std(neg_grn), 1.0])
    neg_red_std = np.max([np.std(neg_red), 1.0])
    
    neg_grn_z = (idat_grn.loc[:,"means"].values - neg_grn_mean) / neg_grn_std
    neg_red_z = (idat_red.loc[:,"means"].values - neg_red_mean) / neg_red_std

    pvals_grn = norm.sf(neg_grn_z)
    pvals_red = norm.sf(neg_red_z)

    pvalues = pd.Series(np.maximum(pvals_grn, pvals_red), index=idat_grn.index)

    return pvalues