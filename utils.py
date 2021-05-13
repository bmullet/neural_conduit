"""
Utility functions
"""

import numpy as np


def mat_to_py(var):
    """
    converts matlab arrays to numpy arrays and makes sure everything is shape Nx1
    """ 
    
    val = np.array(var)
    
    assert len(val.shape) <= 2, "too many dims to conver to Nx1"
    
    if len(val.shape) < 2:
        val = val[:,np.newaxis]
    
    L, W = val.shape

    if L < W:
        val = val.T
    return val

def get_steady_state_vars_and_grads(ss):
    """
    Given a steady state object "ss", return conduit variables and gradients
    Input:
      ss: dict from domeconduit steady state matlab code
    Returns:
      vars: dict of variables
      grads: dict of gradients
    """
    
    conduit_var_indices = {"p" : 0 ,"v" : 1, "phi_g" : 2, "mw" : 3}
    
    var_dict = {}
    grad_dict = {}
    
    y = np.array(ss["y"])
    yp = np.array(ss["yp"])
    
    for var, index in conduit_var_indices.items():
        var_dict[var] = mat_to_py(y[index, :])
        grad_dict[var] = -mat_to_py(yp[index, :]) # Need negative sign for positive up
        
    # Also get the z coordinate
    var_dict["z"] = -mat_to_py(ss["z"])
    
    return var_dict, grad_dict
    
    
    