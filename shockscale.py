# The function "shockscale" calculates the constraints for each industry for a certain scaling factor
# outputs: xshocked = x constraint after scaling; fshocked = f constraint after scaling

import pandas as pd
import numpy as np
from collections import namedtuple

def shockscale(x, xmax, f, fmax, scale_x, scale_f):
    ss_full = xmax/x - 1 # = -\epsilon^S
    ds_full = fmax/f - 1 # = -\epsilon^D
    
    ss = ss_full * scale_x
    xshocked = (1+ss) * x
    
    ds = ds_full * scale_f
    fshocked = (1+ds) * f
    
    output = namedtuple("output", ["xshocked", "fshocked"])
    return output(xshocked, fshocked)

