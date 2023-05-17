# The function "model" calculates the constraints for each industry based on the shocks the industry faces
# outputs: xmax & xmin = supply constraints; fmax & fmin = demand constraints for each final demand category;
# c = sum of different values for the final demand = total final demand 


import numpy as np
import pandas as pd
from collections import namedtuple

def model(x, f, supply_shock, demand_shocks_consumption, demand_shocks_invest, demand_shocks_export):

    N = len(f)
    
    ## supply constraints: xmax = x_0 * (1-\epsilon^S)
    xmax = x.reset_index(drop=True) * (1 - supply_shock) 
    xmax.index = x.index
    xmin = np.repeat(0,N)

    ## demand constraints: fmax = f_0 * (1-\epsilon^D)  
    # household consumption(P3_S14) & non-profit consumption (P3_S15) shocks
    fmax1 = f.iloc[:, 1:3].sum(axis=1).reset_index(drop=True) * (1 - demand_shocks_consumption) 
    # no shocks to domestic governments (P3_S13)
    fmax2 = f.iloc[:, 0].reset_index(drop=True) 
    # investment (P51G) and inventory (P5M) shocks
    fmax3 = f.iloc[:, 3:5].sum(axis=1).reset_index(drop=True) * (1 - demand_shocks_invest)
    # export shocks (Export, RoW_P3_S13, RoW_P3_S14+S15, RoW_P551G, RoW_P5M)
    fmax4 = f.iloc[:, 5:10].sum(axis=1).reset_index(drop=True) * (1 - demand_shocks_export) 
    fmax = fmax1 + fmax2 + fmax3 + fmax4
    fmax.index = x.index
    fmin = np.repeat(0,N)
    
    c = f.sum(axis=1) # sum of different f values for each industry in Belgium (total final demand)

    output = namedtuple("output", ["xmax", "xmin", "fmax", "fmin", "c"])
    return output(xmax, xmin, fmax, fmin, c)

