# The function "supply_scaling" calculates the post_shock values for x and f for the different rationinig variations where the shocks are scaled by a certain scaling factor
# outputs: post-shock values for x and f for the each rationing scheme for different scaling factors

import pandas as pd
import numpy as np
from collections import namedtuple
from rationing_functions import algo_prop, algo_ordered
from shockscale import shockscale

def supply_scaling(x, xmax, f, fmax, A, L, TT=20, RUNS=20):

    seq = np.linspace(0, 1, 100) # scaling factors
    prop_x = [None] * len(seq); mixed_x = [None] * len(seq); prior_x = [None] * len(seq); random_x = [None] * len(seq)
    prop_f = [None] * len(seq); mixed_f = [None] * len(seq); prior_f = [None] * len(seq); random_f = [None] * len(seq)
    prop_i = [None] * len(seq); mixed_i = [None] * len(seq); prior_i = [None] * len(seq); random_i = [None] * len(seq)

    for i in range(len(seq)):
        scaleres = shockscale(x = x, xmax=xmax, f = f, fmax = fmax, scale_x = seq[i], scale_f = 0)

        ## rationing schemes
        prop = algo_prop(A = A, L = L, 
                         f = scaleres.fshocked, xmax = scaleres.xshocked, 
                         prioritize="no", TT=TT)
        mixed = algo_prop(A = A, L = L, 
                          f = scaleres.fshocked, xmax = scaleres.xshocked, 
                          prioritize="firms", TT=TT)
        prior = algo_ordered(A = A, L = L, 
                             f = scaleres.fshocked, xmax = scaleres.xshocked, 
                             ordering ="priority", TT=TT)

        if prop.i == TT:
            prop_x[i] = prop_f[i] = None # don't report if not converged
        else:
            prop_x[i] = prop.d.iloc[:,-1:].sum(axis=0)[0]/x.sum(axis=0)
            prop_f[i] = prop.f.iloc[:,-1:].sum(axis=0)[0]/f.sum(axis=0)
        prop_i[i] = prop.i
    
        if mixed.i == TT:
            mixed_x[i] = mixed_f[i] = None # don't report if not converged
        else:
            mixed_x[i] = mixed.d.iloc[:,-1:].sum(axis=0)[0]/x.sum(axis=0)
            mixed_f[i] = mixed.f.iloc[:,-1:].sum(axis=0)[0]/f.sum(axis=0)
        mixed_i[i] = mixed.i

        if prior.i == TT:
            prior_x[i] = prior_f[i] = None # don't report if not converged
        else:
            prior_x[i] = prior.d.iloc[:,-1:].sum(axis=0)[0]/x.sum(axis=0)
            prior_f[i] = prior.f.iloc[:,-1:].sum(axis=0)[0]/f.sum(axis=0)
        prior_i[i] = prior.i
    
        # random rationing
        random_x[i] = [None] * RUNS; random_f[i] = [None] * RUNS; random_i[i] = [None] * RUNS
        for k in range(RUNS):
            random = algo_ordered(A = A, L = L, 
                               f = scaleres.fshocked, xmax = scaleres.xshocked, 
                               ordering="random", TT=TT)
            if random.i == TT:
                random_x[i][k] = random_f[i][k] = None # don't report if not converged
            else:
                random_x[i][k] = random.d.iloc[:,-1:].sum(axis=0)[0]/x.sum(axis=0)
                random_f[i][k] = random.f.iloc[:,-1:].sum(axis=0)[0]/f.sum(axis=0)
            random_i[i][k] = random.i
            
        # average of k values (removing None's)
        random_x[i] = np.mean(list(filter(lambda j: j is not None, random_x[i])))
        random_f[i] = np.mean(list(filter(lambda j: j is not None, random_f[i])))
        random_i[i] = np.mean(list(filter(lambda j: j is not None, random_i[i])))
        
    output = namedtuple("output", ["prop_x", "prop_f", "prop_i", "mixed_x", "mixed_f", "mixed_i",
                                  "prior_x", "prior_f", "prior_i", "random_x", "random_f", "random_i"])
    return output(prop_x, prop_f, prop_i, mixed_x, mixed_f, mixed_i, 
                  prior_x, prior_f, prior_i, random_x, random_f, random_i) 

