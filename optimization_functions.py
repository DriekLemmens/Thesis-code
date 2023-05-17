# The function "maxX" calculates the post-shock values for x and f when maximizing the gross output (x)
# outputs: maxX_f_sol = f when maximizing x; maxX_x_sol = x when maximizing x
# The function "maxF" calculates the post-shock values for x and f when maximizing the final demand (f)
# outputs: maxF_x_sol = x when maximizing f; maxF_f_sol = f when maximizing f
# The function "LP" combines the results for maximizing both x and f in one table
# outputs: xvalues = table with initial x, xmin, xmax, maxF_x_sol and maxX_x_sol; 
# fvalues = table with initial f, fmin, fmax, maxF_f_sol and maxX_f_sol


import pandas as pd
import numpy as np
from scipy.optimize import linprog
from collections import namedtuple


# maximizing gross output
def maxX(L, xmax, fmax):
    obj = list(-L.sum(axis=0)) # 1^T * L ('-' to get max)

    # constraints
    lhs_ineq = np.concatenate((-L, L))
    rhs_ineq = np.concatenate((np.zeros(len(xmax)), xmax))

    # boundaries (0 \leq f \leq fmax)
    bnd = [0]*len(fmax)
    for i in range(0, len(fmax)):
        bnd[i] = (0, fmax[i])

    opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, bounds=bnd, method="revised simplex")
    maxX_f_sol = opt.x
    maxX_x_sol = L.dot(maxX_f_sol) # x = L*f
    return  maxX_f_sol, maxX_x_sol

# maximizing final consumption
def maxF(A, xmax, fmax):
    I = np.identity(len(A))
    obj = list(-(I - A).sum(axis=0)) # 1^T * (I-A) ('-' to get max)
    
    # constraint
    lhs_ineq = np.concatenate((-(I-A), (I-A)))
    rhs_ineq = np.concatenate((np.zeros(len(fmax)), fmax))
       
    # boundaries 
    bnd = [0]*len(xmax)
    for i in range(0, len(xmax)):
        bnd[i] = (0, xmax[i])

    opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, bounds=bnd, method="revised simplex")
    maxF_x_sol = opt.x
    maxF_f_sol = (I-A).dot(maxF_x_sol) # f = (I-A)*x
    return  maxF_x_sol, maxF_f_sol

def LP(x, f, A, L, xmax, fmax):
    N = len(A.columns)

    f = f.sum(axis=1) 

    optimum_f_max = maxF(A, xmax, fmax) # maximizing final consumption
    maxF_x_sol, maxF_f_sol = optimum_f_max 
    optimum_x_max = maxX(L, xmax, fmax) # maximizing gross output
    maxX_f_sol, maxX_x_sol = optimum_x_max 
        
    x_frame = { 'x': x, 'xmin': 0, 'xmax': xmax, 'x.maxF': maxF_x_sol, 'x.maxX': maxX_x_sol }
    xvalues = pd.DataFrame(x_frame)
    
    f_frame = { 'f': f, 'fmin': 0, 'fmax': fmax, 'f.maxF': maxF_f_sol, 'f.maxX': maxX_f_sol }
    fvalues = pd.DataFrame(f_frame)
    
    xvalues.index = fvalues.index = A.columns
    
    output = namedtuple("output", ["xvalues", "fvalues"])
    return output(xvalues, fvalues)