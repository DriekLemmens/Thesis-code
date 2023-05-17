# The function "algo_prop" calculates the post-shock values for x and f for the proportional rationing scheme (prioritize=="no") or for the mixed rationing scheme (prioritize = "firms")
# outputs: d = post-shock x; f = post-shock f; i = number of iterations until convergence; 
# r = size of bottlenecks; s = largest input bottleneck 
# The function "algo_ordered" calculates the post-shock values for x and f for the priority rationing scheme (ordering=="priority") or for the random rationing scheme (ordering = "random")
# outputs: d = post-shock x; f = post-shock f; i = number of iterations until convergence; s = largest input bottleneck 

import numpy as np
import pandas as pd
from collections import namedtuple

## (mixed) proportional rationing algorithm
def algo_prop(A, L, f, xmax, prioritize="no", TT=20, tol=1e-7):  
    # set of suppliers
    l = []
    for k in range(0, len(A.columns)):
        l.append(np.where(A.iloc[:,k]>0))
    
    fnew = pd.DataFrame(f)
    fnew.index = L.index
    xmax.index = L.index
    s = pd.DataFrame(np.full([len(A), TT], np.nan))    
    o = r = d = None
    
    # main loop
    for i in range(1,TT):
        # updated aggregate demand
        d = pd.DataFrame(pd.concat([d, pd.DataFrame(np.dot(L,fnew.iloc[: , i-1:i]), index = L.index)], axis=1)) # d = L*f
        # updated intermediate demand
        o = pd.DataFrame(np.dot(A, d.iloc[: , i-1:i]), index = L.index)#.sum(axis=1) 
        # internal production constraint 
        if prioritize == "no": # proportional scheme
            xmax.index = L.index
            r = pd.concat([r, xmax/d.iloc[: , i-1:i].squeeze()], axis=1)
        elif prioritize == "firms": # mixed scheme
            xmax.index = L.index
            r = pd.concat([r, xmax/o.squeeze()], axis=1)
            if any(o==0):
                r[o==0] = 1
        
        # determine minimum bottleneck
        for k in range(0, len(l)):
            s.iloc[k,i-1] = pd.concat([r.iloc[l[k]].iloc[ : , i-1:i],pd.Series(1)], ignore_index=True, axis=0 ).squeeze().min()

        # updated final demand
        df1 = np.multiply(s.iloc[: , i-1:i], np.asarray(d.iloc[ : , i-1:i])).squeeze()
        df2 = xmax
        df1.reset_index(drop=True, inplace=True)
        df2.reset_index(drop=True, inplace=True)
        x_updated = pd.concat([df1, df2], axis = 1).min(axis=1)

        df3 = pd.DataFrame(np.dot(A, np.multiply(s.iloc[: , i-1:i], np.asarray(d.iloc[ : , i-1:i]))))
        df4 = pd.DataFrame(x_updated).subtract(df3)
        df4['new'] = 0
        df4.reset_index(drop=True, inplace=True)
        f.reset_index(drop=True, inplace=True)
        f_updated = pd.concat([f.squeeze(), df4.max(axis=1)], axis = 1).min(axis=1)

        fnew.reset_index(drop=True, inplace=True)
        f_updated.reset_index(drop=True, inplace=True)

        fnew = pd.concat([fnew, f_updated], axis=1)
        
        # stopping criteria
        if i > 1:
            if all(d.iloc[ : , i-1:i] == 0) or (all((d.iloc[ : , i-1:i]-d.iloc[ : , i-2:i-1] < tol)[0]) and all((d.iloc[ : , i-1:i]-d.iloc[ : , i-2:i-1] > -tol)[0]) and all((d.iloc[ : , i-1:i] - np.dot(A, d.iloc[ : , i-1:i]) >= -tol)[0])):
                break
        if i == TT:
            print("Warning: algorithm did not converge")
    
    s = s.iloc[ : , 0:i]
    
    output = namedtuple("output", ["d", "f", "i", "r", "s"])
    return output(d, fnew, i, r, s)

## Priority rationing (largest first) / Random rationing
def algo_ordered(A, L, f, xmax, ordering="priority", TT=30, tol=1e-7):
    N = len(A.index)
    
    # set of suppliers
    l = []
    for k in range(0, len(A.columns)):
        l.append(np.where(A.iloc[:,k]>0))
    
    d = np.dot(L, f)
    fnew = pd.DataFrame(f)
    fnew.index = L.index
    xmax.index = L.index
    O = pd.DataFrame(np.multiply(A.values, d.squeeze()))
    s = pd.DataFrame(np.full([len(A), TT], np.nan))    
    
    # determining customer order
    fixed_order = None
    for k in range(0, N):
        demand = O.iloc[k:k+1, :] # k^de rij
        
        if ordering == "priority":
            names_ordered = demand.T.sort_values(by=[k], ascending=False).index
        elif ordering == "random":
            names_ordered = demand.T.sample(frac=1).index
        fixed_order = pd.concat([fixed_order, pd.Series(names_ordered)], axis=1) #elke rij is één ordening
        fixed_order.columns = range(fixed_order.shape[1])
    # rationing matrix
    rk = None
    for k in range(0, N):
        rk = pd.concat([rk, rmat(xmax = xmax[k], demand = O.iloc[k:k+1, :], names_ordered = fixed_order.iloc[:, k:k+1].squeeze())], axis=0)
        
    rk.index = rk.columns
    
    # main loop
    for i in range(1,TT):

        #determine minimum bottleneck
        for k in range(0, len(l)):
            s.iloc[k,i-1] = pd.concat([rk.iloc[l[k]][k],pd.Series(1)], ignore_index=True, axis=0 ).squeeze().min()
        
        #updated final demand
        d = pd.DataFrame(d)
        df1 = np.multiply(s.iloc[: , i-1:i], np.asarray(d.iloc[ : , i-1:i])).squeeze()
        df2 = xmax
        df1.reset_index(drop=True, inplace=True)
        df2.reset_index(drop=True, inplace=True)
        x_updated = pd.concat([df1, df2], axis = 1).min(axis=1)

        df3 = pd.DataFrame(np.dot(A, np.multiply(s.iloc[: , i-1:i], np.asarray(d.iloc[ : , i-1:i]))))
        df4 = pd.DataFrame(x_updated).subtract(df3)
        df4['new'] = 0
        df4.reset_index(drop=True, inplace=True)
        f.reset_index(drop=True, inplace=True)
        f_updated = pd.concat([f.squeeze(), df4.max(axis=1)], axis = 1).min(axis=1)

        fnew.reset_index(drop=True, inplace=True)
        f_updated.reset_index(drop=True, inplace=True)

        fnew = pd.concat([fnew, f_updated], axis=1)
        
        # updated aggregate demand
        d = pd.DataFrame(pd.concat([d, pd.DataFrame(np.dot(L,fnew.iloc[: , i:i+1]))], axis=1)) # d = L*f
        
        # updated intermediate demand
        O = pd.DataFrame(np.dot(A, d.iloc[: , i:i+1]))#.sum(axis=1) 
        # stopping criteria
        if i > 0:
            if all(d.iloc[ : , i:i+1].values == 0) or (all((d.iloc[ : , i:i+1]-d.iloc[ : , i-1:i] < tol).values) and all((d.iloc[ : , i:i+1]-d.iloc[ : , i-1:i] > -tol).values) and all((d.iloc[ : , i:i+1] - np.dot(A, d.iloc[ : , i:i+1]) >= -tol).values)):
                break
        if i == TT:
            print("Warning: algorithm did not converge")
    
    s = s.iloc[ : , 0:i]

    output = namedtuple("output", ["d", "f", "i", "s"])
    return output(d, fnew, i, s)


## helper function for priority rationing 
def rmat(xmax, demand, names_ordered):
    N = len(demand.columns)
    demand.columns = range(demand.shape[1])
    
    demand = demand[names_ordered]
    cs = np.cumsum(demand, axis=1)
    remainder = pd.DataFrame(xmax - cs, columns = cs.columns).iloc[0] #kolom vector
        
    if all(remainder > 0): # all demand can be satisfied
        rvec = pd.Series(np.repeat(1,N))
        rvec.index = cs.columns
    elif all(remainder <= 0): # no demand can be satisfied
        rvec__ = xmax/demand.iloc[0,0]
        rvec = pd.concat([pd.Series(rvec__), pd.Series(np.repeat(0, N-1))])
        rvec.index = cs.columns
    else: # some demand can be satisfied
        idx = np.where(remainder<=0)[0][0] # index for partially met demand
        rvec__ = pd.concat([pd.Series(np.repeat(1, idx)), pd.Series((xmax - cs.iloc[0,idx-1])/demand.iloc[0,idx])])
        rvec = pd.concat([rvec__, pd.Series(np.repeat(0, N-idx-1))])
        rvec.index = cs.columns
    # return with original order
    rvec = pd.DataFrame(rvec[:54])
    rvec = rvec.sort_index()
    
    return rvec.T 
    

