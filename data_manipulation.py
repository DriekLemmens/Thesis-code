# The function "data_manipulation" extracts the required inputs for the different models from the IO table
# outputs: Z = intermediate consumption matrix (domestic); A = technical coefficient matrix; L = Leontief inverse; 
# f = final demand vector; x = gross ouput vector; Z_imp = intermediate consumption matrix (import)
            
import pandas as pd
import numpy as np
from collections import namedtuple

def data_manipulation(country, data):
    # Intermediate consumption matrix (Z)
    Zrows = Zcols = list(range(0,data.columns.get_loc("ZA_T")+1))
    countryrows = countrycols = np.where(data.index.str.contains(country))[0]
    Z_country = data.iloc[countryrows, countrycols] #domestic Z
    Z_imp = data.iloc[Zrows , countrycols].drop(data.index[countryrows],axis=0) # all intermediate outputs from RoW to Belgium
    Z_imp = Z_imp.sum(axis=0)
    Z_exp = data.iloc[countryrows, Zcols].drop(data.columns[countrycols],axis=1) # all intermediate outputs from Belgium to RoW
    Z_exp = Z_exp.sum(axis=1)

    # Gross output (x)
    x = data.iloc[countryrows].sum(axis=1)

    # Final Demand (f)
    fcols = list(range(data.columns.get_loc("AR_P3_S13"), len(data.columns)))
    fcountrycols = list(range(data.columns.get_loc(country + "_P3_S13"), data.columns.get_loc(country + "_P5M")+1))
    frowcols = [i for i in fcols if i not in fcountrycols] 

    # final demand all disaggregated (row = Rest of World)
    f_country = data.iloc[countryrows, fcountrycols] # final demand from Belgium to Belgian industries
    f_row = data.iloc[countryrows, frowcols] # final demand from other countries to Belgian industries

    # final demand categories domestic and foreign (--> we split up f_row into different categories based on the columns)
    f_row_G = f_row.iloc[ : , 0::5].sum(axis=1) # P3_S13 (government consumption)
    f_row_house = f_row.iloc[ : , 1::5].sum(axis=1) # P3_S14 (household consumption)
    f_row_nonprofit = f_row.iloc[ : , 2::5].sum(axis=1) # P3_S15 (non-profit consumption)
    f_row_cons = f_row_house.add(f_row_nonprofit, fill_value=0) # S14 + S15
    f_row_invest = f_row.iloc[ : , 3::5].sum(axis=1) # P51G (gross fixed capital formation)
    f_row_inventory = f_row.iloc[ : , 4::5].sum(axis=1) # P5M (changes in inventories)

    # exporting data
    Z = Z_country
    f = pd.concat([f_country, Z_exp, f_row_G, f_row_cons, f_row_invest, f_row_inventory], axis=1)
    f.columns = ['P3_S13', 'P3_S14', 'P3_S15', 'P51G', 'P5M', 'Export', 'RoW_P3_S13', 'RoW_P3_S14+S15', 'RoW_P51G', 'RoW_P5M']

    # technical coeff matrix & Leontief inverse
    A = np.divide(Z, x)
    B = np.identity(len(x)) - A
    L = pd.DataFrame(np.linalg.inv(B), columns = A.index, index = A.index)

    output = namedtuple("output", ["Z", "A", "L", "f", "x", "Z_imp"]) 
    return output(Z, A, L, f, x, Z_imp)