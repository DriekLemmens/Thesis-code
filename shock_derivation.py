# The function "shocks" calculates the shock estimates for the Belgian industries
# outputs: shocks_raw = initial dataframe; supply_shocks_bel = supply shocks; demand_shocks_consumption = consumption demand shocks;
# demand_shocks_invest = investment demand shocks; demand_shocks_export = export demand shocks

import pandas as pd
import numpy as np
from collections import namedtuple

def shocks(shocks_raw): 
    # supply shock for Belgium: \epsilon^S = (1-RLI)(1-ESS)
    supply_shocks_bel = (1 - shocks_raw["remote_labor_index"]) * (1 - shocks_raw["essential_score_BE"])

    # CBO demand shocks 
    demand_shocks_consumption = shocks_raw["demand_shock_household"]

    # other demand shocks as in Pichler et al. (2020)
    demand_shocks_invest = np.repeat(0.1, len(demand_shocks_consumption))
    demand_shocks_export = np.repeat(0.1, len(demand_shocks_consumption))

    output = namedtuple("output", ["shocks_raw", "supply_shocks_bel",
                                   "demand_shocks_consumption", "demand_shocks_invest", "demand_shocks_export"])
    return output(shocks_raw, supply_shocks_bel,
                 demand_shocks_consumption, demand_shocks_invest, demand_shocks_export)
