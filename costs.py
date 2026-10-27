from .config import COST_BPS
def total_cost_bps(market:str)->float:
    # Estimates are indicative engineering defaults, not broker quotes: spread/commission vary by venue and conditions.
    return float(COST_BPS.get(market,5.0))
