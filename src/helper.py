import math
import torch

# ----- Helpers for extras -----

def month_norm(ym: str) -> float:
    # ym like "2025-09"
    m = int(ym.split("-")[1])
    # map Sep..Dec -> 0..1
    return (m - 9) / 3.0

def sincos_hour(h: int):
    ang = 2 * math.pi * (h / 24.0)
    return math.sin(ang), math.cos(ang)

def is_weekend(day_type: str) -> float:
    return 1.0 if "WEEKENDS" in day_type else 0.0

def get_edges(list_of_station):
    edges = []
    for i in range(len(list_of_station) - 1):
        u = list_of_station[i]
        v = list_of_station[i + 1]
        edges.append((u, v))
        edges.append((v, u))
    return edges


import torch
import numpy as np

# You need these from training time
# - stations: list of STATION_IDs length N (used only for sizing)
# - device: cuda/cpu
# - T_per_day: if each step is 1 hour, T_per_day = 24

def generate_extras(k: int, N: int, device, *,
                    month: int,
                    is_weekend: int,
                    steps_per_day: int = 24):
    """
    Returns extras_next with shape (N, 5) in the SAME order as X[:, 1:6].

    Assumes your extras columns are:
    [MONTH, IN_FLOW_NORM, OUT_FLOW_NORM, IS_WEEKEND, sin_time/cos_time???]
    BUT in your current code, extras are X[:,1:6] so it's 5 columns total.

    If your extras are actually:
    [MONTH, IN_FLOW_NORM, OUT_FLOW_NORM, IS_WEEKEND, sin_time]
    then this won't match.
    So below I implement the most likely intended extras:
    [MONTH, IS_WEEKEND, sin_time, cos_time, 0.0] (or a constant).
    Replace as needed.
    """

    hour = k % steps_per_day  # 0..23 if each step is 1 hour

    sin_time = np.sin(2 * np.pi * hour / steps_per_day)
    cos_time = np.cos(2 * np.pi * hour / steps_per_day)

    # Build per-node extras (same for every station, unless you have station-specific exogenous vars)
    month_col     = torch.full((N, 1), float(month), device=device)
    weekend_col   = torch.full((N, 1), float(is_weekend), device=device)
    sin_col       = torch.full((N, 1), float(sin_time), device=device)
    cos_col       = torch.full((N, 1), float(cos_time), device=device)

    # 5th column: use 0.0 as placeholder unless you have another feature
    extra5_col    = torch.zeros((N, 1), device=device)

    extras_next = torch.cat([month_col, weekend_col, sin_col, cos_col, extra5_col], dim=1)
    return extras_next
