import math

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
