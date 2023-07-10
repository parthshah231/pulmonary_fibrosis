from typing import List, Tuple, Dict

import numpy as np


def best_rect(m: int) -> Tuple[int, int]:
    low = int(np.floor(np.sqrt(m)))
    high = int(np.ceil(np.sqrt(m)))
    prods = [(low, low), (low, low + 1), (high, high), (high, high + 1)]
    for i, prod in enumerate(prods):
        if prod[0] * prod[1] >= m:
            return prod
    raise ValueError("Not possible!")
