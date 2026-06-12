import torch.nn as nn
from typing import *
from utils import *
import numpy as np
import torch
import random


board_size = 12
bound = 5


# Load models using functions 'get_model' without passing any extra
# parameters, so that we can directly call get_model() in player.py and evaluator.py.


def get_opponent():
    # BEGIN YOUR CODE
    import os
    import re
    from submission import GobangModel

    opponent = GobangModel(board_size=board_size, bound=bound)

    base_dir = os.path.dirname(__file__)
    ckpt_dir = os.path.join(base_dir, 'checkpoints')
    preferred_old = [499, 999, 1499, 1999, 2499]
    preferred_paths = [
        os.path.join(ckpt_dir, f'model_{step}.pth')
        for step in preferred_old
        if os.path.exists(os.path.join(ckpt_dir, f'model_{step}.pth'))
    ]

    ckpt_path = os.path.join(base_dir, 'opponent.pth')

    # Prefer explicitly listed older checkpoints for the white (opponent) side.
    if preferred_paths:
        ckpt_path = random.choice(preferred_paths)
    elif not os.path.exists(ckpt_path):
        alt = os.path.join(base_dir, 'model.pth')
        if os.path.exists(alt):
            ckpt_path = alt
        else:
            if os.path.isdir(ckpt_dir):
                cands = [f for f in os.listdir(ckpt_dir) if f.startswith('model_') and f.endswith('.pth')]
                if len(cands) > 0:
                    def _num(name: str) -> int:
                        m = re.findall(r'(\d+)', name)
                        return int(m[-1]) if m else -1
                    cands = [name for name in cands if _num(name) < 2999] or cands
                    cands.sort(key=_num)
                    ckpt_path = os.path.join(ckpt_dir, random.choice(cands))

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "Cannot find opponent weights. Expected 'opponent.pth' (or fallback to model.pth/checkpoints)."
        )

    opponent.load_state_dict(torch.load(ckpt_path, map_location=device))
    opponent.to(device)
    return opponent
    # END YOUR CODE


__all__ = ['get_opponent']
