import torch.nn as nn
from typing import *
from utils import *
import numpy as np
import torch


board_size = 12
bound = 5


# Load models using functions 'get_model' without passing any extra
# parameters, so that we can directly call get_model() in player.py and evaluator.py.


def get_model():
    import os
    import re
    from submission import GobangModel

    model = GobangModel(board_size=board_size, bound=bound)

    base_dir = os.path.dirname(__file__)
    preferred_latest = os.path.join(base_dir, 'checkpoints', 'model_2999.pth')
    ckpt_path = preferred_latest if os.path.exists(preferred_latest) else os.path.join(base_dir, 'model.pth')

    # Fallback: load latest checkpoint if model.pth does not exist.
    if not os.path.exists(ckpt_path):
        ckpt_dir = os.path.join(base_dir, 'checkpoints')
        if os.path.isdir(ckpt_dir):
            cands = [f for f in os.listdir(ckpt_dir) if f.startswith('model_') and f.endswith('.pth')]
            if len(cands) > 0:
                def _num(name: str) -> int:
                    m = re.findall(r'(\d+)', name)
                    return int(m[-1]) if m else -1
                cands.sort(key=_num)
                ckpt_path = os.path.join(ckpt_dir, cands[-1])

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "Cannot find model weights. Expected 'model.pth' or files like 'checkpoints/model_*.pth'."
        )

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    return model


__all__ = ['get_model']
