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
    # 1) 最优先：按提交规范读取 model.pth
    ckpt_path = os.path.join(base_dir, "model.pth")

    # 2) 如果 model.pth 不存在，再去 checkpoints 找“编号最大的 model_*.pth”
    if not os.path.exists(ckpt_path):
        ckpt_dir = os.path.join(base_dir, "checkpoints")
        if os.path.isdir(ckpt_dir):
            cands = [f for f in os.listdir(ckpt_dir) if re.match(r"model_\d+\.pth$", f)]
            if cands:
                cands.sort(key=lambda n: int(re.findall(r"\d+", n)[-1]))
                ckpt_path = os.path.join(ckpt_dir, cands[-1])

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "Cannot find model weights. Expected 'model.pth' or files like 'checkpoints/model_*.pth'."
        )

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()  # 评测阶段建议加上
    return model


__all__ = ['get_model']
