import argparse
import numpy as np
import torch
import torch.nn as nn

from utils import *


# -----------------------------
# CLI (kept compatible)
# -----------------------------
parser = argparse.ArgumentParser(description='args', add_help=False)
parser.add_argument('--num_episodes', type=int, help='number of episodes')
parser.add_argument('--checkpoint', type=int, help='the interval of saving models')
parser.add_argument('--use_wandb', action='store_true', help='use wandb for experiment tracking (requires wandb installed)')
parser.add_argument('--wandb_project', type=str, default='gobang-rl-AI3002', help='wandb project name')
parser.add_argument('--output_dir', type=str, default='checkpoints', help='directory to save checkpoints')
parser.add_argument('--wandb_name', type=str, default=None, help='wandb run name')
args, _ = parser.parse_known_args()
num_episodes = args.num_episodes if args.num_episodes is not None else 5000
checkpoint = args.checkpoint if args.checkpoint is not None else 1000


# -----------------------------
# Attention backbone (Conv stem + Transformer)
# -----------------------------
class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer encoder block: (x + MHA(LN(x))) + MLP(LN(.))
    Designed for small-token board games (N*N tokens, N=12 => 144 tokens).
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout)
        self.drop1 = nn.Dropout(dropout)

        hidden = int(dim * mlp_ratio)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        y = self.ln1(x)
        y = y.transpose(0, 1)  # (L,B,C)
        y, _ = self.attn(y, y, y, need_weights=False)
        y = y.transpose(0, 1)  # (B,L,C)
        x = x + self.drop1(y)
        x = x + self.mlp(self.ln2(x))
        return x


class SpatialAttentionBackbone(nn.Module):
    """
    8-plane input -> conv stem (local patterns) -> tokens -> self-attention (global planning) -> feature map.
    """
    def __init__(self, board_size: int, in_planes: int = 8,
                 dim: int = 96, depth: int = 3, num_heads: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.board_size = board_size
        groups = 8 if dim % 8 == 0 else 4

        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=dim),
            nn.GELU(),
        )

        # Learnable absolute positional embeddings for N*N tokens
        self.pos = nn.Parameter(torch.zeros(1, board_size * board_size, dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        # small init for pos
        try:
            nn.init.trunc_normal_(self.pos, std=0.02)
        except AttributeError:
            nn.init.normal_(self.pos, std=0.02)

    def forward(self, planes: torch.Tensor) -> torch.Tensor:
        # planes: (B, 8, N, N)
        h = self.stem(planes)  # (B, C, N, N)
        B, C, N, _ = h.shape

        tokens = h.flatten(2).transpose(1, 2)  # (B, L=N*N, C)
        tokens = tokens + self.pos

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        h2 = tokens.transpose(1, 2).reshape(B, C, N, N).contiguous()
        return h2


def _ensure_planes(x) -> torch.Tensor:
    """
    Convert various input formats to planes tensor (B, 8, N, N).
    Supported:
      - (8, N, N) numpy/torch
      - (B, 8, N, N) numpy/torch
      - (N, N) board -> convert to planes
      - (B, 1, N, N) board -> convert to planes
    """
    if isinstance(x, torch.Tensor):
        t = x.to(device).to(torch.float32)
    else:
        t = torch.as_tensor(x, dtype=torch.float32, device=device)

    if t.dim() == 2:
        # (N, N) board
        board = t.detach().cpu().numpy()
        planes_np = board_to_8planes(board, 1)
        t = torch.from_numpy(planes_np).to(device)
        t = t.unsqueeze(0)
        return t

    if t.dim() == 3:
        # either (8,N,N) planes or (B,N,N) board batch (rare)
        if t.shape[0] == 8:
            return t.unsqueeze(0)
        else:
            # (B,N,N) board batch
            boards = t.detach().cpu().numpy()
            planes_np = np.stack([board_to_8planes(boards[i], 1) for i in range(boards.shape[0])], axis=0)
            return torch.from_numpy(planes_np).to(device)

    if t.dim() == 4:
        # either (B,8,N,N) planes or (B,1,N,N) board
        if t.shape[1] == 8:
            return t
        if t.shape[1] == 1:
            boards = t[:, 0].detach().cpu().numpy()
            planes_np = np.stack([board_to_8planes(boards[i], 1) for i in range(boards.shape[0])], axis=0)
            return torch.from_numpy(planes_np).to(device)
        raise ValueError(f"Unsupported tensor shape: {tuple(t.shape)}")
    raise ValueError(f"Unsupported input dim: {t.dim()}")


# -----------------------------
# Actor / Critic
# -----------------------------
class Actor(nn.Module):
    """
    Actor outputs a legal probability distribution over N*N moves.
    Input is usually the 8-plane representation from utils.board_to_8planes.
    """
    def __init__(self, board_size: int, lr: float = 1e-4):
        super().__init__()
        self.board_size = board_size

        # Attention backbone: strong global coupling, good for multi-threat planning.
        self.backbone = SpatialAttentionBackbone(
            board_size=board_size,
            in_planes=8,
            dim=96,
            depth=3,
            num_heads=4,
            mlp_ratio=4.0,
            dropout=0.0,
        )

        head_ch = 32
        head_groups = 8 if head_ch % 8 == 0 else 4
        self.policy_head = nn.Sequential(
            nn.Conv2d(96, head_ch, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=head_groups, num_channels=head_ch),
            nn.GELU(),
            nn.Conv2d(head_ch, 1, kernel_size=1, bias=True),  # logits map
        )

        # tactical multi-label head (kept compatible with utils.train_model)
        self.tactical_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, 8),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        planes = _ensure_planes(x)  # (B,8,N,N)
        feat = self.backbone(planes)  # (B,96,N,N)

        logits_map = self.policy_head(feat).squeeze(1)  # (B,N,N)
        logits = logits_map.reshape(logits_map.shape[0], -1)  # (B,N^2)

        # illegal mask from empty plane
        empty_plane = planes[:, 2]
        legal_mask = (empty_plane.reshape(empty_plane.shape[0], -1) > 0)
        masked_logits = logits.masked_fill(~legal_mask, -1e9)

        probs = torch.softmax(masked_logits, dim=1)
        probs = probs * legal_mask.to(torch.float32)
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
        return probs

    def forward_with_tactical(self, x):
        planes = _ensure_planes(x)
        feat = self.backbone(planes)

        logits_map = self.policy_head(feat).squeeze(1)
        logits = logits_map.reshape(logits_map.shape[0], -1)

        empty_plane = planes[:, 2]
        legal_mask = (empty_plane.reshape(empty_plane.shape[0], -1) > 0)
        masked_logits = logits.masked_fill(~legal_mask, -1e9)

        probs = torch.softmax(masked_logits, dim=1)
        probs = probs * legal_mask.to(torch.float32)
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)

        tactical_logits = self.tactical_head(feat)
        return probs, tactical_logits


class Critic(nn.Module):
    """
    Critic estimates Q(s,a). If action is None, returns Q(s, all a) with shape (B, N^2).
    """
    def __init__(self, board_size: int, lr: float = 1e-4):
        super().__init__()
        self.board_size = board_size

        self.backbone = SpatialAttentionBackbone(
            board_size=board_size,
            in_planes=8,
            dim=96,
            depth=3,
            num_heads=4,
            mlp_ratio=4.0,
            dropout=0.0,
        )

        head_ch = 32
        head_groups = 8 if head_ch % 8 == 0 else 4
        self.q_head = nn.Sequential(
            nn.Conv2d(96, head_ch, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=head_groups, num_channels=head_ch),
            nn.GELU(),
            nn.Conv2d(head_ch, 1, kernel_size=1, bias=True),  # Q map
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x, action=None):
        planes = _ensure_planes(x)
        feat = self.backbone(planes)
        q_map = self.q_head(feat).squeeze(1)  # (B,N,N)
        q_all = q_map.reshape(q_map.shape[0], -1)  # (B,N^2)

        if action is None:
            return q_all

        if isinstance(action, torch.Tensor):
            a = action.to(device)
        else:
            a = torch.as_tensor(action, device=device)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        a = a.to(torch.long)
        idx = a[:, 0] * self.board_size + a[:, 1]
        return q_all.gather(1, idx.unsqueeze(1)).squeeze(1)


class GobangModel(nn.Module):
    def __init__(self, board_size: int, bound: int):
        super().__init__()
        self.bound = bound
        self.board_size = board_size
        self.actor = Actor(board_size=board_size)
        self.critic = Critic(board_size=board_size)
        self.to(device)

    def forward(self, x, action):
        return self.actor(x), self.critic(x, action)

    def optimize(self, policy, qs, actions, rewards, next_qs, gamma,
                 eps: float = 1e-6, entropy_coef: float = 0.01,
                 tactical_logits=None, tactical_targets=None, tactical_weight: float = 0.2,
                 max_grad_norm: float = 1.0, q_all=None):
        # Critic target: note '-' is correct under your "always current player = 1" view transform.
        targets = (rewards - gamma * next_qs.detach()).detach()
        critic_loss = nn.MSELoss()(qs, targets)

        # Action -> index
        if isinstance(actions, torch.Tensor):
            a = actions.to(device)
        else:
            a = torch.as_tensor(actions, device=device)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        a = a.to(torch.long)
        indices = a[:, 0] * self.board_size + a[:, 1]

        aimed_policy = policy.gather(1, indices.unsqueeze(1)).squeeze(1)
        dist_entropy = -torch.mean(torch.sum(policy * torch.log(policy + 1e-6), dim=1))
        if q_all is not None:
            v = torch.sum(policy * q_all.detach(), dim=1)
            adv = (qs - v).detach()
        else:
            adv = (targets - qs).detach()
        policy_loss = -torch.mean(torch.log(aimed_policy + eps) * adv)
        actor_loss = policy_loss - entropy_coef * dist_entropy

        if tactical_logits is not None and tactical_targets is not None:
            tactical_loss = nn.BCEWithLogitsLoss()(tactical_logits, tactical_targets)
            actor_loss = actor_loss + tactical_weight * tactical_loss

        # Actor update
        self.actor.optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
        self.actor.optimizer.step()

        # Critic update
        self.critic.optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
        self.critic.optimizer.step()

        return actor_loss, critic_loss


if __name__ == "__main__":
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                config={
                    "num_episodes": num_episodes,
                    "checkpoint": checkpoint,
                    "board_size": 12,
                    "bound": 5,
                }
            )
            print("Wandb initialized successfully.")
        except ImportError:
            print("Warning: wandb not installed. Install with 'pip install wandb' to enable experiment tracking.")
            print("Continuing without wandb...")

    agent = GobangModel(board_size=12, bound=5).to(device)
    train_model(
        agent,
        num_episodes=num_episodes,
        checkpoint=checkpoint,
        output_dir=args.output_dir,
    )

    if args.use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
