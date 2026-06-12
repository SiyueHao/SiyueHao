from utils import *
import numpy as np
import torch
import torch.nn as nn
from typing import *
import sys
import argparse
import os
import re

parser = argparse.ArgumentParser(description='args', add_help=False)
parser.add_argument('--num_episodes', type=int, help='number of episodes')
parser.add_argument('--checkpoint', type=int, help='the interval of saving models')
parser.add_argument('--use_wandb', action='store_true', help='use wandb for experiment tracking (requires wandb installed)')
parser.add_argument('--wandb_project', type=str, default='gobang-rl-AI3002', help='wandb project name')
parser.add_argument('--wandb_name', type=str, default=None, help='wandb run name')
parser.add_argument('--resume', type=str, default=None, help='path to checkpoint .pth to resume training')
parser.add_argument('--output_dir', type=str, default='checkpoints', help='directory to save checkpoints')
args, _ = parser.parse_known_args()
num_episodes = args.num_episodes if args.num_episodes is not None else 5000
checkpoint = args.checkpoint if args.checkpoint is not None else 1000


class Actor(nn.Module):
    """
    The actor is responsible for generating dependable policies to maximize the cumulative reward as much as possible.
    It takes a batch of arrays shaped either (B, 1, N, N) or (N, N) as input, and outputs a tensor shaped (B, N ** 2)
    as the generated policy.
    """

    def __init__(self, board_size: int, lr=1e-4):
        super().__init__()
        self.board_size = board_size
        """
        # Define your NN structures here. Torch modules have to be registered during the initialization process.
        # For example, you can define CNN structures as follows:

        # self.conv_blocks = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=kernel_size, padding=padding),
        #     nn.MaxPool2d(kernel_size=kernel_size, padding=padding, stride=stride),
        #     nn.ReLU(),
        # )

        # Here, channels, kernel_size, padding, and stride are what we would call "Hyperparameters" in deep learning.

        # After convolution, you can flatten (nn.Flatten()) the hidden 2d-representation to obtain the corresponding
        # 1d-representation. Then, fully connected layers can be used to obtain a representation of n**2 dimensions,
        # with each digit indicating the "raw number of policy" (which has to be further constrained and modified
        # in the next step).

        # self.linear_blocks = nn.Sequential(
        #     nn.Linear(in_features=features, out_features=board_size ** 2),
        # )

        # After obtaining a representation of n**2 dimensions, you STILL NEED TO PERFORM ADDITIONAL PROCESSING,
        # including:
        # i) ensuring that all digits corresponding to illegal actions are set to 0 (!!!!!THE MOST IMPORTANT!!!!!);
        # ii) ensuring that the remaining digits satisfy the normalization condition (i.e., the sum of them is equal
        #     to 1).
        # In-place operations are strongly discouraged because they can lead to gradient calculation failures.
        # As an intelligent alternative, consider approaches that can avoid in-place modifications to achieve the goal.

        # You are also encouraged to explore other powerful models and experiment with different techniques,
        # such as using attention modules, different activation functions, or simply adjusting hyperparameter settings.
        """

        # BEGIN YOUR CODE
        # Hyperparameters (can be tuned)
        channels = 128
        res_blocks = 6  # 4~8 are all reasonable; 6 gives ~15 conv layers (incl. heads)
        # Use GroupNorm for stability under small/variable batch sizes.
        groups = 8 if channels % 8 == 0 else 4

        # Encode board as 8 planes: self/opp/empty + tactical threat planes
        self.stem = nn.Sequential(
            nn.Conv2d(8, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=channels),
            nn.ReLU(inplace=True),
        )

        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=groups, num_channels=channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=groups, num_channels=channels),
            )
            for _ in range(res_blocks)
        ])
        self.res_act = nn.ReLU(inplace=True)

        head_ch = 32
        head_groups = 8 if head_ch % 8 == 0 else 4
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, head_ch, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=head_groups, num_channels=head_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_ch, 1, kernel_size=1, bias=True),
        )
        self.tactical_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 8),
        )
        # END YOUR CODE

        # Define your optimizer here, which is responsible for calculating the gradients and performing optimizations.
        # The learning rate (lr) is another hyperparameter that needs to be determined in advance.
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

    def _encode(self, planes: torch.Tensor) -> torch.Tensor:
        h = self.stem(planes)
        for block in self.residual_blocks:
            residual = h
            h = block(h)
            h = self.res_act(h + residual)
        return h

    def forward(self, x: np.ndarray):
        if isinstance(x, torch.Tensor):
            output = x.to(device).to(torch.float32)
        else:
            output = torch.as_tensor(x, dtype=torch.float32, device=device)
        if output.dim() == 2:
            output = output.unsqueeze(0).unsqueeze(0)
        elif output.dim() == 3:
            if output.shape[0] == 8:
                output = output.unsqueeze(0)
            else:
                output = output.unsqueeze(1)

        # Further process and transform the data here. Ensure that the output is shaped (B, n ** 2).
        # We have already ensured that the shape of the raw input is unified to be (B, 1, N, N),
        # where B >= 1 represents the number of data in this batch, and N = n is exactly the size of the board.

        # You can continue processing the data here using the modules that were previously registered during the
        # initialization process. For example:

        # output = self.conv_blocks(output)
        # output = nn.Flatten()(output)
        # output = self.linear_blocks(output)

        # And the reminder AGAIN:

        # ****************************************
        # After obtaining a representation of n**2 dimensions, you STILL NEED TO PERFORM ADDITIONAL DATA PROCESSING,
        # including:
        # i) ensuring that all digits corresponding to illegal actions are set to 0 (!!!!!THE MOST IMPORTANT!!!!!);
        # ii) ensuring that the remaining digits satisfy the normalization condition (i.e., the sum of them is equal
        #     to 1).
        # In-place operations are strongly discouraged because they can lead to gradient calculation failures.
        # ****************************************

        # BEGIN YOUR CODE
        if output.dim() == 4 and output.shape[1] == 8:
            planes = output
        else:
            board = output[:, 0].detach().cpu().numpy()  # (B, N, N)
            planes_np = np.stack([board_to_8planes(board[i], 1) for i in range(board.shape[0])], axis=0)
            planes = torch.from_numpy(planes_np).to(device)

        # CNN-residual encoder
        h = self._encode(planes)

        # Policy head -> logits over N^2 actions
        logits = self.policy_head(h).squeeze(1)  # (B, N, N)
        logits = logits.reshape(logits.shape[0], -1)  # (B, N^2)

        # Constrain the policy: illegal actions -> 0, then renormalize.
        empty_plane = planes[:, 2]
        legal_mask = (empty_plane.reshape(empty_plane.shape[0], -1) > 0)  # bool mask, (B, N^2)
        masked_logits = logits.masked_fill(~legal_mask, -1e9)
        probs = torch.softmax(masked_logits, dim=1)

        # Hard mask + renormalize for strict legality.
        probs = probs * legal_mask.to(torch.float32)
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)

        output = probs
        # END YOUR CODE
        return output

    def forward_with_tactical(self, x: np.ndarray):
        if isinstance(x, torch.Tensor):
            output = x.to(device).to(torch.float32)
        else:
            output = torch.as_tensor(x, dtype=torch.float32, device=device)
        if output.dim() == 2:
            output = output.unsqueeze(0).unsqueeze(0)
        elif output.dim() == 3:
            if output.shape[0] == 8:
                output = output.unsqueeze(0)
            else:
                output = output.unsqueeze(1)

        if output.dim() == 4 and output.shape[1] == 8:
            planes = output
        else:
            board = output[:, 0].detach().cpu().numpy()
            planes_np = np.stack([board_to_8planes(board[i], 1) for i in range(board.shape[0])], axis=0)
            planes = torch.from_numpy(planes_np).to(device)

        h = self._encode(planes)
        logits = self.policy_head(h).squeeze(1)
        logits = logits.reshape(logits.shape[0], -1)

        empty_plane = planes[:, 2]
        legal_mask = (empty_plane.reshape(empty_plane.shape[0], -1) > 0)
        masked_logits = logits.masked_fill(~legal_mask, -1e9)
        probs = torch.softmax(masked_logits, dim=1)
        probs = probs * legal_mask.to(torch.float32)
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)

        tactical_logits = self.tactical_head(h)
        return probs, tactical_logits


class Critic(nn.Module):
    """
    The critic is responsible for generating dependable Q-values to fit the solution of Bellman Equations. It takes
    a batch of arrays (shaped either (B, 1, N, N) or (N, N)) and a batch of actions (shaped (B, 2)) as input, and
    outputs a tensor shaped (B, ) as the Q-values on the specified (s, a) pairs.

    For example, actions can be:
    [[0, 1],
     [2, 3],
     [5, 6]]
    which means that there are three actions leading the model to place the pieces on the coordinates (0, 1), (2, 3),
    and (5, 6), respectively. These actions correspond one-to-one with indices 0 * 12 + 1 = 1, 2 * 12 + 3 = 27,
    and 5 * 12 + 6 = 66, assuming n to be 12. You can easily transform a single action to the corresponding digit by
    using _position_to_index, or using _index_to_position vice versa.

    The main idea is that we first obtain a tensor shaped (B, N ** 2) as the Q-values for all possible actions given
    the unified state tensor shaped (B, 1, N, N), and then extract the Q-values corresponding to each action (i, j)
    from the entire Q-value tensor. (_position_to_index should be fully utilized to get the corresponding action indices).
    Finally, it returns a tensor of shape (B,) containing these Q-values.
    """

    def __init__(self, board_size: int, lr=1e-4):
        super().__init__()
        self.board_size = board_size
        # Define your NN structures here as the same. Torch modules have to be registered during the initialization
        # process.

        # BEGIN YOUR CODE
        # Hyperparameters (can be tuned)
        channels = 128
        res_blocks = 6
        groups = 8 if channels % 8 == 0 else 4

        # Encode board as 8 planes: self/opp/empty + tactical threat planes
        self.stem = nn.Sequential(
            nn.Conv2d(8, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=channels),
            nn.ReLU(inplace=True),
        )

        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=groups, num_channels=channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=groups, num_channels=channels),
            )
            for _ in range(res_blocks)
        ])
        self.res_act = nn.ReLU(inplace=True)

        head_ch = 32
        head_groups = 8 if head_ch % 8 == 0 else 4
        self.q_head = nn.Sequential(
            nn.Conv2d(channels, head_ch, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=head_groups, num_channels=head_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_ch, 1, kernel_size=1, bias=True),
        )
        # END YOUR CODE

        # Define your optimizer here, which is responsible for calculating the gradients and performing optimizations.
        # The learning rate (lr) is another hyperparameter that needs to be determined in advance.
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

    def forward(self, x: np.ndarray, action: Optional[np.ndarray] = None):
        if action is not None:
            if isinstance(action, torch.Tensor):
                a = action.to(device)
            else:
                a = torch.tensor(action).to(device)
            if a.dim() == 1:
                a = a.unsqueeze(0)
            a = a.to(torch.long)
            indices = a[:, 0] * self.board_size + a[:, 1]
        if isinstance(x, torch.Tensor):
            output = x.to(device).to(torch.float32)
        else:
            output = torch.as_tensor(x, dtype=torch.float32, device=device)
        if output.dim() == 2:
            output = output.unsqueeze(0).unsqueeze(0)
        elif output.dim() == 3:
            if output.shape[0] == 8:
                output = output.unsqueeze(0)
            else:
                output = output.unsqueeze(1)

        # BEGIN YOUR CODE
        if output.dim() == 4 and output.shape[1] == 8:
            planes = output
        else:
            board = output[:, 0].detach().cpu().numpy()  # (B, N, N)
            planes_np = np.stack([board_to_8planes(board[i], 1) for i in range(board.shape[0])], axis=0)
            planes = torch.from_numpy(planes_np).to(device)

        # CNN-residual encoder
        h = self.stem(planes)
        for block in self.residual_blocks:
            residual = h
            h = block(h)
            h = self.res_act(h + residual)

        # Q head -> Q-values over all actions, then gather Q(s,a) if needed.
        q_map = self.q_head(h).squeeze(1)  # (B, N, N)
        q_all = q_map.reshape(q_map.shape[0], -1)  # (B, N^2)

        if action is None:
            output = q_all
        else:
            indices = indices.to(torch.long)
            output = q_all.gather(1, indices.unsqueeze(1)).squeeze(1)
        # END YOUR CODE

        return output


class GobangModel(nn.Module):
    """
    The GobangModel class integrates the Actor and Critic classes for computation and training. Given state tensors "x"
    and action tensors "action", it directly outputs self.actor(x) and self.critic(x, action) as the policy and Q-values
    respectively.
    """

    def __init__(self, board_size: int, bound: int):
        super().__init__()
        self.bound = bound
        self.board_size = board_size

        """
        Register the actor and critic modules here. You do not need to further design the structures at this step.
        Feel free to add extra parameters in the __init__ method of either the Actor class or the Critic class for your 
        convenience, if necessary.
        """

        # BEGIN YOUR CODE
        self.actor = Actor(board_size=board_size)
        self.critic = Critic(board_size=board_size)
        # END YOUR CODE

        self.to(device)

    def forward(self, x, action):
        """
        Return the policy vector π(s) and Q-values Q(s, a) given state "x" and action "action".
        """
        return self.actor(x), self.critic(x, action)

    def optimize(self, policy, qs, actions, rewards, next_qs, gamma, eps=1e-6, entropy_coef=0.01,
                 tactical_logits=None, tactical_targets=None, tactical_weight=0.2, q_all=None):
        """
        This function calculates the loss for both the actor and critic.
        Using the obtained loss, we can apply optimization algorithms through actor.optimizer and critic.optimizer
        to either maximize the actor's actual objective or minimize the critic's loss.

        There are 3 bugs in the function "optimize" that prevent the model from executing optimizations correctly.
        Identify and debug all errors.
        """

        targets = (rewards - gamma * next_qs.detach()).detach()
        critic_loss = nn.MSELoss()(qs, targets)

        # Vectorized action -> index (avoid CPU/GPU mismatches)
        if isinstance(actions, torch.Tensor):
            a = actions.to(device)
        else:
            a = torch.tensor(actions).to(device)
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

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
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
    start_episode = 0
    if args.resume:
        if os.path.isabs(args.resume):
            resume_path = args.resume
            candidates = [resume_path]
        else:
            cwd_path = os.path.join(os.getcwd(), args.resume)
            ckpt_path = os.path.join(os.getcwd(), "checkpoints", args.resume)
            candidates = [cwd_path, ckpt_path]
        resume_path = None
        for p in candidates:
            if os.path.exists(p):
                resume_path = p
                break
        if resume_path:
            state = torch.load(resume_path, map_location=device)
            agent.load_state_dict(state)
            print(f"Resumed from checkpoint: {resume_path}")
            base = os.path.basename(resume_path)
            m = re.search(r"model_(\d+)\.pth", base)
            if m:
                start_episode = int(m.group(1)) + 1
        else:
            print(f"Warning: resume checkpoint not found. Tried: {', '.join(candidates)}")
    train_model(
        agent,
        num_episodes=num_episodes,
        checkpoint=checkpoint,
        start_episode=start_episode,
        output_dir=args.output_dir,
    )
    
    if args.use_wandb:
        try:
            import wandb
            wandb.finish()
        except:
            pass
