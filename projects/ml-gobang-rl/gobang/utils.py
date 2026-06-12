import os
import random
import re
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import copy
from typing import *
from tqdm import tqdm
import torch

# 可选导入 wandb，如果未安装则跳过
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with 'pip install wandb' to enable experiment tracking.")

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Current device is {device}.")

SHAPE_SCORES = {
    "FIVE": 100000,
    "LIVE_FOUR": 10000,
    "RUSH_FOUR": 4000,
    "LIVE_THREE": 1500,
    "LIVE_TWO": 100,
}

PATTERNS = [
    ("FIVE", [r"11111"]),
    ("LIVE_FOUR", [r"011110"]),
    ("RUSH_FOUR", [r"011112", r"211110", r"10111", r"11011", r"11101", r"01111", r"11110"]),
    ("LIVE_THREE", [r"01110", r"010110", r"011010"]),
    ("LIVE_TWO", [r"00110", r"01100"]),
]

SHAPING_ALPHA = 0.02
DEFENSE_LAMBDA = 0.9
REWARD_CLIP = 1.5

DIRS = [(1, 0), (0, 1), (1, 1), (1, -1)]


def _in_bounds(x: int, y: int, n: int) -> bool:
    return 0 <= x < n and 0 <= y < n


def rot_coord(x: int, y: int, n: int, k: int) -> Tuple[int, int]:
    k = k % 4
    if k == 0:
        return x, y
    if k == 1:
        return n - 1 - y, x
    if k == 2:
        return n - 1 - x, n - 1 - y
    return y, n - 1 - x


def flip_coord_lr(x: int, y: int, n: int) -> Tuple[int, int]:
    return x, n - 1 - y


def apply_symmetry_board(board: np.ndarray, k: int, flip_lr: bool = False) -> np.ndarray:
    b = np.rot90(board, k, axes=(0, 1))
    if flip_lr:
        b = np.fliplr(b)
    return b


def apply_symmetry_action(x: int, y: int, n: int, k: int, flip_lr: bool = False) -> Tuple[int, int]:
    x2, y2 = rot_coord(x, y, n, k)
    if flip_lr:
        x2, y2 = flip_coord_lr(x2, y2, n)
    return x2, y2


def apply_symmetry_planes(planes: np.ndarray, k: int, flip_lr: bool = False) -> np.ndarray:
    out = np.rot90(planes, k, axes=(1, 2))
    if flip_lr:
        out = out[:, :, ::-1]
    return np.ascontiguousarray(out)


def augment_transition_8x(state: np.ndarray,
                           action_xy: Tuple[int, int],
                           reward: float,
                           next_state: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int], float, np.ndarray]]:
    n = state.shape[0]
    x, y = action_xy
    out = []
    for k in range(4):
        for flip in (False, True):
            s2 = np.ascontiguousarray(apply_symmetry_board(state, k, flip))
            ns2 = np.ascontiguousarray(apply_symmetry_board(next_state, k, flip))
            x2, y2 = apply_symmetry_action(x, y, n, k, flip)
            out.append((s2, (x2, y2), reward, ns2))
    return out


THREAT_PATTERNS = [
    ("FIVE", ["11111"]),
    ("LIVE4", ["011110"]),
    ("RUSH4", ["011112", "211110", "10111", "11011", "11101", "01111", "11110"]),
    ("LIVE3", ["01110", "010110", "011010"]),
    ("LIVE2", ["00110", "01100"]),
]
THREAT_RANK = {"LIVE2": 1, "LIVE3": 2, "RUSH4": 3, "LIVE4": 4, "FIVE": 5}


def classify_move_threat_pattern(board: np.ndarray, x: int, y: int, player: int) -> Tuple[Optional[str], int]:
    if board[x, y] != 0:
        return None, 0

    n = board.shape[0]
    best = None
    live3_dirs = 0

    for dx, dy in DIRS:
        chars = []
        for k in range(-5, 6):
            i, j = x + k * dx, y + k * dy
            if not _in_bounds(i, j, n):
                chars.append('2')
            else:
                v = board[i, j]
                if i == x and j == y:
                    v = player
                chars.append('0' if v == 0 else ('1' if v == player else '2'))
        s = ''.join(chars)
        center = 5

        dir_best = None
        for shape, pats in THREAT_PATTERNS:
            for pat in pats:
                start = 0
                while True:
                    idx = s.find(pat, start)
                    if idx == -1:
                        break
                    if idx <= center < idx + len(pat):
                        if dir_best is None or THREAT_RANK[shape] > THREAT_RANK[dir_best]:
                            dir_best = shape
                        if best is None or THREAT_RANK[shape] > THREAT_RANK[best]:
                            best = shape
                    start = idx + 1

        if dir_best == "LIVE3":
            live3_dirs += 1

    return best, live3_dirs


def board_to_8planes(board: np.ndarray, current_player: int) -> np.ndarray:
    n = board.shape[0]
    opp = 2 if current_player == 1 else 1

    planes = np.zeros((8, n, n), dtype=np.float32)
    planes[0] = (board == current_player).astype(np.float32)
    planes[1] = (board == opp).astype(np.float32)
    planes[2] = (board == 0).astype(np.float32)

    stones = np.argwhere(board != 0)
    if stones.size == 0:
        return planes
    candidate = np.zeros((n, n), dtype=bool)
    for sx, sy in stones:
        x0 = max(0, sx - 2)
        x1 = min(n, sx + 3)
        y0 = max(0, sy - 2)
        y1 = min(n, sy + 3)
        candidate[x0:x1, y0:y1] = True

    for x in range(n):
        for y in range(n):
            if board[x, y] != 0 or not candidate[x, y]:
                continue

            t_self, live3_dirs = classify_move_threat_pattern(board, x, y, current_player)
            if live3_dirs >= 2:
                planes[3, x, y] = 1.0
            elif t_self == "LIVE3":
                planes[3, x, y] = max(planes[3, x, y], 0.66)
            elif t_self == "LIVE2":
                planes[3, x, y] = max(planes[3, x, y], 0.33)

            if t_self == "RUSH4":
                planes[4, x, y] = 1.0
            elif t_self in ("LIVE4", "FIVE"):
                planes[5, x, y] = 1.0

            t_opp, opp_live3_dirs = classify_move_threat_pattern(board, x, y, opp)
            if opp_live3_dirs >= 2:
                planes[6, x, y] = 1.0
            elif t_opp == "LIVE3":
                planes[6, x, y] = max(planes[6, x, y], 0.66)
            elif t_opp == "LIVE2":
                planes[6, x, y] = max(planes[6, x, y], 0.33)

            if t_opp in ("RUSH4", "LIVE4", "FIVE"):
                planes[7, x, y] = 1.0

    return planes


def candidate_mask_from_board(board: np.ndarray, radius: int = 2) -> np.ndarray:
    n = board.shape[0]
    mask = np.zeros((n, n), dtype=bool)
    stones = np.argwhere(board != 0)
    if stones.size == 0:
        center = n // 2
        for i in range(n):
            for j in range(n):
                if abs(i - center) <= radius and abs(j - center) <= radius:
                    mask[i, j] = True
        return mask
    for sx, sy in stones:
        x0 = max(0, sx - radius)
        x1 = min(n, sx + radius + 1)
        y0 = max(0, sy - radius)
        y1 = min(n, sy + radius + 1)
        mask[x0:x1, y0:y1] = True
    return mask


def planes_to_board(planes: np.ndarray) -> np.ndarray:
    board = np.zeros(planes.shape[-2:], dtype=np.int8)
    board[planes[0] > 0.5] = 1
    board[planes[1] > 0.5] = 2
    return board


class UtilGobang:
    def __init__(self, board_size, bound):
        self.board_size, self.bound = board_size, bound
        self.board = np.zeros((board_size, board_size))
        self.window, self.canvas, self.cell_size = None, None, None
        self.action_space = [(i, j) for i in range(board_size) for j in range(board_size)]
        self.model, self.opponent = None, None

    def restart(self):
        self.board = np.zeros((self.board_size, self.board_size))
        self.action_space = [(i, j) for i in range(self.board_size) for j in range(self.board_size)]

    def draw_board(self, random_response, model, opponent):
        opponent_name = "random noise" if random_response else "training model itself"
        print(f"Playing process is being visualized with opponent {opponent_name}.")
        self.model, self.opponent = model, opponent
        self.window = tk.Tk()
        self.window.title("Gobang Board")
        self.canvas = tk.Canvas(self.window, width=400, height=400)
        self.canvas.pack()
        self.cell_size = 400 // self.board_size
        self.visualize_board(random_response)
        self.window.mainloop()

    def visualize_board(self, random_response):
        self.canvas.delete("all")
        color, end_up_gaming = self.update_board(random_response=random_response, learning=False)
        text = "Black wins." if color == 1 else "White wins." if color == 2 else "Tie." if color == 0 else None
        if text is not None:
            message = tk.Message(self.window, text=text, width=100)
            message.pack()
        for i in range(self.board_size):
            for j in range(self.board_size):
                x1 = i * self.cell_size
                y1 = j * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                if self.board[i][j] == 1:
                    self.canvas.create_oval(x1, y1, x2, y2, fill="black")
                elif self.board[i][j] == 2:
                    self.canvas.create_oval(x1, y1, x2, y2, fill="white")
        if end_up_gaming is True:
            print("Game ended.")
        else:
            self.window.after(1000, lambda: self.visualize_board(random_response))

    def judge_legal_position(self, x, y) -> bool:
        return 0 <= x < self.board_size and 0 <= y < self.board_size

    def count_max_connections_for_single_color(self, state, color) -> int:
        directions = [(1, 1), (1, 0), (0, 1), (1, -1)]
        max_connections = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                for direction_x, direction_y in directions:
                    current_pos_x, current_pos_y = i, j
                    current_connections = 0
                    while self.judge_legal_position(current_pos_x, current_pos_y):
                        if state[current_pos_x][current_pos_y] == color:
                            current_connections += 1
                        else:
                            break
                        current_pos_x += direction_x
                        current_pos_y += direction_y
                    max_connections = max(current_connections, max_connections)
        return max_connections

    def count_max_connections(self, state) -> Tuple[int, int]:
        return (self.count_max_connections_for_single_color(state, 1),
                self.count_max_connections_for_single_color(state, 2))

    @staticmethod
    def array_to_hashable(array):
        return tuple([tuple(r) for r in array])

    @staticmethod
    def hashable_to_array(hash_key):
        return np.array([list(r) for r in hash_key])

    def position_to_index(self, x: int, y: int) -> int:
        return x * self.board_size + y

    def index_to_position(self, index: int) -> Tuple[int, int]:
        x = index // self.board_size
        y = index - x * self.board_size
        return x, y

    @staticmethod
    def identity_transform(state: np.array):
        return np.array([
            [1 if r == 2 else 2 if r == 1 else 0 for r in row] for row in state
        ])

    def sample_action_and_response(self, random_response):
        raise NotImplementedError("Not Implemented!")

    def get_connection_and_reward(self, action, response):
        raise NotImplementedError("Not Implemented!")

    def get_next_state(self, action, response):
        raise NotImplementedError("Not Implemented!")

    def update_board(self, random_response, learning: bool = True, attempt: int = 8) -> Tuple[int, bool]:
        action_space = copy.deepcopy(self.action_space)
        (next_state_free_of_response, next_state,
         current_black_connection, current_white_connection,
         next_black_connection, next_white_connection, reward) = [None, None, None, None, None, None, None]
        for _ in range(attempt if learning else 1):
            self.action_space = copy.deepcopy(action_space)
            action, response = self.sample_action_and_response(random_response)
            (current_black_connection, current_white_connection,
             next_black_connection, next_white_connection, reward) = self.get_connection_and_reward(action, response)
            next_state = self.get_next_state(action, response)
            next_state_free_of_response = self.get_next_state(action, None)
        self.board = next_state_free_of_response if next_black_connection >= self.bound else next_state
        return ((1, True) if next_black_connection >= self.bound else
                (2, True) if next_white_connection >= self.bound else
                (0, True) if len(self.action_space) == 0 else
                (-1, False))

    def evaluate_agent_performance(self, random_response, model, opponent, episodes=1000):
        opponent_name = "random noise" if random_response else "training model itself"
        print(f"Start evaluating with opponent {opponent_name}.")
        self.model, self.opponent = model, opponent
        black_wins, white_wins, ties = 0, 0, 0
        for _ in tqdm(range(episodes)):
            self.restart()
            while True:
                color, end_up_gaming = self.update_board(learning=False, random_response=random_response)
                black_wins, white_wins, ties = ((black_wins, white_wins, ties) if end_up_gaming is False else
                                                (black_wins, white_wins, ties + 1) if color == 0 else
                                                (black_wins + 1, white_wins, ties) if color == 1 else
                                                (black_wins, white_wins + 1, ties))
                if end_up_gaming:
                    print(f"Black wins: {black_wins}, white wins: {white_wins}, and ties: {ties}.")
                    print(
                        f"The evaluated winning probability for the black pieces is "
                        f"{black_wins / (black_wins + white_wins + ties)}."
                    )
                    break
        self.restart()
        total_games = black_wins + white_wins + ties
        print(f"Evaluation finished. Black wins: {black_wins}, white wins: {white_wins}, and ties: {ties}.")
        print(
            f"The evaluated winning probability for the black pieces is "
            f"{black_wins / total_games}."
        )
        return {
            "black_wins": black_wins,
            "white_wins": white_wins,
            "ties": ties,
            "total_games": total_games,
            "black_win_rate": black_wins / total_games,
        }


class Gobang(UtilGobang):

    def __init__(self, board_size, bound, training):
        super().__init__(board_size=board_size, bound=bound)
        self.training = training
        self.model, self.opponent = None, None

    def get_all_lines(self, state: np.array) -> List[str]:
        n = self.board_size
        lines = []
        for r in range(n):
            lines.append("3" + "".join(map(str, state[r, :].astype(int))) + "3")
        for c in range(n):
            lines.append("3" + "".join(map(str, state[:, c].astype(int))) + "3")
        for offset in range(-n + 5, n - 4):
            diag = state.diagonal(offset)
            lines.append("3" + "".join(map(str, diag.astype(int))) + "3")
        flipped = np.fliplr(state)
        for offset in range(-n + 5, n - 4):
            diag = flipped.diagonal(offset)
            lines.append("3" + "".join(map(str, diag.astype(int))) + "3")
        return lines

    @staticmethod
    def _normalize_line_for_player(line: str, player_color: int) -> str:
        if player_color == 1:
            return line.replace('3', '2')
        line = line.replace('1', 't').replace('2', '1').replace('t', '2')
        return line.replace('3', '2')

    def evaluate_score(self, state: np.array, player_color: int) -> int:
        score = 0
        lines = self.get_all_lines(state)
        for raw in lines:
            line = self._normalize_line_for_player(raw, player_color)
            if re.search(PATTERNS[0][1][0], line):
                return SHAPE_SCORES["FIVE"]
            for shape_name, regs in PATTERNS[1:]:
                for rg in regs:
                    for _ in re.finditer(rg, line):
                        score += SHAPE_SCORES[shape_name]
                    line = re.sub(rg, lambda m: "X" * len(m.group(0)), line)
        return score

    def _potential(self, state: np.array) -> float:
        black_score = self.evaluate_score(state, 1)
        white_score = self.evaluate_score(state, 2)
        return black_score - DEFENSE_LAMBDA * white_score

    def _terminal_reward(self, state: np.array) -> float:
        black_conn, white_conn = self.count_max_connections(state)
        if black_conn >= self.bound:
            return 1.0
        if white_conn >= self.bound:
            return -1.0
        return 0.0

    def _shaped_reward(self, prev_state: np.array, next_state: np.array) -> float:
        phi_prev = self._potential(prev_state)
        phi_next = self._potential(next_state)
        delta_phi = (phi_next - phi_prev) / 5000.0
        shaped = SHAPING_ALPHA * delta_phi
        terminal = self._terminal_reward(next_state)
        return float(np.clip(terminal + shaped, -REWARD_CLIP, REWARD_CLIP))

    def get_next_state(self, action: Tuple[int, int, int], response: Tuple[int, int, int]) -> np.array:
        black, xb, yb = action
        next_state = copy.deepcopy(self.board)
        next_state[xb][yb] = black

        if response is not None:
            white, x_white, y_white = response
            next_state[x_white][y_white] = white
        return next_state

    def sample_response(self, random_response, x, y) -> Union[Tuple[int, int, int], None]:
        if self.action_space:
            state = self.identity_transform(self.board)
            state[x][y] = 2
            planes = board_to_8planes(state, 1)
            with torch.no_grad():
                policy = self.opponent.actor(planes)[0]
            legal_mask = torch.from_numpy(planes[2].reshape(-1) > 0).to(device)
            cand_mask = torch.from_numpy(candidate_mask_from_board(state).reshape(-1)).to(device)
            if random_response:
                policy = torch.where(policy > 0, torch.ones_like(policy), torch.zeros_like(policy))
            combined_mask = legal_mask & cand_mask
            policy = policy * combined_mask.to(policy.dtype)
            policy = policy / (policy.sum() + 1e-8)
            if policy.sum().item() <= 0:
                policy = policy * legal_mask.to(policy.dtype)
                policy = policy / (policy.sum() + 1e-8)
            if random_response:
                action = torch.multinomial(policy, 1).item()
            else:
                action = torch.argmax(policy).item()
            n = state.shape[0]
            x_, y_ = _index_to_position(n, action)
            if (x_, y_) in self.action_space:
                self.action_space.remove((x_, y_))
                return 2, x_, y_
            return None
        else:
            return None

    def get_connection_and_reward(self, action: Tuple[int, int, int],
                                  response: Tuple[int, int, int]) -> Tuple[int, int, int, int, float]:
        next_state = self.get_next_state(action, response)
        black_1, white_1 = self.count_max_connections(self.board)
        black_2, white_2 = self.count_max_connections(next_state)
        reward = self._shaped_reward(self.board, next_state)
        return black_1, white_1, black_2, white_2, reward

    def sample_action_and_response(self, random_response) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        state = self.board
        planes = board_to_8planes(state, 1)
        with torch.no_grad():
            policy = self.model.actor(planes)[0]
        n = state.shape[0]
        legal_mask = torch.from_numpy(planes[2].reshape(-1) > 0).to(device)
        cand_mask = torch.from_numpy(candidate_mask_from_board(state).reshape(-1)).to(device)
        combined_mask = legal_mask & cand_mask
        policy = policy * combined_mask.to(policy.dtype)
        policy = policy / (policy.sum() + 1e-8)
        if policy.sum().item() <= 0:
            policy = policy * legal_mask.to(policy.dtype)
            policy = policy / (policy.sum() + 1e-8)
        if random_response:
            action = torch.multinomial(policy, 1).item()
        else:
            action = torch.argmax(policy).item()
        x, y = _index_to_position(n, action)
        self.action_space.remove((x, y))
        return (1, x, y), self.sample_response(random_response, x, y)


def _position_to_index(board_size, x: int, y: int) -> int:
    return int(x * board_size + y)


def _index_to_position(board_size, index: int) -> Tuple[int, int]:
    x = index // board_size
    y = index - x * board_size
    return x, y


def _sample_response(chessboard, actor, state_white_view, epsilon=0.0):
    planes = board_to_8planes(state_white_view, 1)
    with torch.no_grad():
        policy = actor(planes)[0]
    if np.random.rand() < epsilon:
        noise = torch.distributions.Dirichlet(torch.full_like(policy, 0.3)).sample()
        policy = 0.8 * policy + 0.2 * noise
    legal_mask = torch.from_numpy(planes[2].reshape(-1) > 0).to(device)
    cand_mask = torch.from_numpy(candidate_mask_from_board(state_white_view).reshape(-1)).to(device)
    combined_mask = legal_mask & cand_mask
    policy = policy * combined_mask.to(policy.dtype)
    policy = policy / (policy.sum() + 1e-8)
    if policy.sum().item() <= 0:
        policy = policy * legal_mask.to(policy.dtype)
        policy = policy / (policy.sum() + 1e-8)
    if policy.sum().item() <= 0:
        fallback_mask = combined_mask if combined_mask.any() else legal_mask
        idx = torch.nonzero(fallback_mask, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            return None
        action = idx[torch.randint(0, idx.numel(), (1,), device=idx.device)].item()
        n = state_white_view.shape[0]
        x_, y_ = _index_to_position(n, action)
        if (x_, y_) in chessboard.action_space:
            chessboard.action_space.remove((x_, y_))
        return 2, x_, y_
    n = state_white_view.shape[0]
    action = torch.multinomial(policy, 1).item()
    x_, y_ = _index_to_position(n, action)
    if (x_, y_) in chessboard.action_space:
        chessboard.action_space.remove((x_, y_))
    return 2, x_, y_


def track_loss(actor_records, critic_records, entropy):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 20))

    ax1.plot(actor_records, label='Actor Loss', color='green')
    ax1.set_title('Actor Loss Tracking')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(critic_records, label='Critic Loss', color='red')
    ax2.set_title('Critic Loss Tracking')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    ax3.plot(entropy, label='Policy Entropy', color='blue')
    ax3.set_title('Policy Entropy Tracking')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Entropy')
    ax3.legend()
    ax3.grid(True)
    ax3.figure.savefig("loss_tracker.png")
    plt.close()


def _sample_action_and_response(chessboard, actor, state, epsilon=0.0):
    planes = board_to_8planes(state, 1)
    with torch.no_grad():
        policy = actor(planes)[0]
    if np.random.rand() < epsilon:
        noise = torch.distributions.Dirichlet(torch.full_like(policy, 0.3)).sample()
        policy = 0.8 * policy + 0.2 * noise
    legal_mask = torch.from_numpy(planes[2].reshape(-1) > 0).to(device)
    cand_mask = torch.from_numpy(candidate_mask_from_board(state).reshape(-1)).to(device)
    combined_mask = legal_mask & cand_mask
    policy = policy * combined_mask.to(policy.dtype)
    policy = policy / (policy.sum() + 1e-8)
    if policy.sum().item() <= 0:
        policy = policy * legal_mask.to(policy.dtype)
        policy = policy / (policy.sum() + 1e-8)
    if policy.sum().item() <= 0:
        fallback_mask = combined_mask if combined_mask.any() else legal_mask
        idx = torch.nonzero(fallback_mask, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            return (1, 0, 0), None
        action_idx = idx[torch.randint(0, idx.numel(), (1,), device=idx.device)].item()
        n = state.shape[0]
        x, y = _index_to_position(n, action_idx)
        if (x, y) in chessboard.action_space:
            chessboard.action_space.remove((x, y))
        return (1, x, y), None
    n = state.shape[0]
    action_idx = torch.multinomial(policy, 1).item()
    x, y = _index_to_position(n, action_idx)
    if (x, y) in chessboard.action_space:
        chessboard.action_space.remove((x, y))
    action = (1, x, y)

    state_after_black = _get_next_state(state, action, None)
    if len(np.nonzero(state_after_black == 0)[0]) == 0:
        return action, None

    state_white_view = chessboard.identity_transform(state_after_black)
    response = _sample_response(chessboard, actor, state_white_view, epsilon)
    return action, response


def _get_next_state(state, action, response):
    black, xb, yb = action
    next_state = copy.deepcopy(state)
    next_state[xb][yb] = black
    if response is not None:
        white, x_white, y_white = response
        next_state[x_white][y_white] = white
    return next_state


def train_model(model, num_episodes=1000, checkpoint=1000, gamma=0.99, start_episode=0, output_dir="checkpoints"):
    chess_board = Gobang(board_size=model.board_size, bound=model.bound, training=True)
    actor_records, critic_records, entropy_records = [], [], []
    for _ in range(start_episode, start_episode + num_episodes):
        states, actions, rewards, next_states = [[] for _ in range(4)]
        chess_board.restart()
        for count in range(chess_board.board_size ** 2 // 2 + 1):
            progress = _ / max(1, start_episode + num_episodes - 1)
            p = min(1.0, progress / 0.3)
            eps_min = 0.02
            curr_epsilon = eps_min + (0.20 - eps_min) * (1 - p)
            state = chess_board.board.copy()
            action, response = _sample_action_and_response(chess_board, model.actor, state, epsilon=curr_epsilon)
            state_after_black = _get_next_state(state, action, None)
            black_after, white_after = chess_board.count_max_connections(state_after_black)
            next_state = _get_next_state(state, action, response)
            black_2, white_2 = chess_board.count_max_connections(next_state)

            stop = True if (black_2 >= model.bound or white_2 >= model.bound
                            or len(np.nonzero(next_state == 0)[0]) == 0) else False

            if black_after >= model.bound:
                next_state = _get_next_state(state, action, None)
                black_2, white_2 = black_after, white_after
                response = None

            reward_black = chess_board._shaped_reward(state, state_after_black)
            next_state_black_view = chess_board.identity_transform(state_after_black)
            planes_state = board_to_8planes(state, 1)
            planes_next = board_to_8planes(next_state_black_view, 1)
            n = state.shape[0]
            for k in range(4):
                for flip in (False, True):
                    ps2 = apply_symmetry_planes(planes_state, k, flip)
                    pns2 = apply_symmetry_planes(planes_next, k, flip)
                    x2, y2 = apply_symmetry_action(action[1], action[2], n, k, flip)
                    states.append(ps2)
                    actions.append([x2, y2])
                    rewards.append(reward_black)
                    next_states.append(pns2)

            if response is not None and not (black_after >= model.bound):
                state_white_view = chess_board.identity_transform(state_after_black)
                next_state_white_view = chess_board.identity_transform(next_state)
                reward_white = chess_board._shaped_reward(state_white_view, next_state_white_view)
                next_state_white_next_turn = chess_board.identity_transform(next_state_white_view)
                planes_state_w = board_to_8planes(state_white_view, 1)
                planes_next_w = board_to_8planes(next_state_white_next_turn, 1)
                n = state_white_view.shape[0]
                for k in range(4):
                    for flip in (False, True):
                        ps2 = apply_symmetry_planes(planes_state_w, k, flip)
                        pns2 = apply_symmetry_planes(planes_next_w, k, flip)
                        x2, y2 = apply_symmetry_action(response[1], response[2], n, k, flip)
                        states.append(ps2)
                        actions.append([x2, y2])
                        rewards.append(reward_white)
                        next_states.append(pns2)
            chess_board.board = next_state
            if stop:
                break

        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(device)
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32)).to(device)
        actions = torch.from_numpy(np.array(actions, dtype=np.int64)).to(device)
        next_states_np = np.array(next_states, dtype=np.float32)

        self_plane3 = states[:, 3]
        self_plane4 = states[:, 4]
        self_plane5 = states[:, 5]
        opp_plane6 = states[:, 6]
        opp_plane7 = states[:, 7]

        self_fork = (self_plane3 == 1.0).any(dim=(1, 2))
        self_live3 = (self_plane3 >= 0.66).any(dim=(1, 2))
        self_live2 = (self_plane3 >= 0.33).any(dim=(1, 2))
        self_rush4 = (self_plane4 > 0).any(dim=(1, 2))
        self_live4 = (self_plane5 > 0).any(dim=(1, 2))

        opp_fork = (opp_plane6 == 1.0).any(dim=(1, 2))
        opp_live3 = (opp_plane6 >= 0.66).any(dim=(1, 2))
        opp_threat = (opp_plane7 > 0).any(dim=(1, 2))

        tactical_targets = torch.stack(
            [
                self_fork,
                self_live3,
                self_live2,
                self_rush4,
                self_live4,
                opp_fork,
                opp_live3,
                opp_threat,
            ],
            dim=1,
        ).to(torch.float32)
        policy, tactical_logits = model.actor.forward_with_tactical(states)
        qs = model.critic(states, actions)

        next_qs = torch.zeros(len(next_states_np), device=device)
        terminal_mask = np.zeros(len(next_states_np), dtype=bool)
        for i, ns in enumerate(next_states_np):
            board_ns = planes_to_board(ns)
            black_conn, white_conn = chess_board.count_max_connections(board_ns)
            if black_conn >= model.bound or white_conn >= model.bound or np.count_nonzero(ns[2] == 1) == 0:
                terminal_mask[i] = True

        non_terminal_indices = np.flatnonzero(~terminal_mask)
        if non_terminal_indices.size > 0:
            next_states_tensor = torch.from_numpy(next_states_np[non_terminal_indices]).to(device)
            with torch.no_grad():
                next_policy = model.actor(next_states_tensor)
                q_all = model.critic(next_states_tensor, action=None)

            empty_plane = next_states_tensor[:, 2]
            legal_mask = (empty_plane.reshape(empty_plane.shape[0], -1) > 0)
            nst_np = next_states_tensor.detach().cpu().numpy()
            cand_mask_np = np.stack([
                candidate_mask_from_board(planes_to_board(ns)).reshape(-1)
                for ns in nst_np
            ])
            cand_mask = torch.from_numpy(cand_mask_np).to(device)
            combined_mask = legal_mask & cand_mask
            masked_policy = next_policy * combined_mask.to(torch.float32)
            masked_policy = masked_policy / (masked_policy.sum(dim=1, keepdim=True) + 1e-8)
            q_all = q_all * combined_mask.to(torch.float32)
            next_qs_vals = torch.sum(masked_policy * q_all, dim=1)
            next_qs[non_terminal_indices] = next_qs_vals

        B = states.shape[0]
        empty_plane = states[:, 2]
        legal_mask = (empty_plane.reshape(B, -1) > 0)
        states_np = states.detach().cpu().numpy()
        cand_mask_np = np.stack([
            candidate_mask_from_board(planes_to_board(s)).reshape(-1)
            for s in states_np
        ])
        cand_mask = torch.from_numpy(cand_mask_np).to(device)
        combined_mask = legal_mask & cand_mask
        policy_m = policy * combined_mask.to(torch.float32)
        policy_m = policy_m / (policy_m.sum(dim=1, keepdim=True) + 1e-8)
        entropy = -float(torch.mean(torch.sum(policy_m * torch.log(policy_m + 1e-6), dim=1)))
        entropy_records.append(entropy)

        p = min(1.0, progress / 0.3)
        ent_min = 0.003
        entropy_coef = ent_min + (0.03 - ent_min) * (1 - p)
        tactical_weight = 0.2 * (1 - progress)

        global SHAPING_ALPHA, DEFENSE_LAMBDA
        SHAPING_ALPHA = 0.06 * (1 - progress) + 0.01 * progress
        DEFENSE_LAMBDA = 1.1 * (1 - progress) + 0.9 * progress
        with torch.no_grad():
            q_all = model.critic(states, action=None)
        actor_loss, critic_loss = model.optimize(
            policy,
            qs,
            actions,
            rewards,
            next_qs,
            gamma,
            entropy_coef=entropy_coef,
            tactical_logits=tactical_logits,
            tactical_targets=tactical_targets,
            tactical_weight=tactical_weight,
            q_all=q_all,
        )
        actor_records.append(float(actor_loss))
        critic_records.append(float(critic_loss))
        
        if WANDB_AVAILABLE and getattr(wandb, "run", None) is not None:
            wandb.log({
                "episode": _,
                "actor_loss": float(actor_loss),
                "critic_loss": float(critic_loss),
                "entropy": entropy,
                "actor_loss_neg": -float(actor_loss),
            })
        
        print(
            f"Episode {_} / {start_episode + num_episodes}: Actor Loss {-actor_loss}, Critic Loss "
            f"{critic_loss}.")
        if (_ + 1) % 10 == 0:
            try:
                track_loss(actor_records, critic_records, entropy_records)
            except Exception as e:
                print(e)
        if (_ + 1) % checkpoint == 0:
            ckpt_dir = os.path.abspath(output_dir)
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_{_}.pth"))


__all__ = ['_position_to_index', '_index_to_position', '_sample_response', 'train_model',
           '_sample_action_and_response', '_get_next_state', 'UtilGobang', 'Gobang', 'device',
           'board_to_8planes', 'classify_move_threat_pattern']
