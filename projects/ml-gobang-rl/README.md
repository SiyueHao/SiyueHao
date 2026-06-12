# Gobang RL Course Project / 机器学习课程大作业

This is my machine learning course project, centered on reinforcement learning for Gobang.

这是我的机器学习课程大作业，核心内容是围绕五子棋任务进行强化学习建模与实现。

## Project Structure / 项目结构

- `warmup/`
  Reinforcement learning warm-up tasks on Cliff Walking, including Monte Carlo, SARSA, Q-Learning, and Double Q-Learning.
- `gobang/`
  A residual CNN based actor-critic implementation for Gobang.
- `gobang_attention/`
  An attention-enhanced Gobang agent that combines convolutional local encoding with transformer-style global interaction.
- `report.pdf`
  The project report.

## Main Ideas / 主要思路

- 8-plane board encoding for richer state representation
- Legal-action masking to ensure valid move distributions
- Actor-critic training with reward shaping
- Symmetry-based augmentation and tactical features
- A comparison between CNN and attention-style backbones

## Files Worth Starting With / 推荐先看

- `gobang/submission.py`
- `gobang_attention/submission.py`
- `warmup/qlearning.py`
- `report.pdf`

## Notes / 说明

The project reflects both the warm-up reinforcement learning exercises and the final Gobang agent implementation.

这个目录同时保留了强化学习预热实验，以及最终五子棋智能体的主要实现。
