# CLAUDE.md

## 项目概述
VLA instruction drift 检测与纠正框架。
从 prefill 阶段提取 instruction intent anchor，rollout 中监测
hidden state 与 anchor 的 cosine similarity，下降时施加 adaptive correction。

## 技术栈
- Python 3.10, PyTorch 2.1
- 模型：OpenVLA（LLaMA 架构）
- 环境：LIBERO（robotic manipulation）
- 集群：Aalto Triton（Slurm）

## 目录结构
- src/         核心方法实现
- scripts/     实验脚本 + Slurm job 脚本（run_*.sh）
- configs/     实验配置文件
- outputs/     实验结果（已 gitignore）

## 代码规范
- 用 argparse 处理命令行参数
- Slurm 脚本以 run_ 开头
- 模型路径、数据路径等通过参数传入，不要硬编码