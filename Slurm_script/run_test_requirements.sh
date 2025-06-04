#!/bin/bash
#SBATCH --job-name=test_env_gpu           # 作业名称
#SBATCH --output=test_env_gpu.out         # 标准输出
#SBATCH --error=test_env_gpu.err          # 错误输出
#SBATCH --partition=gpu_test              # GPU 分区，短测试可用
#SBATCH --gres=gpu:1                      # 申请 1 块 GPU
#SBATCH --cpus-per-task=4                 # 分配 CPU 核数
#SBATCH --mem=16G                         # 内存（可根据需要调整）
#SBATCH --ntasks=1                        # 任务数（一般为1）
#SBATCH --time=00:20:00                   # 最长运行时间

# 加载 Anaconda/Miniconda 模块（如需）
# module load miniconda3/4.10.3/gcc-13.2.0

# 激活你的虚拟环境
# source activate /gpfs/workdir/caozh/envs/BLIP2

# 检查 CUDA 版本（可选）
# nvidia-smi

# 进入你的代码目录
cd /gpfs/workdir/caozh/BLIP2-Hugging-Face-Quickstart-Finetune/

# 运行你的 CUDA 测试脚本
python test_requirements.py

# 可选：打印结束时间
# echo "Job finished at $(date)"