#!/bin/bash
#SBATCH --job-name=test_blip2_load         # 作业名称
#SBATCH --output=test_blip2_load.out       # 标准输出日志
#SBATCH --error=test_blip2_load.err        # 错误输出日志
#SBATCH --partition=gpu_test               # 测试分区，最长1小时，资源紧张用 gpu 分区
#SBATCH --gres=gpu:1                       # 申请1块GPU
#SBATCH --cpus-per-task=4                  # 分配4核CPU
#SBATCH --mem=16G                          # 16GB内存，足够用
#SBATCH --ntasks=1
#SBATCH --time=00:20:00                    # 最长20分钟（环境测试足够）

# 加载Miniconda模块（如有需要，可去掉）
# module load miniconda3/4.10.3/gcc-13.2.0

# 激活你的虚拟环境
# source activate /gpfs/workdir/caozh/envs/BLIP2

# 显示GPU信息，确认拿到节点
# nvidia-smi

# 进入你的项目目录
cd /gpfs/workdir/caozh/BLIP2-Hugging-Face-Quickstart-Finetune/

# 运行模型加载测试脚本
python test_blip2_model_load.py

# echo "Job finished at $(date)"
