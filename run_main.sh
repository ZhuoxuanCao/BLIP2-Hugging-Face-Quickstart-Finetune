#!/bin/bash
#SBATCH --job-name=blip2_infer         # 作业名称
#SBATCH --output=blip2_infer.out       # 标准输出
#SBATCH --error=blip2_infer.err        # 错误输出
#SBATCH --partition=gpu                # 分区（如需长时间推理可换 gpu）
#SBATCH --gres=gpu:1                   # 申请 1 块 GPU
#SBATCH --cpus-per-task=4              # 4核
#SBATCH --mem=16G                      # 24GB内存，可按需调整
#SBATCH --ntasks=1
#SBATCH --time=04:00:00                # 最多 1 小时，可按需求调整

# 加载 Anaconda/Miniconda 模块（如需）
# module load miniconda3/4.10.3/gcc-13.2.0

# 激活你的虚拟环境
# source activate /gpfs/workdir/caozh/envs/BLIP2

# 检查 CUDA 版本（可选）
# nvidia-smi

# 进入项目代码目录（确保在 main.py 所在目录）
cd /gpfs/workdir/caozh/BLIP2_HF_V2/

# 运行你的主程序
python main.py

# 可选：打印结束时间
# echo "Job finished at $(date)"
