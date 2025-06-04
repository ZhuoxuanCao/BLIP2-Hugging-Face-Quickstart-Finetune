#!/bin/bash
#SBATCH --job-name=blip2_finetune            # 作业名称
#SBATCH --output=blip2_finetune.out          # 标准输出日志
#SBATCH --error=blip2_finetune.err           # 错误输出日志
#SBATCH --partition=gpu                      # 使用正式 GPU 分区，最长24h
#SBATCH --gres=gpu:1                         # 申请1块GPU，如多卡可改为gpu:2等
#SBATCH --cpus-per-task=8                    # 分配8核CPU（可根据需求调整）
#SBATCH --mem=24G                            # 24GB内存（大模型建议充裕）
#SBATCH --ntasks=1
#SBATCH --time=23:30:00                      # 最多23小时30分钟（方便提前收尾）
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=recco822@gmail.com


# 如需加载conda模块（根据你的环境决定是否加）
# module load miniconda3/4.10.3/gcc-13.2.0

# 激活你的虚拟环境
# source activate /gpfs/workdir/caozh/envs/BLIP2

# 打印节点显卡信息
# nvidia-smi

# 进入你的代码目录
cd /gpfs/workdir/caozh/BLIP2-Hugging-Face-Quickstart-Finetune/

# 运行微调脚本
python finetune_blip2_caption.py

# echo "Job finished at $(date)"
