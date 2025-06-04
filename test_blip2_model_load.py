import transformers
import os

print("Transformers version:", transformers.__version__)

# ============================================================
# 方式1：联网在线加载（服务器无法联网时不可用，适合本地调试）
# ============================================================
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
# model_name = "Salesforce/blip2-flan-t5-xl"
# print("Try loading processor from HF online...")
# processor = Blip2Processor.from_pretrained(model_name, use_fast=False)
# print("Processor loaded (online)")
# print("Try loading model from HF online...")
# model = Blip2ForConditionalGeneration.from_pretrained(model_name)
# print("Model loaded (online)")

# ============================================================
# 方式2：本地 snapshot 路径加载（推荐！适配服务器/离线）
# ============================================================
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# 请根据你服务器实际路径填写
local_model_path = "/gpfs/workdir/caozh/models--Salesforce--blip2-flan-t5-xl/snapshots/0eb0d3b46c14c1f8c7680bca2693baafdb90bb28"

if not os.path.exists(local_model_path):
    raise FileNotFoundError(f"Local model path does not exist: {local_model_path}")

print("Try loading processor from local path...")
processor = Blip2Processor.from_pretrained(local_model_path, use_fast=False)
print("Processor loaded (local)")

print("Try loading model from local path...")
model = Blip2ForConditionalGeneration.from_pretrained(local_model_path)
print("Model loaded (local)")
