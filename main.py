from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image

# ============================================================
# 方式1：原有 Hugging Face 在线加载（服务器无法联网时不可用）
# ============================================================
# processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", use_fast=False)
# model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-flan-t5-xl",
#     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
# )

# ============================================================
# 方式2：本地路径加载（推荐！完全脱机适配服务器）
# ============================================================
# 请修改此路径为你在服务器解压后的 snapshot 目录
local_model_path = "/gpfs/workdir/caozh/models--Salesforce--blip2-flan-t5-xl/snapshots/0eb0d3b46c14c1f8c7680bca2693baafdb90bb28"

processor = Blip2Processor.from_pretrained(local_model_path, use_fast=False)
model = Blip2ForConditionalGeneration.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

# ============================================================
# 推理部分
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

img = Image.open("./image_test/BlueUp5.jpg")
prompt = "What is in the picture?"
# prompt = "There are several squares in the picture. If there is more than one, what is their relative relationship?"

inputs = processor(img, prompt, return_tensors="pt").to(device)
out = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(out[0], skip_special_tokens=True))
