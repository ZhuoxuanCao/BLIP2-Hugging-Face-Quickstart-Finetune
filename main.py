# main.py
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image

# processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")

# 使用 slow tokenizer
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", use_fast=False)

model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# # 加一句，把模型配置里的num_query_tokens手动写到processor上
# processor.num_query_tokens = getattr(model.config, 'num_query_tokens', 32)

img = Image.open("./image_test/BlueUp5.jpg")
prompt = "What is in the picture?"
# prompt = "There are several squares in the picture. If there is more than one, what is their relative relationship?"

inputs = processor(img, prompt, return_tensors="pt").to(device)
out = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(out[0], skip_special_tokens=True))
