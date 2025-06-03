import transformers
print("Transformers version:", transformers.__version__)
from transformers import BlipProcessor, Blip2ForConditionalGeneration

model_name = "Salesforce/blip2-flan-t5-xl"
print("Try loading processor...")
processor = BlipProcessor.from_pretrained(model_name)
print("Processor loaded")
print("Try loading model...")
model = Blip2ForConditionalGeneration.from_pretrained(model_name)
print("Model loaded")

