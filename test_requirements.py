# test_requirements.py
print("="*30)
print(" 环境依赖测试 - 开始 ")
print("="*30)

try:
    import torch
    print(f"torch            OK   (version: {torch.__version__})")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"torch            FAIL {e}")

try:
    import torchvision
    print(f"torchvision      OK   (version: {torchvision.__version__})")
except Exception as e:
    print(f"torchvision      FAIL {e}")

try:
    import torchaudio
    print(f"torchaudio       OK   (version: {torchaudio.__version__})")
except Exception as e:
    print(f"torchaudio       FAIL {e}")

try:
    import transformers
    print(f"transformers     OK   (version: {transformers.__version__})")
except Exception as e:
    print(f"transformers     FAIL {e}")

try:
    import datasets
    print(f"datasets         OK   (version: {datasets.__version__})")
except Exception as e:
    print(f"datasets         FAIL {e}")

try:
    import huggingface_hub
    print(f"huggingface_hub  OK   (version: {huggingface_hub.__version__})")
except Exception as e:
    print(f"huggingface_hub  FAIL {e}")

try:
    import tokenizers
    print(f"tokenizers       OK   (version: {tokenizers.__version__})")
except Exception as e:
    print(f"tokenizers       FAIL {e}")

try:
    import safetensors
    print(f"safetensors      OK   (version: {safetensors.__version__})")
except Exception as e:
    print(f"safetensors      FAIL {e}")

try:
    import sentencepiece
    print(f"sentencepiece    OK   (version: {sentencepiece.__version__})")
except Exception as e:
    print(f"sentencepiece    FAIL {e}")

try:
    import tensorboard
    print(f"tensorboard      OK   (version: {tensorboard.__version__})")
except Exception as e:
    print(f"tensorboard      FAIL {e}")

try:
    import accelerate
    print(f"accelerate       OK   (version: {accelerate.__version__})")
except Exception as e:
    print(f"accelerate       FAIL {e}")

try:
    import tqdm
    print(f"tqdm             OK   (version: {tqdm.__version__})")
except Exception as e:
    print(f"tqdm             FAIL {e}")

try:
    import numpy
    print(f"numpy            OK   (version: {numpy.__version__})")
except Exception as e:
    print(f"numpy            FAIL {e}")

try:
    import pandas
    print(f"pandas           OK   (version: {pandas.__version__})")
except Exception as e:
    print(f"pandas           FAIL {e}")

try:
    import scipy
    print(f"scipy            OK   (version: {scipy.__version__})")
except Exception as e:
    print(f"scipy            FAIL {e}")

try:
    import PIL
    print(f"Pillow (PIL)     OK   (version: {PIL.__version__})")
except Exception as e:
    print(f"Pillow (PIL)     FAIL {e}")

try:
    import sklearn
    print(f"scikit-learn     OK   (version: {sklearn.__version__})")
except Exception as e:
    print(f"scikit-learn     FAIL {e}")

print("="*30)
print(" 环境依赖测试 - 结束 ")
print("="*30)
