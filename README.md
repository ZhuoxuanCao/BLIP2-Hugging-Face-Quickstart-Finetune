## 环境搭建与依赖说明

本项目依赖最新版 Hugging Face Transformers 源码（支持BLIP-2等新模型），以及PyTorch GPU版本。
**强烈建议按如下步骤手动配置环境，否则可能出现包冲突或显存不可用等问题。**

### 1. 创建conda环境（推荐Python 3.9/3.10）

```bash
conda create -n blip2_env python=3.9
conda activate blip2_env
2. 手动安装GPU版本PyTorch
（根据服务器CUDA版本选择，以下以CUDA 11.8为例）

bash
复制
编辑
pip install torch==2.6.0+cu118 torchvision==0.22.0+cu118 torchaudio==2.7.0+cu118 --index-url https://download.pytorch.org/whl/cu118
如需其它CUDA版本，详见：https://pytorch.org/get-started/locally/

3. 安装源码版transformers与其余依赖
bash
复制
编辑
pip install -r requirements.txt
4. 检查环境
bash
复制
编辑
python -c "import torch; print(torch.cuda.is_available())"