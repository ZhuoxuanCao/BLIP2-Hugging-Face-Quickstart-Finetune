import os
import json
import torch
from PIL import Image
from datasets import Dataset
from transformers import (
    BlipProcessor,          # 保持你原来使用的 Processor
    Blip2ForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 兼容 HF 新 API 可能多出的参数
        if "num_items_in_batch" in inputs:
            inputs.pop("num_items_in_batch")
        return super().compute_loss(model, inputs, return_outputs=return_outputs)


def split_json(data_path, train_path, val_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    train_list, val_list = train_test_split(data_list, test_size=0.2, random_state=42)
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_list, f, ensure_ascii=False, indent=2)
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_list, f, ensure_ascii=False, indent=2)
    print(f"[数据集划分] 已生成 {train_path} 和 {val_path}")


def json_to_jsonl(json_path, jsonl_path):
    with open(json_path, 'r', encoding='utf-8') as fin, \
         open(jsonl_path, 'w', encoding='utf-8') as fout:
        data = json.load(fin)
        for item in data:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"[格式转换] 已生成 {jsonl_path}")


def preprocess_function(example, processor):
    image = Image.open(example["image"]).convert("RGB")
    prompt = ""  # 图像 caption 任务建议 prompt 为空
    inputs = processor(
        images=image,
        text=prompt,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )
    labels = processor.tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt"
    ).input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    inputs["labels"] = labels.squeeze(0)
    return inputs


def main():
    # ===== 1. 数据集划分：80% 训练，20% 验证 =====
    if not (os.path.exists('train_split.json') and os.path.exists('val_split.json')):
        split_json('annotations.json', 'train_split.json', 'val_split.json')
    else:
        print("[跳过划分] 已存在 train_split.json 和 val_split.json")

    # ===== 2. 转换为 jsonl =====
    if not (os.path.exists('train_split.jsonl') and os.path.exists('val_split.jsonl')):
        json_to_jsonl('train_split.json', 'train_split.jsonl')
        json_to_jsonl('val_split.json', 'val_split.jsonl')
    else:
        print("[跳过格式转换] 已存在 train_split.jsonl 和 val_split.jsonl")

    # ===== 3. 加载 jsonl 格式数据集 =====
    with open('train_split.jsonl', 'r', encoding='utf-8') as f:
        train_list = [json.loads(line) for line in f]
    with open('val_split.jsonl', 'r', encoding='utf-8') as f:
        val_list = [json.loads(line) for line in f]
    train_dataset = Dataset.from_list(train_list)
    eval_dataset  = Dataset.from_list(val_list)

    # ============================================================
    # 4. 加载模型和 Processor
    #    【保留两种方案：① 在线加载（已注释）② 本地 snapshot 路径加载（默认启用）】
    # ============================================================

    # ---------- 方案 ①：在线加载（服务器若无法联网将报错，保留作参考） ----------
    # model_name = "Salesforce/blip2-flan-t5-xl"
    # processor = BlipProcessor.from_pretrained(model_name)
    # model     = Blip2ForConditionalGeneration.from_pretrained(model_name)

    # ---------- 方案 ②：本地 snapshot 路径加载（推荐 / 服务器离线必用） ----------
    local_model_path = "/gpfs/workdir/caozh/models--Salesforce--blip2-flan-t5-xl/snapshots/0eb0d3b46c14c1f8c7680bca2693baafdb90bb28"
    processor = BlipProcessor.from_pretrained(local_model_path)
    model     = Blip2ForConditionalGeneration.from_pretrained(local_model_path)

    # ===== 5. 预处理数据 =====
    train_dataset = train_dataset.map(lambda ex: preprocess_function(ex, processor))
    eval_dataset  = eval_dataset.map(lambda ex: preprocess_function(ex, processor))

    # ===== 6. 训练参数设置 =====
    training_args = TrainingArguments(
        output_dir="./blip2_captioning",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        remove_unused_columns=False,
        overwrite_output_dir=True,
        dataloader_num_workers=2,
        seed=42,
        gradient_checkpointing=True,
        push_to_hub=False,
        report_to="tensorboard",
        logging_dir="./blip2_captioning/tensorboard"
    )

    # ===== 7. 构建 Trainer =====
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor,
    )

    # ===== 8. 启动训练 =====
    trainer.train()

    # ===== 9. 保存最终模型 =====
    trainer.save_model("./blip2_captioning/final_model")
    print("训练完成，模型已保存到 ./blip2_captioning/final_model")
    print("\n可使用以下命令实时监控训练过程：")
    print("tensorboard --logdir ./blip2_captioning/tensorboard")
    print("浏览器访问 http://localhost:6006")


if __name__ == "__main__":
    main()




# import os
# import json
# import torch
# from PIL import Image
# from datasets import load_dataset
# from datasets import Dataset
# from transformers import BlipProcessor, Blip2ForConditionalGeneration, TrainingArguments, Trainer
# from sklearn.model_selection import train_test_split
# from transformers import Trainer
#
# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         # 兼容HF新API多余参数
#         if "num_items_in_batch" in inputs:
#             inputs.pop("num_items_in_batch")
#         return super().compute_loss(model, inputs, return_outputs=return_outputs)
#
#
# def main():
#     # ===== 1. 数据集划分：80%训练，20%验证 =====
#     data_path = "annotations.json"
#     with open(data_path, 'r', encoding='utf-8') as f:
#         data_list = json.load(f)
#     train_list, val_list = train_test_split(data_list, test_size=0.2, random_state=42)
#     with open('train_split.json', 'w', encoding='utf-8') as f:
#         json.dump(train_list, f, ensure_ascii=False, indent=2)
#     with open('val_split.json', 'w', encoding='utf-8') as f:
#         json.dump(val_list, f, ensure_ascii=False, indent=2)
#
#     # ===== 2. 加载数据集 =====
#     with open('train_split.jsonl', 'r', encoding='utf-8') as f:
#         train_list = [json.loads(line) for line in f]
#     with open('val_split.jsonl', 'r', encoding='utf-8') as f:
#         val_list = [json.loads(line) for line in f]
#     train_dataset = Dataset.from_list(train_list)
#     eval_dataset  = Dataset.from_list(val_list)
#
#     # ===== 3. 预处理函数 =====
#     model_name = "Salesforce/blip2-flan-t5-xl"
#     processor = BlipProcessor.from_pretrained(model_name)
#     model = Blip2ForConditionalGeneration.from_pretrained(model_name)
#
#     # def preprocess_function(example):
#     #     image = Image.open(example["image"]).convert("RGB")
#     #     inputs = processor(
#     #         images=image,
#     #         text=example["text"],
#     #         padding="max_length",
#     #         truncation=True,
#     #         max_length=32,
#     #         return_tensors="pt"
#     #     )
#     #     inputs = {k: v.squeeze(0) for k, v in inputs.items()}
#     #     return inputs
#
#     def preprocess_function(example):
#         image = Image.open(example["image"]).convert("RGB")
#         # 你可以自定义prompt，比如caption任务建议用空字符串
#         prompt = ""
#         inputs = processor(
#             images=image,
#             text=prompt,
#             padding="max_length",
#             truncation=True,
#             max_length=32,
#             return_tensors="pt"
#         )
#         labels = processor.tokenizer(
#             example["text"],
#             padding="max_length",
#             truncation=True,
#             max_length=32,
#             return_tensors="pt"
#         ).input_ids
#         labels[labels == processor.tokenizer.pad_token_id] = -100
#         inputs = {k: v.squeeze(0) for k, v in inputs.items()}
#         inputs["labels"] = labels.squeeze(0)
#         return inputs
#
#     train_dataset = train_dataset.map(preprocess_function)
#     eval_dataset  = eval_dataset.map(preprocess_function)
#
#     # ===== 4. 训练参数设置 =====
#     training_args = TrainingArguments(
#         output_dir="./blip2_captioning",
#         per_device_train_batch_size=2,
#         gradient_accumulation_steps=8,
#         num_train_epochs=3,
#         learning_rate=5e-5,
#         weight_decay=0.01,
#         fp16=True,
#         logging_steps=10,
#         save_steps=200,
#         save_total_limit=2,
#         remove_unused_columns=False,
#         overwrite_output_dir=True,
#         dataloader_num_workers=2,
#         seed=42,
#         gradient_checkpointing=True,
#         push_to_hub=False,
#         report_to="tensorboard",
#         logging_dir="./blip2_captioning/tensorboard"
#     )
#
#     # ===== 5. 构建 Trainer =====
#     trainer = CustomTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         tokenizer=processor,
#     )
#
#     # ===== 6. 启动训练 =====
#     trainer.train()
#
#     # ===== 7. 保存最终模型 =====
#     trainer.save_model("./blip2_captioning/final_model")
#     print("训练完成，模型已保存到 ./blip2_captioning/final_model")
#     print("\n你可以通过命令行运行如下命令，实时监控训练过程：")
#     print("tensorboard --logdir ./blip2_captioning/tensorboard")
#     print("浏览器访问 http://localhost:6006")
#
# if __name__ == "__main__":
#     main()



# import os
# import json
# from PIL import Image
# from datasets import load_dataset
# from datasets import Dataset
# from transformers import BlipProcessor, Blip2ForConditionalGeneration, TrainingArguments, Trainer
# from sklearn.model_selection import train_test_split
# import torch
#
# # ===== 1. 数据集划分：80%训练，20%验证 =====
#
# # 原始标注文件路径
# data_path = "annotations.json"
#
# # 读取全部标注数据为列表
# with open(data_path, 'r', encoding='utf-8') as f:
#     data_list = json.load(f)
#
# # 划分训练集和验证集（random_state保证复现）
# train_list, val_list = train_test_split(data_list, test_size=0.2, random_state=42)
#
# # 将划分后的数据写入新json文件
# with open('train_split.json', 'w', encoding='utf-8') as f:
#     json.dump(train_list, f, ensure_ascii=False, indent=2)
# with open('val_split.json', 'w', encoding='utf-8') as f:
#     json.dump(val_list, f, ensure_ascii=False, indent=2)
#
# # ===== 2. 加载数据集 =====
#
# # train_dataset = load_dataset('json', data_files='train_split.json', split='train')
# # eval_dataset  = load_dataset('json', data_files='val_split.json', split='train')
#
# # train_dataset = load_dataset('json', data_files='train_split.jsonl', split='train')
# # eval_dataset  = load_dataset('json', data_files='val_split.jsonl', split='train')
#
# with open('train_split.jsonl', 'r', encoding='utf-8') as f:
#     train_list = [json.loads(line) for line in f]
# with open('val_split.jsonl', 'r', encoding='utf-8') as f:
#     val_list = [json.loads(line) for line in f]
#
# train_dataset = Dataset.from_list(train_list)
# eval_dataset  = Dataset.from_list(val_list)
#
# # ===== 3. 预处理函数：转为模型输入格式 =====
#
# model_name = "Salesforce/blip2-flan-t5-xl"
# processor = BlipProcessor.from_pretrained(model_name)
# model = Blip2ForConditionalGeneration.from_pretrained(model_name)
#
# def preprocess_function(example):
#     image = Image.open(example["image"]).convert("RGB")
#     # 处理图片与目标描述文本
#     inputs = processor(
#         images=image,
#         text=example["text"],
#         padding="max_length",
#         truncation=True,
#         max_length=32,             # 输出描述最大长度
#         return_tensors="pt"
#     )
#     # squeeze: 降为一维张量，兼容trainer
#     inputs = {k: v.squeeze(0) for k, v in inputs.items()}
#     return inputs
#
# # 批量处理数据集
# train_dataset = train_dataset.map(preprocess_function)
# eval_dataset  = eval_dataset.map(preprocess_function)
#
# # ===== 4. 训练参数设置（已为8G显存优化，含tensorboard） =====
#
# training_args = TrainingArguments(
#     output_dir="./blip2_captioning",              # 训练过程模型权重保存路径
#     per_device_train_batch_size=2,                # 单卡batch大小
#     gradient_accumulation_steps=8,                # 累积步数
#     num_train_epochs=3,                           # 训练轮次
#     learning_rate=5e-5,                           # 学习率
#     weight_decay=0.01,                            # 权重衰减
#     fp16=True,                                    # 使用混合精度
#     logging_steps=10,                             # 每10步记录一次日志
#     save_steps=200,                               # 每200步保存一次模型
#     save_total_limit=2,                           # 只保留最近2个checkpoint
#     remove_unused_columns=False,                  # 必须为False，防止丢失输入
#     overwrite_output_dir=True,                    # 若output_dir存在则覆盖
#     dataloader_num_workers=2,                     # 数据加载线程数
#     seed=42,                                      # 随机种子
#     gradient_checkpointing=True,                  # 省显存
#     push_to_hub=False,                            # 不上传Hub
#     report_to="tensorboard",                      # 日志输出到tensorboard
#     logging_dir="./blip2_captioning/tensorboard"  # tensorboard日志目录
# )
#
# # ===== 5. 构建 Trainer =====
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,    # 加入验证集
#     tokenizer=processor,
# )
#
# # ===== 6. 启动训练 =====
#
# trainer.train()
#
# # ===== 7. 保存最终模型 =====
#
# trainer.save_model("./blip2_captioning/final_model")
# print("训练完成，模型已保存到 ./blip2_captioning/final_model")
#
# print("\n你可以通过命令行运行如下命令，实时监控训练过程：")
# print("tensorboard --logdir ./blip2_captioning/tensorboard")
# print("浏览器访问 http://localhost:6006")
