import json

for split in ['train_split', 'val_split']:
    with open(f"{split}.json", 'r', encoding='utf-8') as fin, \
         open(f"{split}.jsonl", 'w', encoding='utf-8') as fout:
        data = json.load(fin)
        for item in data:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

print('转换完毕，得到 train_split.jsonl 和 val_split.jsonl')
