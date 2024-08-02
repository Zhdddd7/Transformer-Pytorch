import spacy
from datasets import load_dataset
import torch

# 加载Spacy模型
try:
    spacy_de = spacy.load('de_core_news_sm')
except OSError:
    from spacy.cli import download
    download('de_core_news_sm')
    spacy_de = spacy.load('de_core_news_sm')

try:
    spacy_en = spacy.load('en_core_web_sm')
except OSError:
    from spacy.cli import download
    download('en_core_web_sm')
    spacy_en = spacy.load('en_core_web_sm')

# 定义tokenizer
def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize(example):
    example['de'] = tokenize_de(example['de'])
    example['en'] = tokenize_en(example['en'])
    return example

def load_data(batch_size, device):
    # 加载WMT14德语到英语的数据集
    raw_datasets = load_dataset('wmt14', 'de-en')

    # 提取需要的列
    raw_datasets = raw_datasets.map(lambda example: {'de': example['translation']['de'], 'en': example['translation']['en']})

    # Tokenize the dataset
    tokenized_datasets = raw_datasets.map(tokenize)

    # 获取词汇表
    SRC = tokenized_datasets['train']['de']
    TRG = tokenized_datasets['train']['en']

    def data_iterator(data, batch_size):
        for i in range(0, len(data), batch_size):
            src_batch = data['de'][i:i + batch_size]
            trg_batch = data['en'][i:i + batch_size]
            yield torch.tensor(src_batch, dtype=torch.long, device=device), torch.tensor(trg_batch, dtype=torch.long, device=device)

    train_iterator = data_iterator(tokenized_datasets['train'], batch_size)
    valid_iterator = data_iterator(tokenized_datasets['validation'], batch_size)
    test_iterator = data_iterator(tokenized_datasets['test'], batch_size)

    return train_iterator, valid_iterator, test_iterator, SRC, TRG
