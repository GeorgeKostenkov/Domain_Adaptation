from tqdm import tqdm
import torch
import transformers as ts
from datasets import load_dataset, load_metric
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import os

from others.utils import pad_sents, save, get_mask, fix_random_seed

def tokenizing_fn(instances, tokenizer):
    encoded = tokenizer(instances["document"], truncation=True,  add_special_tokens=False)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(instances["summary"], truncation=True,  add_special_tokens=False)

    encoded["labels"] = labels["input_ids"]
    return encoded

class BartDataset(Dataset):
    '''
    Attributes:
        src: it's a list, each line is a sample for source text.
        tgt: it's a list, each line is a sample for target text.
        src_ids: it's a list, each line is a sample for source index after tokenized.
        tgt_ids: it's a list, each line is a sample for target index after tokenized.
    '''
    def __init__(self, tokenizer, source, target, data):
        self.tokenizer = tokenizer
        temp = data.map(lambda x : tokenizing_fn(x, tokenizer), batched=True).remove_columns(['document', 'summary'])
        self.src_ids = temp['input_ids']
        self.tgt_ids = temp['labels']

    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, idx):
        return self.src_ids[idx], self.tgt_ids[idx]

    def tokenize(self, data):
        tokenized_text = [self.tokenizer.encode(i, add_special_tokens=False) for i in tqdm(data)]
        return tokenized_text

    def collate_fn(self, data):
        # rebuld the raw text and truncate to max length
        max_input_len = 1024
        max_output_len = 256
        raw_src = [pair[0] for pair in data]
        raw_tgt = [pair[1] for pair in data]
        raw_src = [i[:max_input_len-1] for i in raw_src]
        raw_tgt = [i[:max_output_len-1] for i in raw_tgt]
        src = []
        tgt = []
        # remove blank data
        for i in range(len(raw_src)):
            if (raw_src[i] != []) and (raw_tgt[i] != []):
                src.append(raw_src[i])
                tgt.append(raw_tgt[i])
        # make input mask
        mask = torch.tensor(get_mask(src, max_len=max_input_len))
        # make input ids
        src_ids = torch.tensor(pad_sents(src, 1, max_len=max_input_len)[0])
        # make output ids
        decoder_ids = [[0]+i for i in tgt]
        # make output labels
        label_ids = [i+[2] for i in tgt]
        decoder_ids = torch.tensor(pad_sents(decoder_ids, 1, max_len=max_output_len)[0])
        label_ids = torch.tensor(pad_sents(label_ids, -100, max_len=max_output_len)[0])

        return src_ids, decoder_ids, mask, label_ids


def data_builder(args):
    if args.data_name == 'ccdv/cnn_dailymail':
        data = load_dataset(args.data_name, '3.0.0', cache_dir='cache')[args.mode].train_test_split(train_size=args.train_size, shuffle=True, seed=42)['train']
        args.data_name = 'cnn_dailymail'
    else:
        data = load_dataset(args.data_name, cache_dir='cache')[args.mode].train_test_split(train_size=args.train_size, shuffle=True, seed=42)['train']
    source = data[list(data.features.keys())[0]]
    target = data[list(data.features.keys())[1]]
    tokenizer = ts.AutoTokenizer.from_pretrained('facebook/bart-base', cache_dir='cache/tokenizer')
    train_set = BartDataset(tokenizer, source, target, data)
    if args.mode == 'train':
        data_loader = DataLoader(dataset=train_set,
                                batch_size=args.batch_size,
                                shuffle=True,
                                collate_fn=train_set.collate_fn)
    else:
        data_loader = DataLoader(dataset=train_set,
                                batch_size=args.batch_size,
                                shuffle=False,
                                collate_fn=train_set.collate_fn)
    save(data_loader, 'loaders/' + args.data_name + '_' + str(args.train_size) + '_' + args.mode +'.pt')
#     a,b,c,d = next(iter(data_loader))
#     print(a)
#     print(b)
#     print(c)
#     print(d)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_name', default='debate', type=str)
    parser.add_argument('-mode', default='train', type=str)
    parser.add_argument('-train_size', default=500, type=int)
    parser.add_argument('-batch_size', default=4, type=int)
    parser.add_argument('-random_seed', type=int, default=0)
    args = parser.parse_args()

    # set random seed
    fix_random_seed(args.random_seed)
    data_builder(args)

