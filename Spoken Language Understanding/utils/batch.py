#-*- coding:utf-8 -*-
import torch


def from_example_list(args, ex_list, device='cpu', train=True):
    ex_list = sorted(ex_list, key=lambda x: len(x.input_idx), reverse=True)
    batch = Batch(ex_list, device)
    pad_idx = args.pad_idx
    tag_pad_idx = args.tag_pad_idx
    batch.utt = [ex.utt for ex in ex_list]
    input_lens = [len(ex.input_idx) for ex in ex_list]
    max_len = max(input_lens)
    input_ids = [ex.input_idx + [pad_idx] * (max_len - len(ex.input_idx)) for ex in ex_list]
    batch.input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    batch.lengths = input_lens

    if train:
        batch.labels = [ex.slotvalue for ex in ex_list]
        tag_lens = [len(ex.tag_id) for ex in ex_list]
        max_tag_lens = max(tag_lens)
        tag_ids = [ex.tag_id + [tag_pad_idx] * (max_tag_lens - len(ex.tag_id)) for ex in ex_list]
        tag_mask = [[1] * len(ex.tag_id) + [0] * (max_tag_lens - len(ex.tag_id)) for ex in ex_list]
        batch.tag_ids = torch.tensor(tag_ids, dtype=torch.long, device=device)
        batch.tag_mask = torch.tensor(tag_mask, dtype=torch.float, device=device)
    else:
        batch.labels = None
        batch.tag_ids = None
        batch.tag_mask = None

    return batch


def from_example_list_Bert(args, ex_list, device='cpu', train=True):
    ex_list = sorted(ex_list, key=lambda x: len(x.input_idx), reverse=True)
    batch = Batch(ex_list, device)
    pad_idx = args.pad_idx
    tag_pad_idx = args.tag_pad_idx

    batch.utt = [ex.utt for ex in ex_list]
    input_lens = [len(ex.input_idx) for ex in ex_list]
    max_len = max(input_lens)
    input_ids = [ex.input_idx + [pad_idx] * (max_len - len(ex.input_idx)) for ex in ex_list]
    batch.input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    attention_mask = [ex.attention_mask + [pad_idx] * (max_len - len(ex.attention_mask)) for ex in ex_list]
    batch.attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)
    token_type_ids = [ex.token_type_ids + [pad_idx] * (max_len - len(ex.token_type_ids)) for ex in ex_list]
    batch.token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=device)
    batch.lengths = input_lens

    if train:
        batch.labels = [ex.slotvalue for ex in ex_list]
        tag_lens = [len(ex.tag_id) for ex in ex_list]
        max_tag_lens = max_len - 2
        tag_ids = [[0] + ex.tag_id + [tag_pad_idx] * (max_tag_lens - len(ex.tag_id)) + [0] for ex in ex_list]
        tag_mask = [[0] + [1] * len(ex.tag_id) + [0] * (max_tag_lens - len(ex.tag_id)) + [0] for ex in ex_list]
        batch.tag_ids = torch.tensor(tag_ids, dtype=torch.long, device=device)
        batch.tag_mask = torch.tensor(tag_mask, dtype=torch.float, device=device)
    else:
        batch.labels = None
        batch.tag_ids = None
        batch.tag_mask = None

    return batch


def from_example_list_predict(args, ex_list, device='cpu', train=True):
    #ex_list = sorted(ex_list, key=lambda x: len(x.input_idx), reverse=True)
    batch = Batch(ex_list, device)
    pad_idx = args.pad_idx
    tag_pad_idx = args.tag_pad_idx

    batch.utt = [ex.utt for ex in ex_list]
    input_lens = [len(ex.input_idx) for ex in ex_list]
    max_len = max(input_lens)
    input_ids = [ex.input_idx + [pad_idx] * (max_len - len(ex.input_idx)) for ex in ex_list]
    batch.input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    attention_mask = [ex.attention_mask + [pad_idx] * (max_len - len(ex.attention_mask)) for ex in ex_list]
    batch.attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)
    token_type_ids = [ex.token_type_ids + [pad_idx] * (max_len - len(ex.token_type_ids)) for ex in ex_list]
    batch.token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=device)
    batch.lengths = input_lens

    if train:
        batch.labels = [ex.slotvalue for ex in ex_list]
        tag_lens = [len(ex.tag_id) for ex in ex_list]
        max_tag_lens = max_len - 2
        tag_ids = [[0] + ex.tag_id + [tag_pad_idx] * (max_tag_lens - len(ex.tag_id)) + [0] for ex in ex_list]
        tag_mask = [[0] + [1] * len(ex.tag_id) + [0] * (max_tag_lens - len(ex.tag_id)) + [0] for ex in ex_list]
        batch.tag_ids = torch.tensor(tag_ids, dtype=torch.long, device=device)
        batch.tag_mask = torch.tensor(tag_mask, dtype=torch.float, device=device)
    else:
        batch.labels = None
        batch.tag_ids = None
        batch.tag_mask = None

    return batch


class Batch():

    def __init__(self, examples, device):
        super(Batch, self).__init__()

        self.examples = examples
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]