from datasets import load_dataset
from torch.utils.data import IterableDataset
import torch


class StreamingDataset(IterableDataset):
    def __init__(self, split, tokenizer, max_seq_len: int, stride: int = None):
        self.iterable = split
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride or max_seq_len
        self.pad_id = tokenizer.pad_id()
        self.bos_id = tokenizer.bos_id()
        self.eos_id = tokenizer.eos_id()

    def __iter__(self):
        for ex in self.iterable:
            text = ex["text"]
            if not text or not isinstance(text, str):
                continue

            ids = self.tokenizer.EncodeAsIds(text)
            if len(ids) < 2:
                continue

            # BOS와 EOS 추가
            ids = [self.bos_id] + ids + [self.eos_id]
            L = len(ids)

            if L <= self.max_seq_len:
                input_ids = ids[:-1] + [self.pad_id] * (self.max_seq_len - L + 1)
                labels = ids[1:] + [self.pad_id] * (self.max_seq_len - L + 1)
                yield {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            else:
                for start in range(0, L - self.max_seq_len + 1, self.stride):
                    window = ids[start:start + self.max_seq_len]
                    input_ids = window[:-1]
                    labels = window[1:]
                    padding_length = self.max_seq_len - len(input_ids)
                    if padding_length > 0:
                        input_ids += [self.pad_id] * padding_length
                        labels += [self.pad_id] * padding_length

                    yield {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "labels": torch.tensor(labels, dtype=torch.long),
                    }