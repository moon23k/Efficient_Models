import json, torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        super().__init__()
        self.data = self.load_data(split)

    @staticmethod
    def load_data(split):
        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]['text'], self.data[idx]['label']


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_len = tokenizer.model_max_length

    def __call__(self, batch):
        text_batch, label_batch = [], []

        for text, label in batch:
            text_batch.append(text) 
            label_batch.append(label)

        text_encodings = self.tokenzier(text_batch, 
                                        max_length=self.max_len,
                                        padding='max_length', 
                                        truncation=True,
                                        return_tensors='pt')

        return {'input_ids': text_encodings.input_ids, 
                'attention_mask': text_encodings.attention_mask,
                'labels': torch.Tensor(label_batch)}


def load_dataloader(config, tokenizer, split):
    return DataLoader(Dataset(split), 
                      batch_size=config.batch_size, 
                      shuffle=True,
                      collate_fn=Collator(tokenizer),
                      num_workers=2, pin_memory=True)       