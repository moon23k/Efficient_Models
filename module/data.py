import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformer import AutoTokenizer



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




def load_dataloader(config, tokenzier, split):
    pad_id = config.pad_id
    max_len = config.max_len

    def collate_fn(batch):
        text_batch, label_batch = [], []

        for text, label in batch:
            text_batch.append(text) 
            label_batch.append(label)

        text_encodings = tokenzier(text_batch, 
        						   max_length=max_len,
        						   padding='max_length', 
        						   truncation=True,
 								   return_tensors='pt')

        return {'input_ids': text_encodings.input_ids, 
                'attention_mask': text_encodings.attention_mask,
                'labels': torch.Tensor(labels_batch)}

    return DataLoader(Dataset(config.task, split), 
                      batch_size=config.batch_size, 
                      shuffle=True,
                      collate_fn=collate_fn,
                      num_workers=2,
                      pin_memory=True)		