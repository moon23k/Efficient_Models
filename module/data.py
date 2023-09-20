import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence




class Dataset(torch.utils.data.Dataset):
    def __init__(self, task, split):
        super().__init__()
        self.data = self.load_data(task, split)

    @staticmethod
    def load_data(task, split):
        with open(f"data/{task}/{split}.json", 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]['x'], self.data[idx]['y']



class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


    def __call__(self, batch):
        x_batch, y_batch = zip(*batch)
        x_encodings = self.tokenizer(
            x_batch, 
            padding=True, 
            truncation=True,
            return_tensors='pt'
        )
        
        return {'input_ids': x_encodings.input_ids,
                'attention_mask': x_encodings.attention_mask,
                'labels': torch.LongTensor(y_batch)}
        


def load_dataloader(config, tokenizer, split):

    return DataLoader(
        Dataset(config.task, split), 
        batch_size=config.batch_size, 
        shuffle=True,
        collate_fn=Collator(tokenizer),
        num_workers=2, 
        pin_memory=True
    )