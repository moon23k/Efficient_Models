import os, argparse, torch
from transformers import set_seed, AutoTokenizer
from module import (
    load_model, 
    load_dataloader, 
    Trainer, 
    Tester
)



class Config(object):
    def __init__(self, args):

        mname_dict = {
            'bert': 'bert-base-uncased',
            'albert': "albert-base-v2",
            'distil_bert': "distilbert-base-uncased", 
            'mobile_bert': "google/mobilebert-uncased",
            'longformer': "allenai/longformer-base-4096",
            'bigbird': "google/bigbird-roberta-base"
        }

        self.task = args.task
        self.mode = args.mode
        self.model_type = args.model
        self.mname = mname_dict[self.model_type]

        self.lr = 5e-5
        self.n_epochs = 10
        self.batch_size = 4
        self.iters_to_accumulate = 4
        self.clip = 1
        self.early_stop = True
        self.patience = 3

        use_cuda = torch.cuda.is_available()
        self.device_type = 'cuda' if use_cuda else 'cpu'
        self.device = torch.device(self.device_type)
        self.ckpt = f'ckpt/{self.task}/{self.model_type}/model.pt'
        self.model_config_path = self.ckpt.replace('model.pt', 'config.json')

        if self.task == 'imdb':
            self.num_labels = 2
            self.max_len = 4096
        elif self.task == 'ag_news':
            self.num_labels = 4
            self.max_len = 512
        

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



def main(args):
    set_seed(42)
    config = Config(args)
    model = load_model(config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.mname, model_max_length=config.max_len
    )
    
    torch.cuda.empty_cache()

    if config.mode == 'train':
        train_dataloader = load_dataloader(config, tokenizer, 'train')
        valid_dataloader = load_dataloader(config, tokenizer, 'valid')
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()

    elif config.mode == 'test':
        test_dataloader = load_dataloader(config, tokenizer, 'test')
        tester = Tester(config, model, test_dataloader)
        tester.test()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    parser.add_argument('-mode', required=True)
    parser.add_argument('-model', required=True)

    args = parser.parse_args()
    assert args.task.lower() in ['imdb', 'ag_news']
    assert args.mode.lower() in ['train', 'test']
    assert args.model.lower() in [
        'bert', 'albert', 'distil_bert', 
        'mobile_bert', 'longformer', 'bigbird'
    ]

    os.makedirs(f'ckpt/{args.task}/{args.model}', exist_ok=True)

    main(args)