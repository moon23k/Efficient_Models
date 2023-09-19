import os, argparse, torch
from transformers import set_seed, AutoTokenizer
from module import (
    load_model, 
    load_dataloader, 
    load_trainer, 
    Tester
)



class Config(object):
    def __init__(self, args):

        mname_dict = {
            'bert': 'bert-base-uncased',
            'albert': "albert-base-v2",
            'distil_bert': "distilbert-base-uncased", 
            'mobile_bert': "google/mobilebert-uncased",
            'reformer': "google/reformer-enwik8",
            'longformer': "allenai/longformer-base-4096",
            'bigbird': "google/bigbird-roberta-base"
        }

        self.task = args.task
        self.mode = args.mode
        self.model_type = args.model
        self.mname = mname_dict[self.model_type]

        self.lr = 5e-5
        self.n_epochs = 10
        self.batch_size = 32
        self.gradient_accumulation_steps = 4

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        if self.task == 'imdb':
            self.problem_type = "single_label_classification"
            self.num_labels = 2
            self.max_len = 512
        elif self.task == 'ag_news':
            self.problem_type = "multi_label_classification"
            self.num_labels = 4
            self.max_len = 4096
        
    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



def main(args):
    set_seed(42)
    config = Config(args)
    model = load_model(config)
    tokenizer = AutoTokenizer.from_pretrained(config.mname)
    torch.cuda.empty_cache()

    if mode == 'train':
        trainer = load_trainer(config, model, tokenizer)
        trainer.train()
    elif mode == 'test':
        tester = Tester()
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
        'bert', 'albert', 'distil_bert', 'mobile_bert', 
        'reformer', 'longformer', 'bigbird'
    ]

    main(args)