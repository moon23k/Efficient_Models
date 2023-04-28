import os, argparse, torch
from module.test import Tester
from module.train import Trainer
from module.model import load_model
from module.data import load_dataloader
from transformers import set_seed, AutoTokenizer



class Config(object):
    def __init__(self, args):

        mname_dict = {'bert': 'bert-base-uncased',
                      'albert': "albert-base-v2",
                      'distil_bert': "distilbert-base-uncased", 
                      'mobile_bert': "google/mobilebert-uncased",
                      'longformer': "allenai/longformer-base-4096",
                      'bigbird': "google/bigbird-roberta-base"}

        self.mode = args.mode
        self.model_type = args.model
        self.mname = mname_dict[self.model_type]
        self.ckpt = f"ckpt/{self.model_type}.pt"

        self.n_epochs = 10
        self.batch_size = 32
        self.learning_rate = 5e-5
        self.gradient_accumulation_steps = 4

        self.early_stop = True
        self.patience = 3

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")




def main(args):
    set_seed(42)
    config = Config(args)
    model = load_model(config)
    tokenizer = AutoTokenizer.from_pretrained(config.mname)


    if args.mode == 'train':
        train_dataloader = load_dataloader(config, tokenizer, 'train')
        valid_dataloader = load_dataloader(config, tokenizer, 'valid')
        trainer = Trainer(config, model, train_dataloader, valid_dataloader) 
        trainer.train()


    elif args.mode == 'test':
        test_dataloader = load_dataloader(config, tokenizer, 'test')
        tester = Tester(config, model, test_dataloader)
        tester.test()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    parser.add_argument('-model', required=True)

    args = parser.parse_args()
    assert args.mode.lower() in ['train', 'test', 'inference']
    assert args.model.lower() in ['bert', 'albert', 'distil_bert', 
                                  'mobile_bert', 'longformer', 'bigbird']

    main(args)