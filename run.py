import os, yaml, argparse, torch

from module.data import load_dataloader
from module.train import Trainer
from module.test import Tester

from transformers import (set_seed, 
						  AutoModel,
						  AutoTokenizer)

from pynvml import (nvmlInit, 
                    nvmlDeviceGetHandleByIndex, 
                    nvmlDeviceGetMemoryInfo)



def load_model(config):
	model = AutoModel.from_pretrained(config.m_name)

	if config.m_type == 'transformer':
		model = model.encoder

	return model.to(config.device)


class Config(object):
    def __init__(self, args):
        self.mode = args.mode
        self.model = args.model

        self.n_epochs = 1
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.gradient_accumulation_steps = 4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")


def print_memory():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")




def main(args):
    set_seed(42)
    config = Config(args)
    model = load_model(config)
    tokenizer = load_tokenizer(config)


    if args.mode == 'train':
        train_datalaoder = load_dataloader(config, tokenizer, 'train')
        valid_datalaoder = load_dataloader(config, tokenizer, 'valid')
        trainer = Trainer(config, model, tokenizer, train_dataloader, valid_datalaoder) 
        trainer.train()


    elif args.mode == 'test':
        test_dataloader = load_dataloader(config, tokenizer, 'test')
        tester = Tester(config, model, tokenizer, test_dataloader)
        tester.test()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    parser.add_argument('-model', required=True)

    args = parser.parse_args()
    assert args.mode.lower() in ['train', 'test']
    assert args.mode.lower() in ['transformer', 'bert', 'albert', 'distil', 'mobile'
                         		 'transformer_xl', 'reformer', 'longformer', 'bigbird']

    main(args)