import json, argparse, torch
from transformers import set_seed, AutoTokenizer
from module import (
    load_model, 
    load_dataset, 
    set_trainer 
)




class Config(object):
    def __init__(self, args):

        mname_dict = {
            #Base Line Model
            'bert': 'bert-base-uncased',
            
            #Lightened Models
            'albert': "albert-base-v2",
            'distil_bert': "distilbert-base-uncased", 
            'mobile_bert': "google/mobilebert-uncased",

            #Sparse Attention Models 
            'reformer': 'google/reformer-enwik8',
            'longformer': "allenai/longformer-base-4096",
            'bigbird': "google/bigbird-roberta-base"
        }


        self.task = args.task
        self.model_type = args.model
        self.mname = mname_dict[self.model_type]

        self.ckpt = f'ckpt/{self.task}/{self.model_type}'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.lr = 1e-5
        self.n_epochs = 5
        self.batch_size = 32

        if self.task == 'imdb':
            self.num_labels = 2
            self.max_len = 4096
        elif self.task == 'ag_news':
            self.num_labels = 4
            self.max_len = 512
        


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



def main(strategy):

    #Prerequisites
    set_seed(42)
    config = Config(strategy)
    model = load_model(config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.mname, model_max_length=config.max_len
    )


    #Load datasets
    train_dataset = load_dataset(tokenizer, 'train')
    valid_dataset = load_dataset(tokenizer, 'valid')
    test_dataset = load_dataset(tokenizer, 'test')


    #Load Trainer
    trainer = set_trainer(config, model, tokenizer, train_dataset, valid_dataset)    
    

    #Training
    torch.cuda.reset_max_memory_allocated()
    train_output = trainer.train()
    gpu_memory = torch.cuda.max_memory_allocated()
    

    #Evaluating
    eval_output = trainer.evaluate(test_dataset)

    
    #Save Training and Evaluation Rst Report
    report = {**train_output.metrics, **eval_output}
    report['gpu_memory'] = f"{gpu_memory / (1024 ** 3):.2f} GB"
    report['model_params'], report['model_size'] = get_model_desc(model)

    os.makedirs(f"report/{config.task}", exist_ok=True)
    with open(f"report/{config.task}/{config.model_type}.json", 'w') as f:
        json.dump(report, f)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    parser.add_argument('-model', required=True)

    args = parser.parse_args()
    assert args.task.lower() in ['imdb', 'ag_news']
    assert args.model.lower() in ['bert', 
        'albert', 'distil_bert', 'mobile_bert', 
        'reformer', 'longformer', 'bigbird'
    ]

    main(args)