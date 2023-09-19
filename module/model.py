import torch
import torch.nn as nn
from transformers import (
    AutoConfig, 
    AutoModelForSequenceClassification
)




def print_model_desc(model):
    #Number of trainerable parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--- Model Params: {n_params:,}")

    #Model size check
    param_size, buffer_size = 0, 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"--- Model  Size : {size_all_mb:.3f} MB")



def freeze_pretrained_params(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True



def load_model(config):
    if config.mode == 'train':
        
        model = AutoModelForSequenceClassification.from_pretrained(
            config.mname, num_labels=config.num_labels
        )
        print(f"\nPretrained {config.model_type.upper()} Model has loaded")


        #Extend BERT's model max_length for imdb task
        if config.task == 'imdb' and config.model_type == 'bert':
            embeddings = model.bert.embeddings

            max_len = config.max_len
            temp_emb = nn.Embedding(max_len, model.config.hidden_size)
            temp_emb.weight.data[:512] = embeddings.position_embeddings.weight.data
            temp_emb.weight.data[512:] = embeddings.position_embeddings.weight.data[-1][None,:].repeat(max_len-512, 1)

            model.bert.embeddings.position_embeddings = temp_emb

            model.config.max_position_embeddings = max_len
            model.bert.embeddings.position_ids = torch.arange(max_len).expand((1, -1))
            model.bert.embeddings.token_type_ids = torch.zeros(max_len, dtype=torch.long).expand((1, -1))        

        #save model config
        model.config.save_pretrained(config.model_config_path.replace('config.json', '.'))
        
        #Freeze pretrained model params
        freeze_pretrained_params(model)


    #Load FineTuned model states
    if config.mode == 'test':
        model_config = AutoConfig.from_pretrained(config.model_config_path)
        model = AutoModelForSequenceClassification.from_config(model_config)

        model_state = torch.load(
            config.ckpt, 
            map_location=config.device
        )['model_state_dict']
        
        model.load_state_dict(model_state)
        print(f"FineTuned Model States have loaded from {config.ckpt}")

    print_model_desc(model)
    return model.to(config.device)