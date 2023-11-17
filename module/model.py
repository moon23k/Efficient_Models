import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification




def get_model_desc(model):
    #Number of trainerable parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    #Model size check
    param_size, buffer_size = 0, 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2

    return f"{n_params:,}", f"{size_all_mb:.3f} MB"
    



def load_model(config):
    model = AutoModelForSequenceClassification.from_pretrained(
        config.mname, num_labels=config.num_labels
    )

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

    if config.task == 'imdb':
        model.config.use_cache = False  #For Gradient Checkpointing

    return model.to(config.device)
