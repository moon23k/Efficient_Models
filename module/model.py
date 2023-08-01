import torch
import torch.nn as nn
from collections import namedtuple

from transformers import AutoModel, AutoConfig

from pynvml import (
    nvmlInit, 
    nvmlDeviceGetHandleByIndex, 
    nvmlDeviceGetMemoryInfo
)



class Classifier(nn.Module):
    def __init__(self, config, plm):
        self.plm = plm
        self.device = config.device
        self.classifier = nn.Linear(plm.hidden_size, 1)
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.out = namedtuple('Out', 'logit loss')


    def forward(self, input_ids, attention_mask, labels):
        
        last_hiddens = self.plm(
            input_ids=input_ids, 
            attention_mask=attention_mask
        ).last_hidden_state
        
        logits = self.classifier(last_hiddens)
        loss = self.criterion(logits, labels)

        return self.out(logits, loss)



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

    #GPU Memory Occupations
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"--- GPU memory occupied: {info.used//1024**2} MB\n")



def load_model(config):
    if config.mode == 'train':
        model = AutoModel.from_pretrained(config.mname)
        print(f"\nPretrained {config.model_type.upper()} Model has loaded")

    print_model_desc(model)
    return model.to(config.device)