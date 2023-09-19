from module import load_dataloader
from transformers import TrainingArguments, Trainer



def set_training_args(config):

    train_kwargs = {
        'output_dir': config.ckpt,
        'logging_dir': f'{config.ckpt}/logging',

        'learning_rate': config.lr,
        'num_train_epochs': config.n_epochs,
        'use_cpu': config.device.type == 'cpu',
        'per_device_eval_batch_size': config.batch_size,
        'per_device_train_batch_size': config.batch_size,
        'gradient_accumulation_steps': config.gradient_accumulation_steps,

        'fp16': True,
        'fp16_opt_level': '02',

        'logging_first_step': True,
        'logging_strategy': 'epoch',
        'save_strategy': 'epoch',
        'evaluation_strategy': 'epoch',
    }

    return TrainingArguments(**train_kwargs)



def load_trainer(config, model, tokenizer):
    training_args = set_training_args(config)

    train_dataloader = load_dataloader(config, tokenizer, 'train')
    valid_dataloader = load_dataloader(config, tokenizer, 'valid')

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader,
        eval_dataset=valid_dataloader
    )

    return trainer