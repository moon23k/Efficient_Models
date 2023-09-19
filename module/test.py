import torch



class Tester:
    def __init__(self, config, model, test_dataloader):
        super(Tester, self).__init__()
        
        self.model = model
        self.dataloader = test_dataloader

        self.task = config.task
        self.device = config.device
        self.model_type = config.model_type
        
                
    def test(self):
        score = 0.0         
        self.model.eval()
        torch.compile(self.model)

        with torch.no_grad():
            for batch in self.dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']

                preds = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask
                ).logits..argmax(dim=-1)

                acc = (preds == labels).sum().item()
                score += acc

        txt = f"TEST Result on {self.task.upper()} with {self.model_type.upper()} model"
        txt += f"\n-- Acc Score: {round(score/len(self.dataloader), 2)}\n"
        print(txt)
