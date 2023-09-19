import torch



class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader):
        super(Tester, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = test_dataloader

        self.task = config.task
        self.bos_id = config.bos_id
        self.device = config.device
        self.model_type = config.model_type
        
                
    def test(self):
        score = 0.0         
        self.model.eval()
        torch.compile(self.model)

        with torch.no_grad():
            for batch in self.dataloader:
                x = batch['x'].to(self.device)
                y = batch['y']

                pred = self.model(x)
                score += self.evaluate(pred, y)

        txt = f"TEST Result on {self.task.upper()} with {self.model_type.upper()} model"
        txt += f"\n-- Score: {round(score/len(self.dataloader), 2)}\n"
        print(txt)
