import torch, time
from tqdm import tqdm


class Tester:
    def __init__(self, config, model, test_dataloader):
        super(Tester, self).__init__()
        
        self.model = model
        self.device = config.device
        self.dataloader = test_dataloader


    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"


    def test(self):
        self.model.eval()
        acc_score, data_volumn = 0, 0

        start_time = time.time()
        with torch.no_grad():
            for batch in tqdm(self.dataloader):   
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                data_volumn += input_ids.size(0)

                preds = self.model(input_ids, attention_mask)
                acc_score += torch.sum(preds == labels)
        
        acc_score /= data_volumn

        print('Test Results')
        print(f"  >> BLEU Score: {acc_score:.2f}")
        print(f"  >> Spent Time: {self.measure_time(start_time, time.time())}")